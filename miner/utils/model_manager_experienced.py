from pathlib import Path
from typing import Dict, Optional, Tuple, List

from ultralytics import YOLO
from loguru import logger

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # Initializes CUDA driver
import numpy as np

from miner.utils.device import get_optimal_device
from scripts.download_models import download_models

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

# --------------------- INT8 Calibrator ---------------------
class Int8Calibrator(trt.IInt8EntropyCalibrator2):
    """Provides batches for INT8 calibration."""
    def __init__(self, image_dir: Path, batch_size: int, input_shape: Tuple[int,int], cache_file: Path):
        super().__init__()
        self.cache_file = str(cache_file)
        self.batch_size = batch_size
        self.input_shape = input_shape
        # collect all image paths
        self.image_paths = list(image_dir.glob('*'))
        self.current_index = 0
        # allocate device memory for batch
        self.device_input = cuda.mem_alloc(batch_size * 3 * input_shape[0] * input_shape[1] * np.float32().nbytes)

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_batch(self, names: List[str]) -> Optional[List[int]]:
        if self.current_index + self.batch_size > len(self.image_paths):
            return None
        # load batch of images
        batch = self.image_paths[self.current_index:self.current_index + self.batch_size]
        arr = np.zeros((self.batch_size, 3, *self.input_shape), dtype=np.float32)
        for i, img_path in enumerate(batch):
            # basic loading + resize
            import cv2
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.input_shape)
            arr[i] = img.transpose(2,0,1) / 255.0
        # copy to device
        cuda.memcpy_htod(self.device_input, arr.ravel())
        self.current_index += self.batch_size
        return [int(self.device_input)]

    def read_calibration_cache(self) -> Optional[bytes]:
        if Path(self.cache_file).exists():
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        with open(self.cache_file, 'wb') as f:
            f.write(cache)

# --------------------- Model Manager ---------------------
class ModelManager:
    """Manages loading, exporting, and caching of YOLO models and TensorRT engines with enhanced accuracy options."""
    def __init__(
        self,
        device: Optional[str] = None,
        precision: str = "int8",
        model_size: str = "m",       # 'n', 's', 'm', 'l', 'x'
        calibration_dir: Optional[Path] = None,
        calibration_batch: int = 8,
    ):
        self.device = get_optimal_device(device)
        self.precision = precision.lower()
        self.model_size = model_size.lower()
        self.calibration_dir = calibration_dir
        self.calibration_batch = calibration_batch

        self.models: Dict[str, YOLO] = {}
        self.engines: Dict[str, trt.ICudaEngine] = {}

        self.data_dir = Path(__file__).parent.parent / "data"
        self.data_dir.mkdir(exist_ok=True)

        base_names = {
            "player": "football-player-detection",
            "pitch": "football-pitch-detection",
            "ball":   "football-ball-detection",
        }
        self.model_paths = {
            name: self.data_dir / f"{base}.pt"
            for name, base in base_names.items()
        }
        self.onnx_paths = {n: p.with_suffix('.onnx') for n,p in self.model_paths.items()}
        self.trt_paths = {
            name: p.with_suffix(f'.{self.model_size}.{self.precision}.engine')
            for name, p in self.model_paths.items()
        }

        self._ensure_models_exist()

    def _ensure_models_exist(self) -> None:
        missing = [n for n,p in self.model_paths.items() if not p.exists()]
        if missing:
            logger.info(f"Missing models: {', '.join(missing)}. Downloading size={self.model_size}...")
            download_models()

    def load_model(self, name: str) -> YOLO:
        if name in self.models:
            return self.models[name]
        if name not in self.model_paths:
            raise ValueError(f"Unknown model: {name}")
        path = self.model_paths[name]
        logger.info(f"Loading YOLOv8{self.model_size} '{name}' on {self.device}")
        model = YOLO(str(path)).to(self.device)
        self.models[name] = model
        return model

    def export_to_onnx(self, name: str, input_size: Tuple[int,int] = (640,640)) -> Path:
        onnx_path = self.onnx_paths[name]
        if not onnx_path.exists():
            logger.info(f"Exporting '{name}' to ONNX at {onnx_path}")
            self.load_model(name).export(
                format='onnx', imgsz=input_size, opset=14, simplify=True,
                dynamic=False
            )
        return onnx_path

    def build_trt_engine(self, name: str, workspace: int = 4096) -> trt.ICudaEngine:
        engine_path = self.trt_paths[name]
        if engine_path.exists():
            return self._load_engine(engine_path)

        onnx_path = self.export_to_onnx(name)
        logger.info(f"Building TRT engine '{name}' (size={self.model_size}, prec={self.precision})")

        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)

        with open(str(onnx_path), 'rb') as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    logger.error(parser.get_error(i))
                raise RuntimeError("ONNX parse failed")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace * (1 << 20))

        if self.precision == 'fp16' and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        elif self.precision == 'int8' and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            if not self.calibration_dir:
                raise ValueError("INT8 requires calibration_dir")
            cache_file = self.data_dir / f"calib_{name}.cache"
            calibrator = Int8Calibrator(
                image_dir=self.calibration_dir,
                batch_size=self.calibration_batch,
                input_shape=(640, 640),
                cache_file=cache_file
            )
            config.int8_calibrator = calibrator  # Updated API usage for TRT 10.x

        profile = builder.create_optimization_profile()
        profile.set_shape("images", (1,3,640,640), (1,3,640,640), (1,3,640,640))
        config.add_optimization_profile(profile)

        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("Failed to build TensorRT engine")

        with trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(serialized_engine)

        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)

        self.engines[name] = engine
        return engine

    def _load_engine(self, engine_path: Path) -> trt.ICudaEngine:
        logger.info(f"Loading TRT engine from {engine_path}")
        with open(str(engine_path),'rb') as f, trt.Runtime(TRT_LOGGER) as rt:
            engine = rt.deserialize_cuda_engine(f.read())
        key = engine_path.stem.split('.')[0]
        self.engines[key] = engine
        return engine

    def get_trt_engine(self, name: str) -> trt.ICudaEngine:
        if name not in self.engines:
            return self.build_trt_engine(name)
        return self.engines[name]

    def clear_cache(self) -> None:
        self.models.clear()
        self.engines.clear()

# --------------------------- Inference with TTA & NMS ---------------------------
def infer_with_trt(
    engine: trt.ICudaEngine,
    image: np.ndarray,
    scales: List[float] = [1.0],
    flip: bool = False,
    conf_th: float = 0.3,
    iou_th: float = 0.45
) -> np.ndarray:
    """Inference with optional multi-scale & horizontal flip TTA, plus NMS."""
    all_preds = []
    h0, w0 = image.shape[:2]
    for scale in scales:
        size = int(640 * scale)
        import cv2
        img_resized = cv2.resize(image, (size,size))
        for do_flip in ([False, True] if flip else [False]):
            img_t = img_resized[:, ::-1] if do_flip else img_resized
            preds = _run_trt(engine, img_t)
            if do_flip:
                preds[:,[0,2]] = size - preds[:,[2,0]]
            # rescale boxes back
            preds[:,:4] *= (w0/size, h0/size, w0/size, h0/size)
            all_preds.append(preds)
    if not all_preds:
        return np.empty((0,6))
    raw = np.vstack(all_preds)
    # NMS
    mask = raw[:,4]>=conf_th
    raw = raw[mask]
    # sort
    idxs = raw[:,4].argsort()[::-1]
    keep = []
    for i in idxs:
        if all(_compute_iou(raw[i,:4], raw[j,:4]) < iou_th for j in keep):
            keep.append(i)
    return raw[keep]

# Usage
if __name__ == "__main__":
    calib_dir = Path('calibration_images')  # populate with representative images
    mgr = ModelManager(device='cuda:0', precision='int8', model_size='m', calibration_dir=calib_dir)
    engine = mgr.get_trt_engine('player')
    import cv2
    img = cv2.imread('frame.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dets = infer_with_trt(engine, img, scales=[1.0,0.75], flip=True)
    print("Detections:", dets)
