from pathlib import Path
from typing import Dict, Optional
from ultralytics import YOLO
from loguru import logger
from torch.serialization import safe_globals, add_safe_globals
from ultralytics.nn.tasks import DetectionModel
from torch.nn.modules.container import Sequential
from ultralytics.nn.modules import Conv
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.batchnorm     import BatchNorm2d
from torch.nn.modules.activation     import SiLU

from miner.utils.device import get_optimal_device
from scripts.download_models import download_models

add_safe_globals([DetectionModel, Sequential, Conv, Conv2d, BatchNorm2d, SiLU])

class ModelManager:
    """Manages the loading and caching of YOLO models."""
    
    def __init__(self, device: Optional[str] = None):
        self.device = get_optimal_device(device)
        self.models: Dict[str, YOLO] = {}
        self.data_dir = Path(__file__).parent.parent / "data"
        self.data_dir.mkdir(exist_ok=True)
        
        # Define model paths
        self.model_paths = {
            "player": self.data_dir / "football-player-detection.pt",
            "pitch": self.data_dir / "football-pitch-detection.pt",
        }
        self.engine_paths = {
            "player": self.data_dir / "football-player-detection.engine",
            "pitch": self.data_dir / "football-pitch-detection.engine",
        }
        # Check if models exist, download if missing
        self._ensure_models_exist()
    
    def _ensure_models_exist(self) -> None:
        """Check if required models exist, download if missing."""
        missing_models = [
            name for name, path in self.model_paths.items() 
            if not path.exists()
        ]
        
        if missing_models:
            logger.info(f"Missing models: {', '.join(missing_models)}")
            logger.info("Downloading required models...")
            download_models()
    
    def load_model(self, model_name: str) -> YOLO:
        """
        Load a model by name, using cache if available.
        
        Args:
            model_name: Name of the model to load ('player', 'pitch', or 'ball')
            
        Returns:
            YOLO: The loaded model
        """
        model = None
        if model_name in self.models:
            return self.models[model_name]
        
        if model_name not in self.model_paths:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_path = self.model_paths[model_name]
        engine_path = self.engine_paths[model_name]
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}. "
                "Please ensure all required models are downloaded."
            )
        
        logger.info(f"Loading {model_name} model from {model_path} to {self.device}")
        task = "detect" if model_name == 'player' else "pose"
        model = YOLO(str(engine_path), task=task)
        self.models[model_name] = model
            
        return model
    
    def exportTensorRT(self,  model_name:str, dynamic:bool=True,  batch_size:int=-1, device=0) -> bool:
        """Load all models into cache."""
        model_path = self.model_paths[model_name]
        task = "detect" if model_name == 'player' else "pose"
        model = YOLO(str(model_path), task=task).to(device)
        model.export(
            format='engine',
            half=True,
            batch=batch_size,
            device=device,
            imgsz=(736, 1280),
            nms=True
        )
    
    def load_all_models(self) -> None:
        """Load all models into cache."""
        for model_name in self.model_paths.keys():
            self.load_model(model_name)
    
    def get_model(self, model_name: str) -> YOLO:
        """
        Get a model by name, loading it if necessary.
        
        Args:
            model_name: Name of the model to get ('player', 'pitch', or 'ball')
            
        Returns:
            YOLO: The requested model
        """
        return self.load_model(model_name)
    
    def clear_cache(self) -> None:
        """Clear the model cache."""
        self.models.clear() 