import time
from typing import Any, Dict, List
import supervision as sv
from loguru import logger
from miner.utils.model_manager import ModelManager

def process_player_detection(
    index: int,
    batch_frames: List[Any],
    model_manager: ModelManager
) -> Dict[str, Any]:
    """Process a soccer video and return tracking data."""
    # Load a fresh player model instance per thread to avoid shared state issues
    player_model = model_manager.load_model("player")
    # Pre-fuse the model once to merge conv+bn layers and avoid runtime fuse calls
    try:
        # Fuse model layers if available
        if hasattr(player_model, 'model') and hasattr(player_model.model, 'fuse'):
            player_model.model.fuse()
    except Exception:
        logger.warning(f"[batch {index}] Model fusion failed or already applied, skipping")
    # Disable further fuse attempts by overriding the fuse method
    if hasattr(player_model, 'model') and hasattr(player_model.model, 'fuse'):
        player_model.model.fuse = lambda *args, **kwargs: player_model.model

    tracker = sv.ByteTrack()

    try:
        tracking_data: Dict[str, Any] = {"frames": []}
        start_time = time.time()

        if not batch_frames:
            return tracking_data

        for frame_number, frame in enumerate(batch_frames):
            if frame is None:
                logger.warning(f"[batch {index}, frame {frame_number}] None frame, skipping")
                continue

            # If frame is on GPU, download to CPU
            if hasattr(frame, "download"):
                frame = frame.download()

            # Run inference; fusion is already applied
            result = player_model.predict(
                source=frame,
                imgsz=1280,
                verbose=False,
                device=model_manager.device,
                conf=0.25
            )[0]

            detections = sv.Detections.from_ultralytics(result)
            detections = tracker.update_with_detections(detections)

            # Convert to native Python types
            objects = []
            if detections and detections.tracker_id is not None:
                for tid, bbox, cls in zip(
                    detections.tracker_id,
                    detections.xyxy,
                    detections.class_id
                ):
                    objects.append({
                        "id": int(tid),
                        "bbox": [float(x) for x in bbox],
                        "class_id": int(cls)
                    })

            tracking_data["frames"].append({
                "frame_number": frame_number,
                "objects": objects
            })

        proc_time = time.time() - start_time
        tracking_data["processing_time"] = proc_time
        total = len(tracking_data["frames"])
        fps = total / proc_time if proc_time > 0 else 0.0
        logger.info(
            f"[batch {index}] Processed {total} frames in {proc_time:.2f}s ({fps:.2f} fps)"
        )

        return tracking_data

    except Exception:
        logger.exception(f"[batch {index}] Error in process_player_detection")
        raise

