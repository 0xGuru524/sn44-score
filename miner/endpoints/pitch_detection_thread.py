import os
import json
import time
from typing import Dict, Any, List
import supervision as sv
import numpy as np
from loguru import logger
import cv2
from fiber.logging_utils import get_logger
from miner.utils.model_manager import ModelManager
from miner.utils.video_processor_cuda import VideoProcessor

logger = get_logger(__name__)

def process_pitch_detection(
    index: int,
    batch_frames:List[Any],
    model_manager: ModelManager
) -> Dict[str, Any]:
    """Process a soccer video and return tracking data."""
    pitch_model  = model_manager.load_model("pitch")
    
    try:
        tracking_data = {"frames": []}
        start_time = time.time()
        if len(batch_frames) == 0:
            return tracking_data
        
        for frame_number, frame in enumerate(batch_frames):
            if frame is None:
                return tracking_data
            pitch_result = pitch_model.predict(source=frame, verbose=False, device=0, conf=0.25)[0]                
            keypoints = sv.KeyPoints.from_ultralytics(pitch_result)     
            
            # Convert numpy arrays to Python native types
            frame_data = {
                "frame_number": int(frame_number),  # Convert to native int
                "keypoints": keypoints.xy[0].tolist() if keypoints and keypoints.xy is not None else []
            }
            tracking_data["frames"].append(frame_data)
        
        processing_time = time.time() - start_time
        tracking_data["processing_time"] = processing_time
        
        total_frames = len(tracking_data["frames"])
        fps = total_frames / processing_time if processing_time > 0 else 0
        logger.info(
            f"Completed processing {total_frames} frames in {processing_time:.1f}s "
            f"({fps:.2f} fps) on {model_manager.device} device"
        )
        
        return tracking_data
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        
    finally:
        model_manager.clear_cache()
