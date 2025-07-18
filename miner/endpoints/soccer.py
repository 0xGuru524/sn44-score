import os
import json
import time
from typing import Optional, Dict, Any
import supervision as sv
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
import asyncio
from pathlib import Path
from loguru import logger
import torch
import cv2
import subprocess
import torch.nn.functional as F


from fiber.logging_utils import get_logger
from miner.core.models.config import Config
from miner.core.configuration import factory_config
from miner.dependencies import get_config, verify_request, blacklist_low_stake
from sports.configs.soccer import SoccerPitchConfiguration
from miner.utils.device import get_optimal_device
from miner.utils.model_manager import ModelManager
from miner.utils.video_processor import VideoProcessor
from miner.utils.shared import miner_lock
from miner.utils.video_downloader import download_video

logger = get_logger(__name__)

CONFIG = SoccerPitchConfiguration()

# Global model manager instance
model_manager = None

def get_model_manager(config: Config = Depends(get_config)) -> ModelManager:
    global model_manager
    if model_manager is None:
        model_manager = ModelManager(device=config.device)
        model_manager.load_all_models()
    return model_manager


# Load models
model_manager = get_model_manager(get_config())

player_model = model_manager.get_model("player")
pitch_model = model_manager.get_model("pitch")
tracker = sv.ByteTrack()

    
def get_batches_frames(path, batch_size=16):
    """
    Yields batches of frames (as NumPy arrays) from a video at `path`,
    each batch covering approximately `duration_sec` seconds.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")

    batch = []
    while True:
        ret, frame = cap.read()
        if not ret:
            if batch:
                yield batch
            break
        
        gpu = cv2.cuda_GpuMat()
        gpu.upload(frame)
        
        # print("GPU empty?", gpu.empty())
        # # Multiple operations
        frame = cv2.cuda.resize(gpu, (1280, 736))
        frame = cv2.cuda.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # gpu_rgb = gpu_rgb.convertTo(cv2.CV_32F, alpha=1.0 / 255.0, beta=0, stream=None)
        frame = frame.download()
        
        batch.append(frame)  # frame is a NumPy array on the CPU
        if len(batch) >= batch_size:
            yield batch
            batch = []
            
    cap.release()

async def process_soccer_video(
    video_path: str
) -> Dict[str, Any]:
    """Process a soccer video and return tracking data."""
    
    batch_size = 16 
    imgsz = (736, 1280)  # height, width
    device = get_optimal_device('cuda')
    
    try:             
        start_time = time.time()
        tracking_data = {"frames": []}
        pitch_frame_data = []
        player_frame_data = []
        for frame_number, frames in enumerate(get_batches_frames(video_path, batch_size)):
            exceed_size = 0
            t0 = time.time()
            # Create two CUDA streams
            B = len(frames)
            s1 = torch.cuda.Stream()
            s2 = torch.cuda.Stream()
                # Convert to torch.Tensor [B, C, H, W]
            if B < batch_size:
                exceed_size = batch_size - B
                empty_frame = np.zeros((imgsz[0], imgsz[1], 3), dtype=np.uint8)
                frames.extend([empty_frame] * (batch_size - B))
                
            # Launch model1 on stream1
            with torch.cuda.stream(s1):
                pitch_results = pitch_model.predict(source=frames, verbose=False, conf=0.25, iou=0.45, agnostic_nms=False, max_det=2, device=0)    # asynchronous launch into s1
                if exceed_size > 0:
                    pitch_results = pitch_results[:-exceed_size]
                for idx, pitch_result in enumerate(pitch_results):
                    logger.info(f"Processed frames: {pitch_result[0]}")
                    
                    keypoints = []
                    try:
                        keypoints = sv.KeyPoints.from_ultralytics(pitch_result[0])
                    except AttributeError:
                        keypoints = []
                        
                    frame_data = {
                        "frame_number": int(frame_number * batch_size + idx),  # Convert to native int
                        "keypoints": keypoints.xy[0].tolist() if keypoints and keypoints.xy is not None else [],
                    }
                    pitch_frame_data.append(frame_data)
                    
            # Launch model2 on stream2
            with torch.cuda.stream(s2):
                player_results = player_model.predict(source=frames, verbose=False, conf=0.25, iou=0.45, agnostic_nms=False, max_det=50, device=0)    # asynchronous launch into s2
                if exceed_size > 0:
                    player_results = player_results[:-exceed_size]
                for idx, player_result in enumerate(player_results):
                    detections = sv.Detections.from_ultralytics(player_result[0])
                    detections = tracker.update_with_detections(detections)
                    frame_data = {
                        "frame_number": int(frame_number * 16 + idx),  # Convert to native int
                        "objects": [
                            {
                                "id": int(tracker_id),  # Convert numpy.int64 to native int
                                "bbox": [float(x) for x in bbox],  # Convert numpy.float32/64 to native float
                                "class_id": int(class_id)  # Convert numpy.int64 to native int
                            }
                            for tracker_id, bbox, class_id in zip(
                                detections.tracker_id,
                                detections.xyxy,
                                detections.class_id
                            )
                        ] if detections and detections.tracker_id is not None else []
                    }
                    player_frame_data.append(frame_data)
            # Wait for both streams to finish
            torch.cuda.synchronize()
              
            elapsed = time.time() - t0
            fps = frame_number / elapsed if elapsed > 0 else 0
            logger.info(f"Processed {B} frames in {elapsed:.1f}s ({fps:.2f} fps)")
        
        frame_data = pitch_frame_data
        if len(pitch_frame_data) > len(player_frame_data):
            frame_data = player_frame_data
            
        logger.info(
            f"Pitch Data Length:  {len(pitch_frame_data)} vs Player Data Length: {len(player_frame_data)}"
        )
        for idx, frame in enumerate(frame_data):
            tracked = {
                "frame_number": int(idx),
                "keypoints": pitch_frame_data[idx]["keypoints"],
                "objects": player_frame_data[idx]["objects"]
            }
            tracking_data["frames"].append(tracked)
            
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
        raise HTTPException(status_code=500, detail=f"Video processing error: {str(e)}")

async def process_challenge(
    request: Request,
    config: Config = Depends(get_config)
):
    logger.info("Attempting to acquire miner lock...")
    async with miner_lock:
        logger.info("Miner lock acquired, processing challenge...")
        try:
            challenge_data = await request.json()
            challenge_id = challenge_data.get("challenge_id")
            video_url = challenge_data.get("video_url")
            
            logger.info(f"Received challenge data: {json.dumps(challenge_data, indent=2)}")
            
            if not video_url:
                raise HTTPException(status_code=400, detail="No video URL provided")
            
            logger.info(f"Processing challenge {challenge_id} with video {video_url}")
            
            video_path = await download_video(video_url)
            
            try:
                tracking_data = await process_soccer_video(
                    video_path
                )
                
                response = {
                    "challenge_id": challenge_id,
                    "frames": tracking_data["frames"],
                    "processing_time": tracking_data["processing_time"]
                }
                
                logger.info(f"Completed challenge {challenge_id} in {tracking_data['processing_time']:.2f} seconds")
                return response
                
            finally:
                try:
                    os.unlink(video_path)
                except:
                    pass
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing soccer challenge: {str(e)}")
            logger.exception("Full error traceback:")
            raise HTTPException(status_code=500, detail=f"Challenge processing error: {str(e)}")
        finally:
            logger.info("Releasing miner lock...")

# Create router with dependencies
router = APIRouter()
router.add_api_route(
    "/challenge",
    process_challenge,
    tags=["soccer"],
    dependencies=[Depends(blacklist_low_stake), Depends(verify_request)],
    methods=["POST"],
)