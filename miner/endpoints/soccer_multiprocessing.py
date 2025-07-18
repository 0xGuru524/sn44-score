import os
import json
import time
from typing import Optional, Dict, Any, List, Tuple
import supervision as sv
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
import asyncio
from pathlib import Path
from loguru import logger
import cv2
import math
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

from fiber.logging_utils import get_logger
from miner.core.models.config import Config
from miner.core.configuration import factory_config
from miner.dependencies import get_config, verify_request, blacklist_low_stake
from sports.configs.soccer import SoccerPitchConfiguration
from miner.utils.model_manager import ModelManager
from miner.utils.shared import miner_lock
from miner.utils.video_downloader import download_video
from miner.endpoints.player_detection_thread import process_player_detection
from miner.endpoints.pitch_detection_thread import process_pitch_detection
from miner.utils.video_processor_cuda import VideoProcessor
from miner.utils.device import get_optimal_device

# Make sure we use 'spawn' so CUDA contexts don't get copied
mp.set_start_method('spawn', force=True)

logger = get_logger(__name__)

CONFIG = SoccerPitchConfiguration()

N_CHUNKS = 3           # ← change to 10 pieces
WORKERS  = 1   # e.g. cap at 4 ffmpeg jobs at once (tweak to your CPU/RAM)

model_managers = []

for i in range(WORKERS):
    model_manager = ModelManager(device=get_optimal_device('gpu'), idx=i)
    model_manager.load_all_models()
    model_managers.append(model_manager)

def get_batches_frames(path, duration_sec=3.0):
    """
    Yields batches of frames (as NumPy arrays) from a video at `path`,
    each batch covering approximately `duration_sec` seconds.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    per_bs = math.ceil(fps * duration_sec)

    batch = []
    while True:
        ret, frame = cap.read()
        if not ret:
            if batch:
                yield batch
            break
        
        # gpu = cv2.cuda_GpuMat()
        # gpu.upload(frame)
        
        # print("GPU empty?", gpu.empty())
        # # Multiple operations
        frame = cv2.resize(frame, (640, 640))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # gpu_rgb = gpu_rgb.convertTo(cv2.CV_32F, alpha=1.0 / 255.0, beta=0, stream=None)
        # result = gpu_rgb.download()
        batch.append(frame)  # frame is a NumPy array on the CPU
        if len(batch) >= per_bs:
            yield batch
            batch = []
            
    cap.release()

def get_duration(path):
    """Return total duration in seconds via ffprobe."""
    cmd = (
        "ffprobe -v error "
        "-show_entries format=duration "
        "-of default=noprint_wrappers=1:nokey=1 "
        f"{shlex.quote(path)}"
    )
    out = subprocess.run(shlex.split(cmd), capture_output=True, text=True, check=True).stdout
    return float(out.strip())

def _worker_player(
    idx: int,
    batch_frames: List[Any],
    model_manger: ModelManager
) -> Tuple[int, Dict[str, Any]]:
    """
    This runs in a separate process:
     - Re-build the ModelManager from config
     - Run both player & pitch detection on the batch
    Returns (batch_index, player_results, pitch_results)
    """    
    # 2) Run player detection
    player_data = process_player_detection(idx, batch_frames, model_manager)

    return idx, player_data

def _worker_pitch(
    idx: int,
    batch_frames: List[Any],
    model_manger: ModelManager
) -> Tuple[int, Dict[str, Any]]:
    """
    This runs in a separate process:
     - Re-build the ModelManager from config
     - Run both player & pitch detection on the batch
    Returns (batch_index, player_results, pitch_results)
    """    

    # 3) Run pitch detection
    pitch_data  = process_pitch_detection(idx, batch_frames, model_manager)

    return idx, pitch_data

async def process_soccer_video(
    video_path: str
) -> Dict[str, Any]:
    """
    Process a soccer video via CUDA-accelerated decode and TRT inference.
    """
    start_time = time.time()
    
    total = get_duration(video_path)
    seg_len = total / WORKERS
    print(f"Total duration: {total:.2f}s → {WORKERS} slices of {seg_len:.2f}s each")
    
    
     # 1) Build the list of batches
    batches = list(enumerate(get_batches_frames(video_path, seg_len)))

    # 2) Pull out only the serializable bits to re-create ModelManager in each child

    player_results = {}
    pitch_results  = {}

    # 3) Fire up the process pool
    with ThreadPoolExecutor(max_workers=WORKERS) as exe:
        futures_player = {
            exe.submit(_worker_player, idx, batch, model_managers[idx]): idx
            for idx, batch in batches
        }
        futures_pitch = {
            exe.submit(_worker_player, idx, batch, model_managers[idx]): idx
            for idx, batch in batches
        }

        # 4) As each process finishes, gather its results
        for fut in as_completed(futures_player):
            idx = futures_player[fut]
            try:
                _, player_data = fut.result()
                player_results[idx] = player_data
            except Exception as e:
                logger.exception(f"Batch {idx} crashed in subprocess")
                # Decide if you want to keep going or abort
                raise
        # 4) As each process finishes, gather its results
        for fut in as_completed(futures_pitch):
            idx = futures_pitch[fut]
            try:
                _, pitch_data = fut.result()
                pitch_results[idx] = pitch_data
            except Exception as e:
                logger.exception(f"Batch {idx} crashed in subprocess")
                # Decide if you want to keep going or abort
                raise
    # 5) Re-order into lists
    ordered_idxs   = sorted(player_results.keys())
    player_output  = [player_results[i] for i in ordered_idxs]
    pitch_output   = [pitch_results[i]  for i in ordered_idxs]     
        
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