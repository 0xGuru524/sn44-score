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
import cv2
import math
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

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

logger = get_logger(__name__)

CONFIG = SoccerPitchConfiguration()

N_CHUNKS = 10           # ← change to 10 pieces
WORKERS  = 10   # e.g. cap at 4 ffmpeg jobs at once (tweak to your CPU/RAM)

# Global model manager instance
model_manager = None

def get_model_manager(config: Config = Depends(get_config)) -> ModelManager:
    global model_manager
    if model_manager is None:
        model_manager = ModelManager(device=config.device)
        model_manager.load_all_models()
    return model_manager

model_manager = get_model_manager(config = get_config())

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

async def process_soccer_video(
    video_path: str,
    model_manager: ModelManager,
) -> Dict[str, Any]:
    """
    Process a soccer video via CUDA-accelerated decode and TRT inference.
    """
    start_time = time.time()
    
    total = get_duration(video_path)
    seg_len = total / N_CHUNKS
    print(f"Total duration: {total:.2f}s → {N_CHUNKS} slices of {seg_len:.2f}s each")
    
    tasks_pitch = []
    tasks_player = []
    for i in range(N_CHUNKS):
        start = i * seg_len
        tasks_player.append(
            asyncio.create_task(process_player_detection(video_path, i, start, seg_len, model_manager))
        )
        tasks_pitch.append(
            asyncio.create_task(process_pitch_detection(video_path, i, start, seg_len, model_manager))
        )

    pitch_results = await asyncio.gather(*tasks_pitch)   # runs them concurrently
    player_results = await asyncio.gather(*tasks_player)   # runs them concurrently
    
    processing_time = time.time() - start_time
    tracking_data = {"frames": []}
    keypoints = {}
    objects = {}
    
    for index, r in pitch_results:
        print("✔ done:", "")
        
    for r in player_results:
        print("✔ done:", "")          
        
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
                    video_path,
                    model_manager
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