#!/usr/bin/env python3
import asyncio
import json
import os
import sys
from pathlib import Path
import time
from loguru import logger
from typing import List, Dict, Union

miner_dir = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, miner_dir)

from utils.video_downloader_ffmpeg import download_video


TEST_VIDEO_URL = "https://pub-a55bd0dbae3c4afd86bd066961ab7d1e.r2.dev/test_10secs.mov"


async def main():
    try:
        logger.info("Starting video processing test")
        start_time = time.time()
        
        logger.info(f"Downloading test video from {TEST_VIDEO_URL}")
        t0 = time.time()
        video_path = await download_video(TEST_VIDEO_URL)
        logger.info(video_path)
        
        t1 = time.time()
        logger.info(f"Video downloaded to {video_path} in {t1 - t0}")
            
    finally:
        try:
            # video_path.unlink()
            logger.info("Cleaned up temporary video file")
        except Exception as e:
            logger.error(f"Error cleaning up video file: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 
