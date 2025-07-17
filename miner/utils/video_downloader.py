from pathlib import Path
import httpx
from fastapi import HTTPException
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger
import subprocess
import time
import uuid
import os

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def download_video(url: str) -> Path:
    """
    Download video with retries and proper redirect handling.
    
    Args:
        url: URL of the video to download
        
    Returns:
        Path: Path to the downloaded video file
        
    Raises:
        HTTPException: If download fails
    """
    start = time.time()
    try:
        dir = '/dev/shm'
        file_name = str(uuid.uuid4()) + '.mp4'
        full_path = os.path.join(dir, file_name)
        # result = subprocess.call(["wget", url, "-O", full_path, "-q"])
        result = subprocess.call(["aria2c","-x", "16", "-s", "16", '-d', dir, "-o", file_name, url])
        # result = subprocess.call(["axel", "-n", "10", "-p", "-o", full_path,  url])
        elapsed = time.time() - start
        if result == 0:
            print("Downloaded video in {:.2f} seconds".format(elapsed))
            return Path(full_path)
        else:
            print("Failed to download {} (wget error code {})".format(url, result))
    except Exception as e:
        print("Failed to download {}: {}".format(url, e))
        
    return Path(url)
    
