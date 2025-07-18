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

from utils.model_manager import ModelManager
from utils.device import get_optimal_device

async def main():
    try:
        start_time = time.time()
        model_manager = None
        device = get_optimal_device('cuda')
        model_manager = ModelManager(device=device)
        logger.info(f"Using device: {device}")
        logger.info("Starting model exporting...")
        model_manager.exportTensorRT(model_name="pitch", dynamic=False, batch_size=32, device=device)
        model_manager.exportTensorRT(model_name="player", dynamic=False, batch_size=32, device=device)
        logger.info("Models exported successfully")
        total_time = time.time() - start_time
        logger.info("Processing completed successfully!")
        logger.info(f"Total time (including download): {total_time:.2f} seconds")
        
        model_manager.clear_cache()
            
    finally:
        logger.info("Cleaned up temporary file")

if __name__ == "__main__":
    asyncio.run(main()) 