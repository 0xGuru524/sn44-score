import os
import json
import time
from typing import Optional, Dict, Any, Tuple
import numpy as np
import cv2
from fastapi import APIRouter, Depends, HTTPException, Request
import asyncio
import supervision as sv
from loguru import logger
from pathlib import Path
import subprocess

from fiber.logging_utils import get_logger
from miner.core.models.config import Config
from miner.core.configuration import factory_config
from miner.dependencies import get_config, verify_request, blacklist_low_stake
from sports.configs.soccer import SoccerPitchConfiguration
from miner.utils.device import get_optimal_device
from miner.utils.model_manager import ModelManager
from miner.utils.video_downloader import download_video
from miner.utils.video_processor import VideoProcessor
from miner.utils.shared import miner_lock

logger = get_logger(__name__)
# CONFIG = factory_config(SoccerPitchConfiguration)

# Global model manager instance
model_manager: Optional[ModelManager] = None

def get_model_manager(config: Config = Depends(get_config)) -> ModelManager:
    global model_manager
    if model_manager is None:
        # Use small fp16 TRT engines for balanced speed/accuracy
        model_manager = ModelManager(
            device=config.device,
        )
        # Eagerly build engines
        for name in ("player", "pitch", "ball"):
            model_manager.get_trt_engine(name)
    return model_manager


# --- 2. Video Processor (Stream-optimized) ---
async def probe_video(url: str) -> Tuple[int, int, float, str]:
    """Probe resolution, fps, and codec via ffprobe JSON parsing with fallback download."""
    import tempfile, requests, os, json

    async def _run_ffprobe_json(path: str) -> Tuple[int, int, float, str]:
        # Use JSON output to reliably parse streams
        cmd = [
            "ffprobe", "-v", "error",
            "-print_format", "json",
            "-show_streams",
            path
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        out, err = await proc.communicate()
        if proc.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"ffprobe JSON error for {path}: {err.decode().strip()}"
            )
        info = json.loads(out)
        streams = info.get("streams", [])
        for s in streams:
            if s.get("codec_type") == "video":
                w = s.get("width")
                h = s.get("height")
                fps_str = s.get("r_frame_rate", "0/1")
                codec = s.get("codec_name", "")
                if w is None or h is None:
                    continue
                # parse fps
                if "/" in fps_str:
                    num_str, den_str = fps_str.split("/")
                    try:
                        num, den = int(num_str), int(den_str)
                    except ValueError:
                        num, den = 0, 1
                else:
                    try:
                        num, den = int(fps_str), 1
                    except ValueError:
                        num, den = 0, 1
                fps = num / den if den else float(num)
                return w, h, fps, codec
        raise HTTPException(
            status_code=400,
            detail="No video stream found in ffprobe output"
        )

    # Try probing URL directly
    try:
        return await _run_ffprobe_json(url)
    except HTTPException as e:
        if "Protocol not found" in getattr(e, 'detail', ''):
            tmpf = tempfile.NamedTemporaryFile(suffix=os.path.splitext(url)[-1], delete=False)
            tmpf.close()
            try:
                resp = requests.get(url, stream=True, timeout=10)
                resp.raise_for_status()
                with open(tmpf.name, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=1<<20):
                        if chunk:
                            f.write(chunk)
                return await _run_ffprobe_json(tmpf.name)
            finally:
                try: os.remove(tmpf.name)
                except: pass
        raise



async def process_soccer_video(
    video_path: str,
    model_manager: ModelManager,
) -> Dict[str, Any]:
    """
    Process a soccer video via CUDA-accelerated decode and TRT inference.
    """
    start = time.time()
    width, height, _, codec = await probe_video(video_path)
    frame_size = width*height*3
    
    # Prepare TRT engines
    engine_player = model_manager.load_model("player")
    engine_pitch  = model_manager.load_model("pitch")
    engine_ball   = model_manager.load_model("ball")
    tracker = sv.ByteTrack()
    
    tracking_data = {"frames": []}
        
    ffmpeg_cmd = [
        'ffmpeg', "-hide_banner",
        "-hwaccel", "cuda",
        '-loglevel', 'error',
        '-fflags', 'nobuffer',
        '-flags', 'low_delay',
        '-i', video_path,
        '-c:v', 'h264_cuvid',
        '-preset', 'ultrafast',
        '-tune', 'zerolatency',
        '-pix_fmt', 'bgr24',
        '-f', 'mpegts',
        '-an',
        '-'
    ]
    pipe = await asyncio.create_subprocess_exec(
        *ffmpeg_cmd,
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE
    )
    try:
        frame_number = 0
        while True:
            # Read raw bytes for a single frame
            print("Reading", frame_size, "bytes...")
            raw_frame = await pipe.stdout.read(frame_size)
            print("Got", len(raw_frame), "bytes.")
            if raw_frame == b'':
                # EOF — only break if the process has exited
                if pipe.returncode is not None:
                    print("FFmpeg exited early.")
                    break
                else:
                    await asyncio.sleep(0.1)
                    continue

            if len(raw_frame) < frame_size:
                print(f"Incomplete frame: got {len(raw_frame)}")
                continue

            # Convert to NumPy array and reshape
            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3))
            frame_number = frame_number + 1
           # Convert BGR->RGB

            # Pitch keypoints
            pitch_preds = engine_pitch(rgb)
            keypoints = sv.KeyPoints(xy=pitch_preds[:, :2] if pitch_preds.size else np.empty((0,2)))

            # Player detections + tracking
            player_preds = engine_player(rgb)
            detections = sv.Detections(
                xyxy=player_preds[:, :4],
                confidence=player_preds[:, 4],
                class_id=player_preds[:, 5].astype(int)
            )
            detections = tracker.update_with_detections(det)

            # players
            frame_data = {
                "frame_number": int(frame_number),  # Convert to native int
                "keypoints": keypoints.xy[0].tolist() if keypoints and keypoints.xy is not None else [],
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
            tracking_data["frames"].append(frame_data)

            # Log progress
            if frame_number % 100 == 0:
                elapsed = time.time() - start
                fps = frame_number / elapsed if elapsed>0 else 0
                logger.info(f"Processed {frame_number} frames in {elapsed:.1f}s ({fps:.2f} fps)")

        processing_time = time.time() - start
        tracking_data["processing_time"] = processing_time
        total = len(tracking_data["frames"])
        logger.info(f"Completed {total} frames in {processing_time:.1f}s ({total/processing_time:.2f} fps)")
        return tracking_data
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Video processing error: {str(e)}")

    finally:
        pipe.terminate()  # Gracefully request termination
        await pipe.wait()  # Wait for it to exit
async def process_challenge(
    request: Request,
    config: Config = Depends(get_config),
    model_manager: ModelManager = Depends(get_model_manager),
):
    async with miner_lock:
        data = await request.json()
        cid = data.get("challenge_id")
        url = data.get("video_url")
        if not url:
            raise HTTPException(400, "No video URL provided")
        logger.info(f"Challenge {cid}: downloading {url}")
        try:
            stats = await process_soccer_video(video_path, model_manager)
            return {"challenge_id": cid, **stats}
        finally:
            try: os.unlink(video_path)
            except: pass

router = APIRouter()
router.add_api_route(
    "/challenge",
    process_challenge,
    tags=["soccer"],
    dependencies=[Depends(blacklist_low_stake), Depends(verify_request)],
    methods=["POST"],
)
