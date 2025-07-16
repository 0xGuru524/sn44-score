import asyncio
import subprocess
import time
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
from fastapi import APIRouter, Depends, HTTPException, Request
import supervision as sv
from loguru import logger
import av

# Local imports
from fiber.logging_utils import get_logger
from miner.core.models.config import Config
from miner.core.configuration import factory_config
from miner.dependencies import get_config, verify_request, blacklist_low_stake
from sports.configs.soccer import SoccerPitchConfiguration
from miner.utils.device import get_optimal_device
from miner.utils.model_manager import ModelManager
from miner.utils.shared import miner_lock

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



# --- 1. Health Management ---
async def health_check():
    """Endpoint for availability and resource monitoring"""
    gpu = torch.cuda.is_available()
    cpu_load = __import__('os').getloadavg()[0]
    return {"status": "ok", "gpu_available": gpu, "cpu_load": cpu_load}

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
    video_url: str,
    model_manager: ModelManager,
) -> Dict[str, Any]:
    """
    Stream-process a soccer video using CUDA decode, batching, pinned-memory,
    overlapped I/O/compute, and ball detection, with a single robust fallback.
    """
    import tempfile, requests, os

    # Determine input source: try streaming; on failure, download locally
    input_source = video_url
    tmp_path: Optional[str] = None

    try:
        # Get metadata
        width, height, _, codec = await probe_video(input_source)
        frame_size = width*height*3
        max_batch = model_manager.max_batch_size
        device = torch.device(model_manager.device)
        stream = torch.cuda.Stream(device=device) if device.type!="cpu" else None

        # Pre-allocate GPU tensor
        gpu_batch = None
        if device.type!="cpu":
            gpu_batch = torch.empty((max_batch,3,height,width),dtype=torch.float16,device=device)

        queue: asyncio.Queue = asyncio.Queue(maxsize=4)

        async def producer():
            ffmpeg_cmd = [
                'ffmpeg', "-hide_banner",
                "-hwaccel", "cuda",
                '-loglevel', 'error',
                '-fflags', 'nobuffer',
                '-flags', 'low_delay',
                '-i', video_url,
                '-c:v', 'h264_cuvid',
                '-preset', 'ultrafast',
                '-tune', 'zerolatency',
                '-pix_fmt', 'bgr24',
                '-f', 'mpegts',
                '-an',
                '-'
            ]
            p = await asyncio.create_subprocess_exec(
                ffmpeg_cmd,
                 stdout=subprocess.PIPE, 
                 stderr=subprocess.PIPE
            )
            count=0; batch=[]; idx=[]
            try:
                container = av.open(ffmpeg_process.stdout, format='mpegts')
                for frame in enumerate(container.decode(video=0)):
                    img = frame.to_ndarray(format='bgr24')
                    count+=1; 
                    batch.append(img); 
                    idx.append(count)
                    if len(batch)==max_batch:
                        await queue.put((batch.copy(),idx.copy())); 
                        batch.clear(); 
                        idx.clear()
                
            except asyncio.IncompleteReadError:
                if batch: await queue.put((batch,idx))
            finally:
                await p.wait(); await queue.put(None)

        async def consumer():
            data={"frames":[]}; 
            track=sv.ByteTrack()
            m_p=model_manager; 
            pm=m_p.get_model("pitch"); 
            plm=m_p.get_model("player"); 
            while True:
                item=await queue.get()
                if item is None: break
                frames, ids = item; bsz=len(frames)
                if gpu_batch is not None:
                    tns=[torch.from_numpy(f).permute(2,0,1).float().pin_memory() for f in frames]
                    for i,ht in enumerate(tns): gpu_batch[i].copy_(ht.to(device,non_blocking=True).half())
                    inp=gpu_batch[:bsz]
                else: inp=frames
                if stream:
                    with torch.cuda.stream(stream): res_p=pm.model(inp); res_pl=plm.model(inp,imgsz=1280)
                    torch.cuda.current_stream(device).synchronize()
                else:
                    res_p=pm.model(inp); res_pl=plm.model(inp,imgsz=1280)
                for rp,rpl,rb,i in zip(res_p,res_pl,res_b,ids):
                    kp=sv.KeyPoints.from_ultralytics(rp); det=sv.Detections.from_ultralytics(rpl)
                    det=track.update_with_detections(det); bd=sv.Detections.from_ultralytics(rb)
                    objs=[{"id":int(t),"bbox":[float(x) for x in bb],"class_id":int(c)} for t,bb,c in zip(det.tracker_id,det.xyxy,det.class_id)] if det.tracker_id is not None else []
                    data["frames"].append({"frame_number":i,"keypoints":(kp.xy[0].tolist() if kp.xy is not None else []),"objects":objs})
            return data

        # run
        start_time = time.time()
        prod=asyncio.create_task(producer()); 
        result=await consumer(); 
        await prod
        tot=len(result["frames"]); 
        dt=time.time()-start_time; 
        result["processing_time"]=dt
        logger.info(f"Processed {tot} frames in {dt:.1f}s ({tot/dt:.2f}fps)")
        return result
    finally:
        if tmp_path: os.remove(tmp_path)
# --- 3. Process Challenge Endpoint ---
async def process_challenge(
    request: Request,
    config: Config = Depends(get_config),
    model_manager: ModelManager = Depends(get_model_manager),
):
    data = await request.json()
    challenge_id = data.get("challenge_id")
    video_url = data.get("video_url")
    if not video_url:
        raise HTTPException(400, "No video URL provided")

    logger.info(f"Processing challenge {challenge_id} -> {video_url}")
    stats = await process_soccer_video(video_url, model_manager)
    return {
        "challenge_id": challenge_id,
        "frames": stats["frames"],
        "processing_time": stats["processing_time"]
    }
            
# Create router with dependencies
router = APIRouter()
router.add_api_route(
    "/challenge",
    process_challenge,
    tags=["soccer"],
    dependencies=[Depends(blacklist_low_stake), Depends(verify_request)],
    methods=["POST"],
)