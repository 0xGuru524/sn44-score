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


async def main():
    width, height, _, codec = await probe_video('input.mp4')
    print(f"width: {width}")
    print(f"height: {height}")
    print(f"codec: {codec}")
    
if __name__ == "__main__":
    asyncio.run(main()) 