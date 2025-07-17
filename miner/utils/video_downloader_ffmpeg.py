import argparse
import subprocess
import sys
import os
from pathlib import Path

def run_ffmpeg_segment(url, segment_time, output_pattern):
    cmd = [
        "ffmpeg", "-y",
        '-hwaccel', 'cuda',
        '-loglevel', 'debug',
        "-i", url,
        "-c", "copy",
        "-map", "0",
        "-f", "segment",
        "-segment_time", str(segment_time),
        "-reset_timestamps", "1",
        '-segment_format', 'mp4',
        output_pattern
    ]

    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)
    
async def download_video(url: str):
    dir = '/dev/shm'
    file_name = "out_%03d.mp4"
    full_path = os.path.join(dir, file_name)
    try:
        run_ffmpeg_segment(
            url=url,
            segment_time=1,
            output_pattern=full_path,
        )
        return Path(dir)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg failed with exit code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)
    return url