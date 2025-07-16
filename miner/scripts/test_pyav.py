import av
import subprocess
import cv2
import os

video_url = "https://scoredata.me/2025_06_26/2025_06_26_fe9b2d06/2025_06_26_fe9b2d06_ae7f7046fb4845388408d98c8afe4e_29a8fa51.mp4"

ffmpeg_cmd = [
    'ffmpeg',
     "-hwaccel", "cuda",
    '-loglevel', 'error',
    '-i', video_url,
    '-c:v', 'libx264',
    '-preset', 'ultrafast',
    '-tune', 'zerolatency',
    '-f', 'mpegts',
    '-'
]


ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

container = av.open(ffmpeg_process.stdout, format='mpegts')

output_dir = "frames"
os.makedirs(output_dir, exist_ok=True)

try:
    for idx, frame in enumerate(container.decode(video=0)):
        img = frame.to_ndarray(format='bgr24')

        # Save each frame (or do your processing here)
        cv2.imwrite(f"{output_dir}/frame_{idx:06d}.jpg", img)

finally:
    ffmpeg_process.stdout.close()
    ffmpeg_process.stderr.close()
    ffmpeg_process.wait()