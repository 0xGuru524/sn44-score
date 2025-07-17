import asyncio
import time
from typing import AsyncGenerator, Optional, Tuple
import cv2
import numpy as np
import supervision as sv
from loguru import logger
import math

class VideoProcessor:
    """Handles video processing with frame streaming and timeout management."""
    
    def __init__(
        self,
        device: str = "cuda",
        cuda_timeout: float = 900.0,  # 15 minutes for CUDA
        mps_timeout: float = 1800.0,  # 30 minutes for MPS
        cpu_timeout: float = 10800.0,  # 3 hours for CPU
    ):
        self.device = device
        # Set timeout based on device
        if device == "cuda":
            self.processing_timeout = cuda_timeout
        elif device == "mps":
            self.processing_timeout = mps_timeout
        else:  # cpu or any other device
            self.processing_timeout = cpu_timeout
            
        logger.info(f"Video processor initialized with {device} device, timeout: {self.processing_timeout:.1f}s")
        
    def gpu_batches_opencv(path, duration_sec=3.0):
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
            
            batch.append(frame)  # frame is a NumPy array on the CPU
            if len(batch) >= per_bs:
                yield batch
                batch = []
        cap.release()
    
    async def stream_frames(
        self,
        video_path: str,
        start_sec: float, 
        duration_sec: float
    ) -> AsyncGenerator[Tuple[int, np.ndarray], None]:
        """
        Stream video frames asynchronously with timeout protection.
        Process ALL frames regardless of compute device.
        
        Args:
            video_path: Path to the video file
            
        Yields:
            Tuple[int, np.ndarray]: Frame number and frame data
        """
        start_time = time.time()
        reader = cv2.cudacodec.createVideoReader(video_path)
        if reader is None:
            raise IOError(f"Cannot create CUDA VideoReader for: {video_path}")
        
        fps = reader.get(cv2.CAP_PROP_FPS)
        total_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
        
        start_frame = int(start_sec * fps)
        frames_to_read = int(duration_sec * fps)
        # 4. Clamp and handle out-of-range
        if start_frame >= total_frames:
            reader.release()
        
        end_frame = min(total_frames, start_frame + frames_to_read)
        # 5. Seek to the desired start frame (approximate: keyframe‐aligned)
        reader.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        try:
            frame_count = 0
            while frame_count < frames_to_read:
                elapsed_time = time.time() - start_time
                if elapsed_time > self.processing_timeout:
                    logger.warning(
                        f"Video processing timeout reached after {elapsed_time:.1f}s "
                        f"on {self.device} device ({frame_count} frames processed)"
                    )
                    break
                
                # Use run_in_executor to prevent blocking the event loop
                ret, frame = await asyncio.get_event_loop().run_in_executor(
                    None, reader.nextFrame
                )
                
                if not ret:
                    logger.info(f"Completed processing {frame_count} frames in {elapsed_time:.1f}s on {self.device} device")
                    break
                
                if frame is None:
                    logger.info(f"Completed processing {frame_count} frames in {elapsed_time:.1f}s on {self.device} device")
                    break
                
                # gpu = cv2.cuda_GpuMat()
                # gpu.upload(frame)
                
                # print("GPU empty?", gpu.empty())
                # # Multiple operations
                gpu_resized = cv2.cuda.resize(gpu, (640, 640))
                gpu_rgb = cv2.cuda.cvtColor(gpu_resized, cv2.COLOR_BGR2RGB)
                # gpu_rgb = gpu_rgb.convertTo(cv2.CV_32F, alpha=1.0 / 255.0, beta=0, stream=None)
                
                # result = gpu_rgb.download()
                # img = np.transpose(result, (2, 0, 1))
                # img = np.expand_dims(img, axis=0)
                
                frame = gpu_rgb.download()
                
                yield frame_count, frame
                frame_count += 1
                
                # Small delay to prevent CPU hogging while still processing all frames
                await asyncio.sleep(0)
        
        finally:
            reader.release()
    
    @staticmethod
    def get_video_info(video_path: str) -> sv.VideoInfo:
        """Get video information using supervision."""
        return sv.VideoInfo.from_video_path(video_path)
    
    @staticmethod
    async def ensure_video_readable(video_path: str, timeout: float = 5.0) -> bool:
        """
        Check if video is readable within timeout period.
        
        Args:
            video_path: Path to video file
            timeout: Maximum time to wait for video check
            
        Returns:
            bool: True if video is readable
        """
        try:
            async def _check_video():
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    return False
                ret, _ = cap.read()
                cap.release()
                return ret
            
            return await asyncio.wait_for(_check_video(), timeout)
        
        except asyncio.TimeoutError:
            logger.error(f"Timeout while checking video readability: {video_path}")
            return False
        except Exception as e:
            logger.error(f"Error checking video readability: {str(e)}")
            return False 