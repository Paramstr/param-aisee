import asyncio
import logging
import cv2
import threading
import time
import base64
import tempfile
import os
from typing import Optional, Dict, Any
from datetime import datetime
import numpy as np

from ..config import settings
from ..events import Event, EventType, event_bus

logger = logging.getLogger(__name__)


def fourcc_to_string(fourcc):
    """Convert fourcc code back to string for logging"""
    try:
        return "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    except:
        return str(fourcc)


class VideoRecorder:
    """Video recording system for tool use"""
    
    def __init__(self, vision_processor):
        self.vision_processor = vision_processor
        
        # Video settings - use H.264 codec for web compatibility
        self.fps = 30  # Target FPS for video recording
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MPEG-4 Part 2 - fallback
        
        # Try H.264 codec for better web browser compatibility
        # Common H.264 fourcc codes: 'H264', 'avc1', 'X264'
        h264_fourccs = ['avc1', 'H264', 'X264']  # Order of preference for web compatibility
        
        # Test which codec works best on this system
        self.best_fourcc = self._find_best_codec(h264_fourccs)
        
        # Create recordings directory if it doesn't exist
        self.recordings_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "recordings")
        os.makedirs(self.recordings_dir, exist_ok=True)
        logger.info(f"📁 Video recordings will be saved to: {self.recordings_dir}")
        logger.info(f"🎬 Using video codec: {self.best_fourcc}")
    
    def _find_best_codec(self, h264_fourccs):
        """Find the best available codec for video recording"""
        # Test with a small dummy video to see which codec works
        test_size = (320, 240)
        test_fps = 10
        
        for fourcc_str in h264_fourccs:
            try:
                fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
                # Try to create a test video writer
                temp_path = "/tmp/test_codec.mp4"
                test_writer = cv2.VideoWriter(temp_path, fourcc, test_fps, test_size)
                
                if test_writer.isOpened():
                    test_writer.release()
                    # Clean up test file
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                    logger.info(f"✅ Found working H.264 codec: {fourcc_str}")
                    return fourcc
                else:
                    test_writer.release()
                    logger.debug(f"❌ Codec {fourcc_str} not available")
                    
            except Exception as e:
                logger.debug(f"❌ Codec {fourcc_str} failed: {e}")
                continue
        
        # Fallback to mp4v if no H.264 codec works
        logger.warning("⚠️ No H.264 codec available, falling back to mp4v (may not play in browsers)")
        return self.fourcc
        
    async def record_video(self, duration: int) -> Dict[str, Any]:
        """Record a video clip of specified duration"""
        if not self.vision_processor.is_capturing:
            return {
                "success": False,
                "error": "Camera not available"
            }
        
        try:
            # Validate duration
            duration = max(1, min(300, duration))  # Clamp to 1-300 seconds (5 minutes)
            
            await event_bus.publish(Event(
                type=EventType.TOOL_EVENT,
                action="recording_start",
                data={
                    "duration": duration,
                    "message": f"Starting {duration}-second video recording"
                }
            ))
            
            # Start recording
            video_result = await self._record_video_async(duration)
            
            if video_result["success"]:
                await event_bus.publish(Event(
                    type=EventType.TOOL_EVENT,
                    action="recording_complete",
                    data={
                        "duration": duration,
                        "file_size": video_result.get("file_size", 0),
                        "file_path": video_result.get("file_path", ""),
                        "message": f"Video recording completed ({duration}s)"
                    }
                ))
            else:
                await event_bus.publish(Event(
                    type=EventType.TOOL_EVENT,
                    action="recording_failed",
                    data={
                        "duration": duration,
                        "error": video_result.get("error"),
                        "message": f"Video recording failed: {video_result.get('error')}"
                    }
                ))
            
            return video_result
            
        except Exception as e:
            logger.error(f"Video recording error: {e}")
            await event_bus.publish(Event(
                type=EventType.TOOL_EVENT,
                action="recording_failed",
                data={
                    "duration": duration,
                    "error": str(e),
                    "message": f"Video recording failed: {e}"
                }
            ))
            
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _record_video_async(self, duration: int) -> Dict[str, Any]:
        """Record video in async context"""
        # Create timestamped filename for permanent storage
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"recording_{timestamp}_{duration}s.mp4"
        video_path = os.path.join(self.recordings_dir, video_filename)
        
        try:
            # Get current frame to determine dimensions
            current_frame = self.vision_processor.get_latest_frame()
            if current_frame is None:
                return {
                    "success": False,
                    "error": "No camera frame available"
                }
            
            height, width = current_frame.shape[:2]
            
            # Initialize video writer with best available codec
            video_writer = cv2.VideoWriter(
                video_path,
                self.best_fourcc,
                self.fps,
                (width, height)
            )
            
            if not video_writer.isOpened():
                # Try fallback to mp4v if preferred codec failed
                logger.warning(f"Primary codec failed, trying mp4v fallback")
                video_writer.release()
                video_writer = cv2.VideoWriter(
                    video_path,
                    self.fourcc,  # mp4v fallback
                    self.fps,
                    (width, height)
                )
                
                if not video_writer.isOpened():
                    return {
                        "success": False,
                        "error": "Failed to initialize video writer with any codec"
                    }
            
            logger.info(f"🎬 Video writer initialized with {fourcc_to_string(self.best_fourcc)}/MP4 format")
            logger.info(f"📁 Saving video to: {video_path}")
            
            # Record video frames - pass dimensions for reference
            result = await self._record_frames(video_writer, duration, width, height)
            video_writer.release()
            
            if not result["success"]:
                self._cleanup_file(video_path)
                return result
            
            # Convert to base64 for LLM processing
            try:
                # Verify video file was created and has content
                if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
                    return {
                        "success": False,
                        "error": "Video file was not created or is empty"
                    }
                
                file_size = os.path.getsize(video_path)
                logger.info(f"🎬 Video file created: {file_size} bytes")
                logger.info(f"💾 Video saved locally: {video_path}")
                
                with open(video_path, 'rb') as f:
                    video_bytes = f.read()
                    
                    if len(video_bytes) == 0:
                        return {
                            "success": False,
                            "error": "Video file is empty"
                        }
                    
                    video_base64 = base64.b64encode(video_bytes).decode('utf-8')
                    logger.info(f"🎬 Video encoded to base64: {len(video_base64)} characters")
                
                return {
                    "success": True,
                    "video_base64": video_base64,
                    "duration": duration,
                    "file_size": file_size,
                    "file_path": video_path,
                    "filename": video_filename,
                    "frames_recorded": result["frames_recorded"]
                }
                
            except Exception as e:
                self._cleanup_file(video_path)
                return {
                    "success": False,
                    "error": f"Failed to encode video: {e}"
                }
            
        except Exception as e:
            self._cleanup_file(video_path)
            return {
                "success": False,
                "error": f"Recording failed: {e}"
            }
    
    async def _record_frames(self, video_writer: cv2.VideoWriter, duration: int, expected_width: int, expected_height: int) -> Dict[str, Any]:
        """Record frames for specified duration"""
        start_time = time.time()
        frame_interval = 1.0 / self.fps  # Time between frames
        frames_recorded = 0
        
        logger.info(f"🎬 Starting frame recording: {duration}s at {self.fps}fps (interval: {frame_interval:.3f}s)")
        
        try:
            while time.time() - start_time < duration:
                frame_start = time.time()
                
                # Get current frame from vision processor (already flipped)
                frame = self.vision_processor.get_latest_frame()
                
                if frame is not None:
                    # Ensure frame is in correct format (BGR, uint8)
                    if frame.dtype != np.uint8:
                        frame = frame.astype(np.uint8)
                    
                    # Ensure frame has correct dimensions if needed
                    height, width = frame.shape[:2]
                    if width != expected_width or height != expected_height:
                        frame = cv2.resize(frame, (expected_width, expected_height))
                    
                    # Write frame to video (write() returns None, not boolean)
                    video_writer.write(frame)
                    frames_recorded += 1
                    
                    if frames_recorded % 30 == 0:  # Log every 30 frames (every second at 30fps)
                        elapsed = time.time() - start_time
                        logger.debug(f"🎬 Recorded {frames_recorded} frames in {elapsed:.1f}s")
                else:
                    logger.warning("No frame available during recording")
                
                # Maintain target FPS
                elapsed = time.time() - frame_start
                sleep_time = max(0, frame_interval - elapsed)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)  # Use synchronous sleep instead of async
            
            actual_duration = time.time() - start_time
            logger.info(f"🎬 Recording completed: {frames_recorded} frames in {actual_duration:.2f}s")
            
            return {
                "success": True,
                "frames_recorded": frames_recorded,
                "actual_duration": actual_duration
            }
            
        except Exception as e:
            logger.error(f"Frame recording error: {e}")
            return {
                "success": False,
                "error": f"Frame recording failed: {e}",
                "frames_recorded": frames_recorded
            }
    
    def _cleanup_file(self, file_path: str):
        """Clean up video file"""
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except Exception as e:
            logger.warning(f"Failed to cleanup file {file_path}: {e}")
    
    def is_available(self) -> bool:
        """Check if video recording is available"""
        return (self.vision_processor is not None and 
                self.vision_processor.is_capturing) 