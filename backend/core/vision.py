import asyncio
import logging
import cv2
import threading
import time
import base64
from typing import Optional
import numpy as np

from ..config import settings
from ..events import Event, EventType, event_bus

logger = logging.getLogger(__name__)


class VisionProcessor:
    def __init__(self):
        self.camera: Optional[cv2.VideoCapture] = None
        self.is_capturing = False
        self.capture_thread: Optional[threading.Thread] = None
        self.should_stop = False
        
        # Latest frame storage
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_jpeg: Optional[bytes] = None
        self._frame_lock = threading.Lock()
        
        # Frame capture settings
        self.jpeg_quality = 95
        
    async def start_capture(self):
        """Start camera capture"""
        if self.is_capturing:
            return
        
        if not self._init_camera():
            logger.error("Failed to initialize camera")
            await event_bus.publish(Event(
                type=EventType.ERROR,
                action="camera_init_failed",
                data={"error": "Camera initialization failed"}
            ))
            return
        
        self.is_capturing = True
        self.should_stop = False
        
        # Start capture thread
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        logger.info("Vision processor started")
        await event_bus.publish(Event(
            type=EventType.SYSTEM_STATUS,
            action="camera_active",
            data={"message": "Camera capture started"}
        ))
    
    async def stop_capture(self):
        """Stop camera capture"""
        self.should_stop = True
        self.is_capturing = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        if self.camera:
            self.camera.release()
            self.camera = None
        
        logger.info("Vision processor stopped")
    
    def _init_camera(self) -> bool:
        """Initialize camera"""
        try:
            self.camera = cv2.VideoCapture(settings.camera_index)
            
            if not self.camera.isOpened():
                logger.error(f"Cannot open camera at index {settings.camera_index}")
                return False
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, settings.camera_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.camera_height)
            self.camera.set(cv2.CAP_PROP_FPS, settings.camera_fps)
            
            # Test frame capture
            ret, frame = self.camera.read()
            if not ret:
                logger.error("Failed to capture test frame")
                return False
            
            logger.info(f"Camera initialized: {frame.shape[1]}x{frame.shape[0]} @ {settings.camera_fps}fps")
            return True
        except Exception as e:
            logger.error(f"Camera initialization error: {e}")
            return False
    
    def _capture_loop(self):
        """Main camera capture loop"""
        fps_target = settings.camera_fps
        frame_time = 1.0 / fps_target
        
        while not self.should_stop and self.camera:
            start_time = time.time()
            
            try:
                ret, frame = self.camera.read()
                if not ret:
                    logger.warning("Failed to capture frame")
                    continue
                
                # Process frame (flip once here)
                self._process_frame(frame)
                
                # Maintain target FPS
                elapsed = time.time() - start_time
                sleep_time = max(0, frame_time - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                logger.error(f"Frame capture error: {e}")
    
    def _process_frame(self, frame: np.ndarray):
        """Process captured frame - flip once and store"""
        try:
            # Flip frame horizontally (across y-axis) - ONLY ONCE
            flipped_frame = cv2.flip(frame, 1)
            
            # Encode frame as JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            ret, jpeg_buffer = cv2.imencode('.jpg', flipped_frame, encode_param)
            
            if ret:
                jpeg_bytes = jpeg_buffer.tobytes()
                
                # Update latest frame (already flipped)
                with self._frame_lock:
                    self._latest_frame = flipped_frame.copy()
                    self._latest_jpeg = jpeg_bytes
            else:
                logger.warning("Failed to encode frame as JPEG")
                
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the latest captured frame (already flipped)"""
        with self._frame_lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None
    
    def get_latest_jpeg(self) -> Optional[bytes]:
        """Get the latest frame as JPEG bytes"""
        with self._frame_lock:
            return self._latest_jpeg
    
    def get_latest_jpeg_base64(self) -> Optional[str]:
        """Get the latest frame as base64 encoded JPEG"""
        jpeg_bytes = self.get_latest_jpeg()
        if jpeg_bytes:
            return base64.b64encode(jpeg_bytes).decode('utf-8')
        return None
    
    def capture_frame_for_llm(self) -> Optional[str]:
        """Capture current frame and return as base64 for LLM processing"""
        frame = self.get_latest_frame()
        if frame is None:
            return None
        
        try:
            # Frame is already flipped from _process_frame, no need to flip again
            
            # Resize frame if too large (for API efficiency)
            max_dimension = 800
            height, width = frame.shape[:2]
            
            if max(height, width) > max_dimension:
                if width > height:
                    new_width = max_dimension
                    new_height = int(height * max_dimension / width)
                else:
                    new_height = max_dimension
                    new_width = int(width * max_dimension / height)
                
                frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Encode as JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            ret, jpeg_buffer = cv2.imencode('.jpg', frame, encode_param)
            
            if ret:
                return base64.b64encode(jpeg_buffer.tobytes()).decode('utf-8')
            else:
                logger.error("Failed to encode frame for LLM")
                return None
                
        except Exception as e:
            logger.error(f"Error capturing frame for LLM: {e}")
            return None


# Remove global instance - will be handled by dependency injection
