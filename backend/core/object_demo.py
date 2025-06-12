import asyncio
import base64
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import cv2
from PIL import Image
import moondream as md
import shutil
import tempfile
import functools
import concurrent.futures

from ..events import event_bus, Event, EventType
from ..config import settings
from .shared import container

logger = logging.getLogger(__name__)

class MoondreamProcessor:
    """Handles both Moondream Cloud and local Moondream Server inference"""
    
    def __init__(self, cpu_pool: Optional[concurrent.futures.ThreadPoolExecutor] = None):
        self.cloud_model = None
        self.local_model = None
        self.cloud_available = False
        self.local_available = False
        self.use_cloud = False  # Default to local
        self.system_prompt = "What objects can you see in this image? Look for any identifiable objects, vehicles, people, or items clearly visible in the scene. Describe what you see in a concise manner. If you cannot see any clear objects respond with null."
        self.cpu_pool = cpu_pool
        
    async def initialize(self):
        """Initialize both Moondream Cloud and local Moondream Server"""
        
        # Initialize Moondream Cloud
        try:
            if hasattr(settings, 'moondream_api_key') and settings.moondream_api_key:
                self.cloud_model = md.vl(api_key=settings.moondream_api_key)
                # Test the connection
                test_result = await self._test_cloud_connection()
                if test_result:
                    self.cloud_available = True
                    logger.info("Moondream Cloud available")
                else:
                    logger.warning("Moondream Cloud API key invalid or quota exceeded")
            else:
                logger.warning("No Moondream API key found - cloud inference disabled")
        except Exception as e:
            logger.warning(f"Failed to initialize Moondream Cloud: {e}")
        
        # Initialize local Moondream Server
        try:
            local_endpoint = getattr(settings, 'moondream_local_endpoint', 'http://localhost:2020/v1')
            self.local_model = md.vl(endpoint=local_endpoint)
            # Test the connection
            test_result = await self._test_local_connection()
            if test_result:
                self.local_available = True
                logger.info(f"Moondream Server available at {local_endpoint}")
            else:
                logger.warning(f"Moondream Server not available at {local_endpoint}")
        except Exception as e:
            logger.warning(f"Failed to initialize Moondream Server: {e}")
        
        if not self.cloud_available and not self.local_available:
            logger.error("Neither Moondream Cloud nor local server available!")
    
    def set_system_prompt(self, prompt: str) -> Dict:
        """Update the system prompt for inference"""
        old_prompt = self.system_prompt
        self.system_prompt = prompt
        logger.info(f"Updated system prompt from '{old_prompt[:50]}...' to '{prompt[:50]}...'")
        return {
            "message": "System prompt updated successfully",
            "old_prompt": old_prompt,
            "new_prompt": prompt
        }
    
    def get_system_prompt(self) -> str:
        """Get the current system prompt"""
        return self.system_prompt

    async def _test_cloud_connection(self) -> bool:
        """Test if Moondream Cloud is working"""
        try:
            # Create a small test image
            test_image = Image.new('RGB', (100, 100), color='white')
            
            # Run in thread pool to avoid blocking
            if self.cpu_pool:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.cpu_pool, 
                    lambda: self.cloud_model.caption(test_image)
                )
            else:
                result = self.cloud_model.caption(test_image)
            
            return result is not None
        except Exception as e:
            logger.warning(f"Moondream Cloud test failed: {e}")
            return False
    
    async def _test_local_connection(self) -> bool:
        """Test if local Moondream Server is working"""
        try:
            # Create a small test image
            test_image = Image.new('RGB', (100, 100), color='white')
            
            # Run in thread pool to avoid blocking
            if self.cpu_pool:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.cpu_pool, 
                    lambda: self.local_model.caption(test_image)
                )
            else:
                result = self.local_model.caption(test_image)
            
            return result is not None
        except Exception as e:
            logger.warning(f"Moondream Server test failed: {e}")
            return False
    
    async def detect_objects(self, frame, query: Optional[str] = None) -> Dict:
        """Detect objects using either Moondream Cloud or local server"""
        # Use provided query or fall back to system prompt
        query_text = query or self.system_prompt
        
        if self.use_cloud and self.cloud_available:
            return await self._detect_cloud(frame, query_text)
        elif not self.use_cloud and self.local_available:
            return await self._detect_local(frame, query_text)
        else:
            # Fallback to available option
            if self.cloud_available:
                return await self._detect_cloud(frame, query_text)
            elif self.local_available:
                return await self._detect_local(frame, query_text)
            else:
                raise RuntimeError("No inference method available")
    
    def _detect_local_sync(self, image_rgb, query: str) -> Dict:
        """Synchronous local inference for thread pool execution"""
        start_time = time.time()
        
        try:
            pil_image = Image.fromarray(image_rgb)
            
            # Use Moondream Server
            result = self.local_model.query(pil_image, query)
            response = result["answer"]
            
            latency = (time.time() - start_time) * 1000
            
            return {
                "response": response,
                "latency": latency,
                "inference_type": "local"
            }
            
        except Exception as e:
            logger.error(f"Moondream Server inference failed: {e}")
            return {
                "response": f"Error: {str(e)}",
                "latency": (time.time() - start_time) * 1000,
                "inference_type": "local"
            }
    
    def _detect_cloud_sync(self, image_rgb, query: str) -> Dict:
        """Synchronous cloud inference for thread pool execution"""
        start_time = time.time()
        
        try:
            pil_image = Image.fromarray(image_rgb)
            
            # Use Moondream Cloud
            result = self.cloud_model.query(pil_image, query)
            response = result["answer"]
            
            latency = (time.time() - start_time) * 1000
            
            return {
                "response": response,
                "latency": latency,
                "inference_type": "cloud"
            }
            
        except Exception as e:
            logger.error(f"Moondream Cloud inference failed: {e}")
            return {
                "response": f"Error: {str(e)}",
                "latency": (time.time() - start_time) * 1000,
                "inference_type": "cloud"
            }
    
    async def _detect_local(self, frame, query: str) -> Dict:
        """Local inference using Moondream Server - now async with thread pool"""
        try:
            # Convert frame to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run inference in thread pool to avoid blocking the event loop
            if self.cpu_pool:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.cpu_pool,
                    self._detect_local_sync,
                    image_rgb,
                    query
                )
            else:
                # Fallback to direct execution if no thread pool
                result = self._detect_local_sync(image_rgb, query)
            
            return result
            
        except Exception as e:
            logger.error(f"Moondream Server inference failed: {e}")
            return {
                "response": f"Error: {str(e)}",
                "latency": 0,
                "inference_type": "local"
            }
    
    async def _detect_cloud(self, frame, query: str) -> Dict:
        """Cloud inference using Moondream Cloud - now async with thread pool"""
        try:
            # Convert frame to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run inference in thread pool to avoid blocking the event loop
            if self.cpu_pool:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.cpu_pool,
                    self._detect_cloud_sync,
                    image_rgb,
                    query
                )
            else:
                # Fallback to direct execution if no thread pool
                result = self._detect_cloud_sync(image_rgb, query)
            
            return result
            
        except Exception as e:
            logger.error(f"Moondream Cloud inference failed: {e}")
            return {
                "response": f"Error: {str(e)}",
                "latency": 0,
                "inference_type": "cloud"
            }


class ObjectDetectionManager:
    """Enhanced object detection manager with upload capability"""
    
    def __init__(self):
        self.moondream_processor = None  # Will be initialized with CPU pool
        self.is_running = False
        self.detection_task: Optional[asyncio.Task] = None
        self.videos_dir = Path("backend/sample_videos")
        self.videos_dir.mkdir(exist_ok=True)
        self.current_video_id = None  # Track uploaded video
        
    async def initialize(self):
        """Initialize Moondream processor with CPU thread pool"""
        # Get CPU pool from container
        cpu_pool = container.shared.cpu_pool if container.shared else None
        self.moondream_processor = MoondreamProcessor(cpu_pool=cpu_pool)
        await self.moondream_processor.initialize()
        logger.info("Object detection manager initialized with CPU thread pool")
    
    def set_inference_mode(self, use_cloud: bool) -> Dict:
        """Switch between cloud and local inference"""
        if use_cloud and not self.moondream_processor.cloud_available:
            return {"error": "Moondream Cloud not available"}
        if not use_cloud and not self.moondream_processor.local_available:
            return {"error": "Moondream Server not available"}
        
        self.moondream_processor.use_cloud = use_cloud
        mode = "cloud" if use_cloud else "local"
        logger.info(f"Switched to {mode} inference")
        return {"message": f"Switched to {mode} inference", "mode": mode}
    
    def set_system_prompt(self, prompt: str) -> Dict:
        """Update the system prompt for inference"""
        return self.moondream_processor.set_system_prompt(prompt)
    
    def get_system_prompt(self) -> str:
        """Get the current system prompt"""
        return self.moondream_processor.get_system_prompt()
    
    async def upload_video(self, video_content: bytes, filename: str) -> Dict:
        """Upload a custom video file"""
        try:
            # Validate file extension
            if not filename.lower().endswith(('.mp4', '.mov')):
                return {"error": "Only MP4 and MOV files are supported"}
            
            # Use the original filename (without extension) as the video ID for simplicity
            video_id = Path(filename).stem
            file_extension = Path(filename).suffix.lower()
            video_path = self.videos_dir / f"{video_id}{file_extension}"
            
            # Save the uploaded video
            with open(video_path, 'wb') as f:
                f.write(video_content)
            
            # Validate the video can be opened
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                video_path.unlink()  # Delete invalid file
                return {"error": "Invalid video file - cannot be processed"}
            
            # Get video metadata
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = int(frame_count / fps) if fps > 0 else 0
            cap.release()
            
            self.current_video_id = video_id
            
            logger.info(f"Uploaded custom video: {filename} -> {video_id}{file_extension} ({duration}s)")
            
            return {
                "message": "Video uploaded successfully",
                "video_id": video_id,
                "filename": filename,
                "duration": duration,
                "path": str(video_path)
            }
            
        except Exception as e:
            logger.error(f"Error uploading video: {e}")
            return {"error": f"Failed to upload video: {str(e)}"}
    
    def get_videos(self) -> List[Dict]:
        """Get all available sample videos"""
        videos = []
        
        # Check for all .mp4 and .mov files in the sample_videos directory
        video_extensions = ["*.mp4", "*.mov"]
        for pattern in video_extensions:
            for video_file in self.videos_dir.glob(pattern):
                try:
                    cap = cv2.VideoCapture(str(video_file))
                    if cap.isOpened():
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        duration = int(frame_count / fps) if fps > 0 else 0
                        cap.release()
                        
                        # Extract video name without extension
                        video_name = video_file.stem
                        video_id = video_name
                        
                        # Use the filename as display name, cleaning it up
                        display_name = video_name.replace("_", " ").replace("-", " ").title()
                        thumbnail = "🎥"
                        video_type = "sample"
                        
                        videos.append({
                            "id": video_id,
                            "name": display_name,
                            "description": f"Object detection sample - {duration}s duration",
                            "duration": duration,
                            "thumbnail": thumbnail,
                            "type": video_type,
                            "filename": video_file.name
                        })
                except Exception as e:
                    logger.warning(f"Could not process video {video_file}: {e}")
        
        # Sort videos by name for consistent ordering
        videos.sort(key=lambda x: x["name"])
        return videos

    async def start_video_detection(self, video_id: str) -> Dict:
        """Start processing video frames with Moondream"""
        if self.is_running:
            return {"error": "Detection already running"}
        
        # Look for video file with the given ID as filename (without extension)
        video_path = None
        for ext in ['.mp4', '.mov']:
            potential_path = self.videos_dir / f"{video_id}{ext}"
            if potential_path.exists():
                video_path = potential_path
                break
        
        if video_path is None:
            return {"error": f"Video file {video_id} not found (checked .mp4 and .mov)"}
        
        self.is_running = True
        self.detection_task = asyncio.create_task(self._process_video(str(video_path)))
        
        # Use the video_id (filename without extension) as the display name
        video_name = video_id.replace("_", " ").replace("-", " ").title()
        
        await event_bus.publish(Event(
            type=EventType.OBJECT_DEMO,
            action="detection_started",
            data={"video_id": video_id, "video_name": video_name}
        ))
        
        return {"message": f"Started detection on {video_name}", "video_id": video_id}
    
    async def start_realtime_detection(self) -> Dict:
        """Start real-time object detection using live camera feed"""
        if self.is_running:
            return {"error": "Detection already running"}
        
        self.is_running = True
        self.detection_task = asyncio.create_task(self._process_realtime())
        
        await event_bus.publish(Event(
            type=EventType.OBJECT_DEMO,
            action="detection_started",
            data={"video_id": "realtime", "video_name": "Live Camera Feed"}
        ))
        
        return {"message": "Started real-time detection", "video_id": "realtime"}
    
    async def stop_detection(self) -> Dict:
        """Stop detection"""
        if not self.is_running:
            return {"error": "No detection running"}
        
        self.is_running = False
        if self.detection_task:
            self.detection_task.cancel()
        
        await event_bus.publish(Event(
            type=EventType.OBJECT_DEMO,
            action="detection_stopped",
            data={}
        ))
        
        return {"message": "Detection stopped"}
    
    async def _process_video(self, video_path: str):
        """Process video frame by frame with Moondream"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_skip = int(fps/2) if fps > 0 else 15  # Process 2 frames per second
            frame_index = 0
            
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every nth frame
                if frame_index % frame_skip == 0:
                    # Calculate video timestamp
                    video_timestamp_sec = frame_index / fps if fps > 0 else frame_index / 30
                    video_timestamp_formatted = f"{int(video_timestamp_sec // 60):02d}:{int(video_timestamp_sec % 60):02d}.{int((video_timestamp_sec % 1) * 1000):03d}"
                    
                    # Send frame to Moondream processor (now runs in CPU thread pool)
                    result = await self.moondream_processor.detect_objects(frame)
                    
                    if result["response"].strip().lower() != "null":
                        # Convert frame to base64
                        _, buffer = cv2.imencode('.jpg', frame)
                        frame_b64 = base64.b64encode(buffer).decode('utf-8')
                        
                        # Stream result to frontend
                        await event_bus.publish(Event(
                            type=EventType.OBJECT_DEMO,
                            action="detection_result",
                            data={
                                "timestamp": video_timestamp_formatted,
                                "videoTimestamp": video_timestamp_sec,
                                "latency": result["latency"],
                                "detectedObjects": result["response"],
                                "frameUrl": f"data:image/jpeg;base64,{frame_b64}",
                                "frameIndex": frame_index // frame_skip
                            }
                        ))
                    
                    # Yield control after processing to keep event loop responsive
                    await asyncio.sleep(0.01)
                else:
                    # Yield control more frequently for better responsiveness
                    if frame_index % 10 == 0:
                        await asyncio.sleep(0.001)
                
                frame_index += 1
            
            cap.release()
            self.is_running = False
            
            await event_bus.publish(Event(
                type=EventType.OBJECT_DEMO,
                action="detection_completed",
                data={}
            ))
            
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            await event_bus.publish(Event(
                type=EventType.OBJECT_DEMO,
                action="detection_error",
                data={"error": str(e)}
            ))
        finally:
            if 'cap' in locals():
                cap.release()
            self.is_running = False
    
    async def _process_realtime(self):
        """Process live camera frames with Moondream"""
        try:
            frame_counter = 0
            
            while self.is_running:
                # Get latest frame from the main vision processor
                frame = container.vision_processor.get_latest_frame()
                
                if frame is not None:
                    # Process every 15 frames (roughly 2 frames per second at 30fps)
                    if frame_counter % 15 == 0:
                        # Send frame to Moondream processor (now runs in CPU thread pool)
                        result = await self.moondream_processor.detect_objects(frame)
                        
                        if result["response"].strip().lower() != "null":
                            # Convert frame to base64
                            _, buffer = cv2.imencode('.jpg', frame)
                            frame_b64 = base64.b64encode(buffer).decode('utf-8')
                            
                            # Stream result to frontend
                            await event_bus.publish(Event(
                                type=EventType.OBJECT_DEMO,
                                action="detection_result",
                                data={
                                    "timestamp": "LIVE",
                                    "videoTimestamp": time.time(),
                                    "latency": result["latency"],
                                    "detectedObjects": result["response"],
                                    "frameUrl": f"data:image/jpeg;base64,{frame_b64}",
                                    "frameIndex": frame_counter // 15
                                }
                            ))
                        
                        # Yield control after processing to keep event loop responsive
                        await asyncio.sleep(0.01)
                    else:
                        # Yield control for unprocessed frames to keep event loop responsive
                        await asyncio.sleep(0.001)
                    
                    frame_counter += 1
                else:
                    # No frame available, wait a bit longer
                    await asyncio.sleep(0.1)
            
            self.is_running = False
            
            await event_bus.publish(Event(
                type=EventType.OBJECT_DEMO,
                action="detection_completed",
                data={}
            ))
            
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Real-time processing failed: {e}")
            await event_bus.publish(Event(
                type=EventType.OBJECT_DEMO,
                action="detection_error",
                data={"error": str(e)}
            ))
        finally:
            self.is_running = False
    
    def get_status(self) -> Dict:
        """Get demo status"""
        video_count = len(list(self.videos_dir.glob("*.mp4"))) + len(list(self.videos_dir.glob("*.mov")))
        return {
            "is_running": self.is_running,
            "cloud_available": self.moondream_processor.cloud_available,
            "local_available": self.moondream_processor.local_available,
            "current_mode": "cloud" if self.moondream_processor.use_cloud else "local",
            "videos_available": video_count,
            "system_prompt": self.moondream_processor.get_system_prompt()
        }

# Global object detection manager instance
object_detection_manager = ObjectDetectionManager() 