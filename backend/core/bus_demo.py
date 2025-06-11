import asyncio
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import cv2
from PIL import Image
import moondream as md

from ..events import event_bus, Event, EventType
from ..config import settings

logger = logging.getLogger(__name__)

class VisionProcessor:
    """Handles both Moondream Cloud and local Moondream Server inference"""
    
    def __init__(self):
        self.cloud_model = None
        self.local_model = None
        self.cloud_available = False
        self.local_available = False
        self.use_cloud = False  # Default to local
        
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
        
    async def _test_cloud_connection(self) -> bool:
        """Test if Moondream Cloud is working"""
        try:
            # Create a small test image
            test_image = Image.new('RGB', (100, 100), color='white')
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
            result = self.local_model.caption(test_image)
            return result is not None
        except Exception as e:
            logger.warning(f"Moondream Server test failed: {e}")
            return False
    
    async def detect_bus_number(self, frame, query: str = "What bus numbers can you see in this image?") -> Dict:
        """Detect bus numbers using either Moondream Cloud or local server"""
        if self.use_cloud and self.cloud_available:
            return await self._detect_cloud(frame, query)
        elif not self.use_cloud and self.local_available:
            return await self._detect_local(frame, query)
        else:
            # Fallback to available option
            if self.cloud_available:
                return await self._detect_cloud(frame, query)
            elif self.local_available:
                return await self._detect_local(frame, query)
            else:
                raise RuntimeError("No inference method available")
    
    async def _detect_local(self, frame, query: str) -> Dict:
        """Local inference using Moondream Server"""
        start_time = time.time()
        
        try:
            # Convert frame to PIL Image
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # Use Moondream Server
            result = self.local_model.query(pil_image, query)
            response = result["answer"]
            
            latency = (time.time() - start_time) * 1000
            bus_numbers = self._parse_bus_numbers(response)
            
            return {
                "response": response,
                "bus_numbers": bus_numbers,
                "latency": latency,
                "inference_type": "local"
            }
            
        except Exception as e:
            logger.error(f"Moondream Server inference failed: {e}")
            return {
                "response": f"Error: {str(e)}",
                "bus_numbers": [],
                "latency": (time.time() - start_time) * 1000,
                "inference_type": "local"
            }
    
    async def _detect_cloud(self, frame, query: str) -> Dict:
        """Cloud inference using Moondream Cloud"""
        start_time = time.time()
        
        try:
            # Convert frame to PIL Image
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # Use Moondream Cloud
            result = self.cloud_model.query(pil_image, query)
            response = result["answer"]
            
            latency = (time.time() - start_time) * 1000
            bus_numbers = self._parse_bus_numbers(response)
            
            return {
                "response": response,
                "bus_numbers": bus_numbers,
                "latency": latency,
                "inference_type": "cloud"
            }
            
        except Exception as e:
            logger.error(f"Moondream Cloud inference failed: {e}")
            return {
                "response": f"Error: {str(e)}",
                "bus_numbers": [],
                "latency": (time.time() - start_time) * 1000,
                "inference_type": "cloud"
            }
    
    def _parse_bus_numbers(self, response: str) -> List[str]:
        """Extract bus numbers from response"""
        import re
        patterns = [r'(?:Route|Bus|Line)\s*(\d+)', r'#(\d+)', r'\b(\d{1,3})\b']
        bus_numbers = []
        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            bus_numbers.extend(matches)
        return sorted(list(set(bus_numbers)))

class BusDemoManager:
    """Simple bus demo manager"""
    
    def __init__(self):
        self.vision_processor = VisionProcessor()
        self.is_running = False
        self.detection_task: Optional[asyncio.Task] = None
        self.videos_dir = Path("backend/bus_videos")
        self.videos_dir.mkdir(exist_ok=True)
        
    async def initialize(self):
        """Initialize vision processor"""
        await self.vision_processor.initialize()
        logger.info("Bus demo manager initialized")
    
    def set_inference_mode(self, use_cloud: bool) -> Dict:
        """Switch between cloud and local inference"""
        if use_cloud and not self.vision_processor.cloud_available:
            return {"error": "Moondream Cloud not available"}
        if not use_cloud and not self.vision_processor.local_available:
            return {"error": "Moondream Server not available"}
        
        self.vision_processor.use_cloud = use_cloud
        mode = "cloud" if use_cloud else "local"
        logger.info(f"Switched to {mode} inference")
        return {"message": f"Switched to {mode} inference", "mode": mode}
    
    def get_videos(self) -> List[Dict]:
        """Get predefined bus videos"""
        videos = []
        # Check for bus_video_1.mp4, bus_video_2.mp4, etc.
        for i in range(1, 6):  # Support up to 5 videos
            video_file = self.videos_dir / f"bus_video_{i}.mp4"
            if video_file.exists():
                try:
                    cap = cv2.VideoCapture(str(video_file))
                    if cap.isOpened():
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        duration = int(frame_count / fps) if fps > 0 else 0
                        cap.release()
                        
                        videos.append({
                            "id": str(i),
                            "name": f"Bus Video {i}",
                            "description": f"Bus detection video - {duration}s duration",
                            "duration": duration,
                            "thumbnail": "🚌"
                        })
                except Exception as e:
                    logger.warning(f"Could not process video {video_file}: {e}")
        return videos
    
    async def start_detection(self, video_id: str) -> Dict:
        """Start processing video frames with Moondream"""
        if self.is_running:
            return {"error": "Detection already running"}
        
        # Look for bus_video_{id}.mp4
        video_path = self.videos_dir / f"bus_video_{video_id}.mp4"
        if not video_path.exists():
            return {"error": f"Video file bus_video_{video_id}.mp4 not found"}
        
        self.is_running = True
        self.detection_task = asyncio.create_task(self._process_video(str(video_path)))
        
        await event_bus.publish(Event(
            type=EventType.BUS_DEMO,
            action="detection_started",
            data={"video_id": video_id, "video_name": f"Bus Video {video_id}"}
        ))
        
        return {"message": f"Started detection on Bus Video {video_id}", "video_id": video_id}
    
    async def stop_detection(self) -> Dict:
        """Stop detection"""
        if not self.is_running:
            return {"error": "No detection running"}
        
        self.is_running = False
        if self.detection_task:
            self.detection_task.cancel()
        
        await event_bus.publish(Event(
            type=EventType.BUS_DEMO,
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
            frame_skip = int(fps) if fps > 0 else 30  # Process 1 frame per second
            frame_index = 0
            
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every nth frame
                if frame_index % frame_skip == 0:
                    # Send frame to vision processor
                    result = await self.vision_processor.detect_bus_number(frame)
                    
                    if result["bus_numbers"]:
                        # Convert frame to base64
                        _, buffer = cv2.imencode('.jpg', frame)
                        frame_b64 = base64.b64encode(buffer).decode('utf-8')
                        
                        # Stream result to frontend
                        await event_bus.publish(Event(
                            type=EventType.BUS_DEMO,
                            action="detection_result",
                            data={
                                "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
                                "latency": result["latency"],
                                "busNumber": ", ".join(result["bus_numbers"]),
                                "frameUrl": f"data:image/jpeg;base64,{frame_b64}",
                                "frameIndex": frame_index // frame_skip
                            }
                        ))
                    
                    # Small delay between processing
                    await asyncio.sleep(0.1)
                
                frame_index += 1
            
            cap.release()
            self.is_running = False
            
            await event_bus.publish(Event(
                type=EventType.BUS_DEMO,
                action="detection_completed",
                data={}
            ))
            
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            await event_bus.publish(Event(
                type=EventType.BUS_DEMO,
                action="detection_error",
                data={"error": str(e)}
            ))
        finally:
            if 'cap' in locals():
                cap.release()
            self.is_running = False
    
    def get_status(self) -> Dict:
        """Get demo status"""
        return {
            "is_running": self.is_running,
            "cloud_available": self.vision_processor.cloud_available,
            "local_available": self.vision_processor.local_available,
            "current_mode": "cloud" if self.vision_processor.use_cloud else "local",
            "videos_available": len(list(self.videos_dir.glob("*.mp4")))
        }

# Global bus demo manager instance
bus_demo_manager = BusDemoManager() 