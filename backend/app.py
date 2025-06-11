from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import asyncio
import logging
import uvicorn
from contextlib import asynccontextmanager
import sounddevice as sd
import cv2
from pathlib import Path

from .config import settings
from .events import event_bus, Event, EventType
from .core.shared import container
from .core.conversation import conversation_storage
from .core.bus_demo import bus_demo_manager
from .core.service_registry import service_registry, ServiceMode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager"""
    logger.info("🚀 Starting Osmo Assistant...")
    
    try:
        # Initialize dependency container
        await container.initialize()
        
        # Initialize bus demo manager
        try:
            await bus_demo_manager.initialize()
            logger.info("✅ Bus demo manager initialized")
        except Exception as e:
            logger.warning(f"⚠️ Bus demo manager failed to initialize: {e}")
        
        # Start all background tasks
        await container.task_manager.start_all_tasks()
        logger.info("✅ Osmo Assistant started successfully")
        yield
    except Exception as e:
        logger.error(f"❌ Failed to start Osmo Assistant: {e}")
        raise
    finally:
        logger.info("🛑 Stopping Osmo Assistant...")
        await container.cleanup()
        logger.info("✅ Osmo Assistant stopped")


# Create FastAPI app
app = FastAPI(
    title="Osmo Assistant",
    description="Always-listening, always-watching AI assistant",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    status = await container.task_manager.get_system_status()
    return {
        "status": "healthy" if status["is_running"] else "unhealthy",
        "system": status,
        "message": "Osmo Assistant is running" if status["is_running"] else "System not running"
    }


@app.get("/frame")
async def get_current_frame():
    """Get the current camera frame as JPEG"""
    jpeg_bytes = container.vision_processor.get_latest_jpeg()
    
    if jpeg_bytes is None:
        raise HTTPException(status_code=404, detail="No frame available")
    
    return Response(content=jpeg_bytes, media_type="image/jpeg")


@app.get("/status")
async def get_status():
    """Get detailed system status"""
    return await container.task_manager.get_system_status()


@app.get("/conversation/stats")
async def get_conversation_stats():
    """Get conversation statistics"""
    return conversation_storage.get_stats()


@app.post("/conversation/clear")
async def clear_conversation():
    """Clear conversation history"""
    conversation_storage.clear_conversation()
    return {"message": "Conversation history cleared"}


class ManualMessageRequest(BaseModel):
    message: str


class VoiceDictationToggleRequest(BaseModel):
    enabled: bool


class CameraCaptureToggleRequest(BaseModel):
    enabled: bool


class ServiceModeRequest(BaseModel):
    mode: str


@app.post("/service/mode")
async def set_service_mode(request: ServiceModeRequest):
    """Switch between service modes (osmo/bus_demo)"""
    try:
        mode = ServiceMode(request.mode)
        await container.task_manager.switch_mode(mode)
        return {
            "message": f"Switched to {mode.value} mode",
            "mode": mode.value,
            "status": service_registry.get_status()
        }
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid mode: {request.mode}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to switch mode: {str(e)}")


@app.get("/service/status")
async def get_service_status():
    """Get current service registry status"""
    return service_registry.get_status()


@app.post("/voice/toggle")
async def toggle_voice_dictation(request: VoiceDictationToggleRequest):
    """Toggle voice dictation on/off"""
    await container.audio_processor.set_voice_dictation_enabled(request.enabled)
    return {
        "message": f"Voice dictation {'enabled' if request.enabled else 'disabled'}",
        "enabled": request.enabled
    }


@app.post("/camera/toggle")
async def toggle_camera_capture(request: CameraCaptureToggleRequest):
    """Toggle camera capture on/off"""
    await container.vision_processor.set_camera_capture_enabled(request.enabled)
    return {
        "message": f"Camera capture {'enabled' if request.enabled else 'disabled'}",
        "enabled": request.enabled
    }


@app.post("/conversation/send")
async def send_manual_message(request: ManualMessageRequest):
    """Send a manual message to the AI for processing"""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    # Trigger context ready event with the manual message
    await event_bus.publish(Event(
        type=EventType.AUDIO_EVENT,
        action="context_ready",
        data={"transcript": request.message.strip()}
    ))
    
    return {"message": "Message sent for processing", "content": request.message.strip()}


class ConnectionManager:
    """Manage WebSocket connections"""
    
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_to_all(self, message: dict):
        """Send message to all connected clients with service filtering"""
        if not self.active_connections:
            return
        
        # Filter events based on active service mode
        if message.get("type") == "event":
            event_data = message.get("event", {})
            event_type = event_data.get("type", "")
            if not service_registry.should_process_event(event_type):
                logger.debug(f"WebSocket event {event_type} filtered for current mode")
                return
        
        dead_connections = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send to WebSocket: {e}")
                dead_connections.append(connection)
        
        # Remove dead connections
        for dead_conn in dead_connections:
            self.disconnect(dead_conn)


# Global connection manager
manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time events"""
    await manager.connect(websocket)
    
    # Subscribe to events
    event_queue = await event_bus.subscribe()
    
    try:
        # Send initial status
        status = await container.task_manager.get_system_status()
        await websocket.send_json({
            "type": "status",
            "data": status
        })
        
        # Event forwarding loop
        while True:
            try:
                # Get event from queue (with timeout to check WebSocket)
                event = await asyncio.wait_for(event_queue.get(), timeout=1.0)
                
                # Send event to client
                await websocket.send_json({
                    "type": "event",
                    "event": event.to_dict()
                })
                
            except asyncio.TimeoutError:
                # Ping the client to check if connection is alive
                try:
                    await websocket.send_json({"type": "ping"})
                except:
                    break
                continue
            except Exception as e:
                logger.error(f"Error in WebSocket event loop: {e}")
                break
                
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected normally")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        manager.disconnect(websocket)


# ================== BUS DEMO ENDPOINTS ==================

@app.get("/bus-demo/videos")
async def get_bus_videos():
    """Get available bus videos for detection"""
    try:
        return {"videos": bus_demo_manager.get_videos()}
    except Exception as e:
        logger.error(f"Error getting bus videos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/bus-demo/video/{video_id}")
async def get_video_info(video_id: str):
    """Get detailed video information"""
    try:
        video_path = Path("backend/bus_videos") / f"bus_video_{video_id}.mp4"
        if not video_path.exists():
            raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
        
        return {
            "id": video_id,
            "filename": f"bus_video_{video_id}.mp4",
            "path": str(video_path),
            "url": f"/bus-demo/stream/{video_id}"
        }
    except Exception as e:
        logger.error(f"Error getting video info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Mount static files for video streaming
app.mount("/bus-demo/videos", StaticFiles(directory="backend/bus_videos"), name="bus_videos")


@app.post("/bus-demo/start/{video_id}")
async def start_bus_detection(video_id: str):
    """Start bus number detection for a specific video"""
    try:
        result = await bus_demo_manager.start_detection(video_id)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except Exception as e:
        logger.error(f"Error starting bus detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/bus-demo/stop")
async def stop_bus_detection():
    """Stop bus number detection"""
    try:
        result = await bus_demo_manager.stop_detection()
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except Exception as e:
        logger.error(f"Error stopping bus detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/bus-demo/status")
async def get_bus_demo_status():
    """Get bus demo status"""
    try:
        return bus_demo_manager.get_status()
    except Exception as e:
        logger.error(f"Error getting bus demo status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class InferenceModeRequest(BaseModel):
    use_cloud: bool


@app.post("/bus-demo/inference-mode")
async def set_inference_mode(request: InferenceModeRequest):
    """Switch between cloud and local inference"""
    try:
        result = bus_demo_manager.set_inference_mode(request.use_cloud)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except Exception as e:
        logger.error(f"Error setting inference mode: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class SystemPromptRequest(BaseModel):
    prompt: str


@app.post("/bus-demo/system-prompt")
async def set_system_prompt(request: SystemPromptRequest):
    """Update the system prompt for inference"""
    try:
        result = bus_demo_manager.set_system_prompt(request.prompt)
        return result
    except Exception as e:
        logger.error(f"Error setting system prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/bus-demo/system-prompt")
async def get_system_prompt():
    """Get the current system prompt"""
    try:
        prompt = bus_demo_manager.get_system_prompt()
        return {"prompt": prompt}
    except Exception as e:
        logger.error(f"Error getting system prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/bus-demo/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload a custom MP4 video for detection"""
    try:
        # Validate file type
        if not file.filename or not file.filename.lower().endswith('.mp4'):
            raise HTTPException(status_code=400, detail="Only MP4 files are supported")
        
        # Read file content
        video_content = await file.read()
        
        # Upload via bus demo manager
        result = await bus_demo_manager.upload_video(video_content, file.filename)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading video: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload video: {str(e)}")


# ================== TEST ENDPOINTS ==================

@app.post("/test-wake-word")
async def test_wake_word():
    """Test endpoint to simulate wake word detection"""
    await event_bus.publish(Event(
        type=EventType.AUDIO_EVENT,
        action="wake_word_detected",
        data={"transcript": "test wake word"}
    ))
    return {"message": "Wake word event triggered"}


@app.post("/test-transcript")
async def test_transcript(transcript: str):
    """Test endpoint to simulate transcript"""
    await event_bus.publish(Event(
        type=EventType.AUDIO_EVENT,
        action="context_ready",
        data={"transcript": transcript}
    ))
    return {"message": f"Transcript event triggered: {transcript}"}


@app.post("/test-video-start")
async def test_video_start(duration: int = 3):
    """Test endpoint to simulate video_start event"""
    await event_bus.publish(Event(
        type=EventType.TOOL_EVENT,
        action="video_start",
        data={
            "tool": "get_video",
            "duration": duration
        }
    ))
    return {"message": f"video_start event triggered with duration {duration}s"}


@app.post("/test-video-recording")
async def test_video_recording(duration: int = 3):
    """Test endpoint to simulate video_recording event"""
    await event_bus.publish(Event(
        type=EventType.TOOL_EVENT,
        action="video_recording",
        data={
            "duration": duration,
            "message": f"Recording {duration}s video..."
        }
    ))
    return {"message": f"video_recording event triggered with duration {duration}s"}


@app.post("/test-video-complete")
async def test_video_complete(success: bool = True, duration: int = 1):
    """Test endpoint to simulate video_complete event"""
    await event_bus.publish(Event(
        type=EventType.TOOL_EVENT,
        action="video_complete",
        data={
            "tool": "get_video",
            "duration": duration,
            "success": success,
            "frames_recorded": duration * 30,
            "file_size": 1024
        }
    ))
    return {"message": f"video_complete event triggered (success={success})"}


@app.post("/test-video-sequence")
async def test_video_sequence(duration: int = 3, delay: float = 3.0):
    """Test endpoint to simulate full video recording sequence with delays"""
    
    # Step 1: video_start
    await event_bus.publish(Event(
        type=EventType.TOOL_EVENT,
        action="video_start",
        data={
            "tool": "get_video", 
            "duration": duration
        }
    ))
    
    # Wait a bit
    await asyncio.sleep(delay)
    
    # Step 2: video_recording  
    await event_bus.publish(Event(
        type=EventType.TOOL_EVENT,
        action="video_recording", 
        data={
            "duration": duration,
            "message": f"Recording {duration}s video..."
        }
    ))
    
    # Wait for "recording duration"
    await asyncio.sleep(delay)
    
    # Step 3: video_complete
    await event_bus.publish(Event(
        type=EventType.TOOL_EVENT,
        action="video_complete",
        data={
            "tool": "get_video",
            "duration": duration, 
            "success": True,
            "frames_recorded": duration * 30,
            "file_size": 1024
        }
    ))
    
    return {"message": f"Full video sequence triggered with {delay}s delays"}


class DeviceUpdateRequest(BaseModel):
    device_type: str  # "audio" or "video"
    device_id: int


@app.get("/devices/audio")
async def get_audio_devices():
    """Get list of available audio input devices"""
    try:
        devices = sd.query_devices()
        audio_devices = []
        default_input_device = sd.query_devices(kind='input')['index']
        
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:  # Input devices only
                audio_devices.append({
                    "id": i,
                    "name": device['name'],
                    "channels": device['max_input_channels'],
                    "sample_rate": device['default_samplerate'],
                    "is_default": i == default_input_device
                })
        
        # Use system default if no device is explicitly set
        current_device = settings.audio_device_index if settings.audio_device_index is not None else default_input_device
        
        return {
            "devices": audio_devices,
            "current_device": current_device
        }
    except Exception as e:
        logger.error(f"Error getting audio devices: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/devices/video")
async def get_video_devices():
    """Get list of available video capture devices"""
    try:
        import subprocess
        import re
        import os
        
        # Get actual camera names from system_profiler on macOS
        def get_camera_names():
            try:
                result = subprocess.run(['system_profiler', 'SPCameraDataType'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    # Parse camera names from system_profiler output
                    camera_names = []
                    lines = result.stdout.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line.endswith(':') and 'Camera' in line:
                            # Extract camera name (remove the trailing colon)
                            name = line[:-1].strip()
                            if name and name != 'Camera':
                                camera_names.append(name)
                    return camera_names
            except Exception as e:
                logger.warning(f"Could not get camera names from system_profiler: {e}")
            return []
        
        camera_names = get_camera_names()
        video_devices = []
        available_cameras = []
        
        # Suppress OpenCV warnings temporarily during camera detection
        old_log_level = os.environ.get('OPENCV_LOG_LEVEL', '')
        os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
        
        try:
            # Test up to 10 camera indices to find available cameras
            for i in range(10):
                cap = cv2.VideoCapture(i)
                if cap is not None and cap.isOpened():
                    # Try to read a frame to verify the device works
                    ret, _ = cap.read()
                    if ret:
                        available_cameras.append(i)
                        # Use actual camera name if available, otherwise fallback to generic name
                        if i < len(camera_names):
                            device_name = camera_names[i]
                        else:
                            device_name = f"Camera {i}"
                        
                        video_devices.append({
                            "id": i,
                            "name": device_name,
                            "is_current": i == settings.camera_index
                        })
                    cap.release()
        finally:
            # Restore original log level
            if old_log_level:
                os.environ['OPENCV_LOG_LEVEL'] = old_log_level
            else:
                os.environ.pop('OPENCV_LOG_LEVEL', None)
        
        # Default to the first available camera (typically the built-in camera)
        default_camera = available_cameras[0] if available_cameras else 0
        current_device = settings.camera_index if settings.camera_index in available_cameras else default_camera
        
        return {
            "devices": video_devices,
            "current_device": current_device
        }
    except Exception as e:
        logger.error(f"Error getting video devices: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/devices/update")
async def update_device(request: DeviceUpdateRequest):
    """Update the selected audio or video device"""
    try:
        if request.device_type == "audio":
            # Update audio device
            settings.audio_device_index = request.device_id
            
            # Restart audio processor with new device
            if container.audio_processor.is_listening:
                await container.audio_processor.stop_listening()
                await container.audio_processor.start_listening()
            
            await event_bus.publish(Event(
                type=EventType.SYSTEM_STATUS,
                action="audio_device_changed",
                data={"device_id": request.device_id, "message": f"Audio device changed to device {request.device_id}"}
            ))
            
        elif request.device_type == "video":
            # Update video device
            settings.camera_index = request.device_id
            
            # Restart vision processor with new device
            if container.vision_processor.is_capturing:
                await container.vision_processor.stop_capture()
                await container.vision_processor.start_capture()
            
            await event_bus.publish(Event(
                type=EventType.SYSTEM_STATUS,
                action="video_device_changed",
                data={"device_id": request.device_id, "message": f"Video device changed to device {request.device_id}"}
            ))
            
        else:
            raise HTTPException(status_code=400, detail="Invalid device_type. Must be 'audio' or 'video'")
        
        return {"message": f"{request.device_type.title()} device updated successfully", "device_id": request.device_id}
        
    except Exception as e:
        logger.error(f"Error updating device: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "backend.app:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level="info",
        access_log=False  # Disable access logging to prevent /frame spam
    )
