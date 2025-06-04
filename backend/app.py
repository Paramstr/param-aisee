from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import asyncio
import logging
import uvicorn
from contextlib import asynccontextmanager

from .config import settings
from .events import event_bus, Event, EventType
from .core.shared import container
from .core.conversation import conversation_storage

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
        """Send message to all connected clients"""
        if not self.active_connections:
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


if __name__ == "__main__":
    uvicorn.run(
        "backend.app:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level="info"
    )
