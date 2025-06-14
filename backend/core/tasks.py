import asyncio
import logging
from typing import Optional, Dict, Callable, Any

from ..events import Event, EventType, event_bus
from .service_registry import service_registry, ServiceMode

logger = logging.getLogger(__name__)


class TaskManager:
    def __init__(self, audio_processor, vision_processor, llm_processor, tool_registry=None, object_detection_manager=None):
        # Dependency injection - no more global imports
        self.audio_processor = audio_processor
        self.vision_processor = vision_processor
        self.llm_processor = llm_processor
        self.tool_registry = tool_registry
        self.object_detection_manager = object_detection_manager
        
        self.is_running = False
        self.event_handler_task: Optional[asyncio.Task] = None
        
        # Lookup table for event handlers - much cleaner than if/elif chain
        self.event_handlers: Dict[EventType, Dict[str, Callable]] = {
            EventType.SYSTEM_STATUS: {
                "listening": self._handle_system_listening,
                "camera_active": self._handle_system_camera_active,
                "camera_disabled": self._handle_system_camera_disabled,
                "whisper_loading": self._handle_system_whisper_loading,
                "whisper_ready": self._handle_system_whisper_ready,
                "system_ready": self._handle_system_ready,
            },
            EventType.AUDIO_EVENT: {
                "transcription_start": self._handle_audio_transcription_start,
                "transcription_end": self._handle_audio_transcription_end,
                "raw_transcript": self._handle_audio_raw_transcript,
                "wake_word_detected": self._handle_audio_wake_word,
                "context_ready": self._handle_audio_context_ready,
            },
            EventType.LLM_EVENT: {
                "response_start": self._handle_llm_start,
                "response_chunk": self._handle_llm_chunk,
                "response_end": self._handle_llm_end,
            },
            EventType.TTS_EVENT: {
                "start": self._handle_tts_start,
                "end": self._handle_tts_end,
            },
            EventType.TOOL_EVENT: {
                "tool_start": self._handle_tool_start,
                "tool_complete": self._handle_tool_complete,
                "tool_error": self._handle_tool_error,
                "photo_start": self._handle_photo_start,
                "photo_complete": self._handle_photo_complete,
                "video_start": self._handle_video_start,
                "video_complete": self._handle_video_complete,
                "recording_start": self._handle_recording_start,
                "recording_complete": self._handle_recording_complete,
            },
            EventType.VOICE_CONTROL: {
                "dictation_toggled": self._handle_voice_dictation_toggled,
            },
            EventType.CAMERA_CONTROL: {
                "capture_toggled": self._handle_camera_capture_toggled,
            },
            EventType.TTS_CONTROL: {
                "tts_toggled": self._handle_tts_toggled,
            },
            EventType.OBJECT_DEMO: {
                            "detection_started": self._handle_object_detection_started,
            "detection_result": self._handle_object_detection_result,
            "detection_completed": self._handle_object_detection_completed,
            "detection_stopped": self._handle_object_detection_stopped,
            "detection_error": self._handle_object_detection_error,
            },
            EventType.ERROR: {
                "*": self._handle_error,  # Handle all error actions
            }
        }
        
    async def start_all_tasks(self):
        """Start all background tasks"""
        if self.is_running:
            logger.warning("Tasks already running")
            return
        
        self.is_running = True
        
        try:
            # Start event handler first
            self.event_handler_task = asyncio.create_task(self._event_handler())
            
            # Default to osmo mode on startup
            await service_registry.set_mode(ServiceMode.OSMO)
            await self._start_active_services()
            
            logger.info("All tasks started successfully")
            await event_bus.publish(Event(
                type=EventType.SYSTEM_STATUS,
                action="system_ready",
                data={"message": "Osmo Assistant is ready"}
            ))
            
        except Exception as e:
            logger.error(f"Failed to start tasks: {e}")
            await self.stop_all_tasks()
            raise
    
    async def stop_all_tasks(self):
        """Stop all background tasks"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        try:
            # Cancel event handler
            if self.event_handler_task:
                self.event_handler_task.cancel()
                try:
                    await self.event_handler_task
                except asyncio.CancelledError:
                    pass
                self.event_handler_task = None
            
            # Stop all processors
            await self.audio_processor.stop_listening()
            await self.vision_processor.stop_capture()
            await self.llm_processor.stop()
            
            logger.info("All tasks stopped")
            
        except Exception as e:
            logger.error(f"Error stopping tasks: {e}")
    
    async def switch_mode(self, mode: ServiceMode):
        """Switch to a different service mode"""
        if not self.is_running:
            await service_registry.set_mode(mode)
            return
        
        # Stop current services
        await self._stop_active_services()
        
        # Switch mode
        await service_registry.set_mode(mode)
        
        # Start new services
        await self._start_active_services()
        
        logger.info(f"Switched to {mode.value} mode")
    
    async def _start_active_services(self):
        """Start services based on current active mode"""
        logger.info(f"Starting active services: {service_registry.active_services}")
        
        if service_registry.is_service_active("llm_processor"):
            logger.info("🤖 Starting LLM processor")
            await self.llm_processor.start()
        
        if service_registry.is_service_active("vision_processor"):
            logger.info("📹 Starting vision processor")
            await self.vision_processor.start_capture()
        
        if service_registry.is_service_active("audio_processor"):
            logger.info("🎤 Starting audio processor")
            await self.audio_processor.start_listening()
        
        if service_registry.is_service_active("object_detection_manager") and self.object_detection_manager:
            logger.info("🎯 Object detection manager is already initialized and ready")
    
    async def _stop_active_services(self):
        """Stop currently active services"""
        logger.info(f"Stopping active services: {service_registry.active_services}")
        
        if service_registry.is_service_active("audio_processor"):
            logger.info("🎤 Stopping audio processor")
            await self.audio_processor.stop_listening()
        
        if service_registry.is_service_active("vision_processor"):
            logger.info("📹 Stopping vision processor")
            await self.vision_processor.stop_capture()
        
        if service_registry.is_service_active("llm_processor"):
            logger.info("🤖 Stopping LLM processor")
            await self.llm_processor.stop()
        
        if service_registry.is_service_active("object_detection_manager"):
            logger.info("🎯 Object detection manager remains available")
    
    async def _event_handler(self):
        """Main event handling loop"""
        logger.info("Event handler started")
        
        try:
            while self.is_running:
                try:
                    # Get next event (with timeout to allow cancellation)
                    event = await asyncio.wait_for(event_bus.get_event(), timeout=1.0)
                    await self._process_event(event)
                    
                except asyncio.TimeoutError:
                    # Normal timeout, continue loop
                    continue
                except asyncio.CancelledError:
                    logger.info("Event handler cancelled")
                    break
                except Exception as e:
                    logger.error(f"Event handler error: {e}")
                    # Continue processing other events
                    continue
                    
        except Exception as e:
            logger.error(f"Critical event handler error: {e}")
        finally:
            logger.info("Event handler stopped")
    
    async def _process_event(self, event: Event):
        """Process a single event using lookup table dispatch"""
        try:
            logger.debug(f"Processing event: {event.type.value}:{event.action}")
            
            # Filter events based on active service mode
            if not service_registry.should_process_event(event.type.value):
                logger.debug(f"Event {event.type.value} filtered out for current mode")
                return
            
            # Get handlers for this event type
            type_handlers = self.event_handlers.get(event.type)
            if not type_handlers:
                logger.debug(f"No handlers for event type: {event.type.value}")
                return
            
            # Find specific handler for this action
            handler = type_handlers.get(event.action)
            if not handler:
                # Check for wildcard handler
                handler = type_handlers.get("*")
            
            if handler:
                await handler(event)
            else:
                logger.debug(f"No handler for action: {event.action}")
                
        except Exception as e:
            logger.error(f"Error processing event {event.type.value}:{event.action}: {e}")
    
    # System status handlers
    async def _handle_system_listening(self, event: Event):
        """Handle system listening status"""
        logger.info("🎤 System started listening")
    
    async def _handle_system_camera_active(self, event: Event):
        """Handle camera activation"""
        logger.info("📹 Camera activated")
    
    async def _handle_system_camera_disabled(self, event: Event):
        """Handle camera disabled"""
        logger.info("📹 Camera disabled")
    
    async def _handle_system_whisper_loading(self, event: Event):
        """Handle Whisper model loading"""
        logger.info("🔄 Loading Whisper model...")
    
    async def _handle_system_whisper_ready(self, event: Event):
        """Handle Whisper model ready"""
        model = event.data.get("model", "unknown")
        logger.info(f"✅ Whisper model ready: {model}")
    
    async def _handle_system_ready(self, event: Event):
        """Handle system ready"""
        logger.info("🚀 Osmo Assistant is ready!")
    
    # Audio event handlers
    async def _handle_audio_transcription_start(self, event: Event):
        """Handle transcription start"""
        logger.debug("🎯 Starting transcription...")
    
    async def _handle_audio_transcription_end(self, event: Event):
        """Handle transcription end"""
        logger.debug("✅ Transcription completed")
    
    async def _handle_audio_raw_transcript(self, event: Event):
        """Handle raw transcript"""
        transcript = event.data.get("transcript", "")
        logger.debug(f"📝 Raw transcript: '{transcript}'")
    
    async def _handle_audio_wake_word(self, event: Event):
        """Handle wake word detection"""
        logger.info("🎤 Wake word detected - listening for context...")
    
    async def _handle_audio_context_ready(self, event: Event):
        """Handle context ready - this is where we send to LLM"""
        transcript = event.data.get("transcript", "")
        logger.info(f"📝 Context ready: '{transcript}'")
        
        if transcript:
            # Process with LLM (inject vision processor dependency)
            await self.llm_processor.process_query(transcript, self.vision_processor)
        else:
            logger.warning("Empty context received")
    
    # LLM event handlers
    async def _handle_llm_start(self, event: Event):
        """Handle LLM response start"""
        transcript = event.data.get("transcript", "")
        has_image = event.data.get("has_image", False)
        
        logger.info(f"🤖 LLM processing: '{transcript}' (vision: {has_image})")
    
    async def _handle_llm_chunk(self, event: Event):
        """Handle LLM response chunk"""
        # Just log for debugging, frontend handles streaming
        chunk = event.data.get("chunk", "")
        logger.debug(f"🔄 LLM chunk: {len(chunk)} chars")
    
    async def _handle_llm_end(self, event: Event):
        """Handle LLM response end"""
        response = event.data.get("full_response", "")
        logger.info(f"✅ LLM response complete: {len(response)} characters")
    
    # TTS event handlers
    async def _handle_tts_start(self, event: Event):
        """Handle TTS start"""
        text = event.data.get("text", "")
        logger.info(f"🔊 Starting TTS: {len(text)} characters")
    
    async def _handle_tts_end(self, event: Event):
        """Handle TTS end"""
        logger.info("🔇 TTS completed")
    
    # Tool event handlers
    async def _handle_tool_start(self, event: Event):
        """Handle tool start"""
        tool_name = event.data.get("tool", "unknown")
        args = event.data.get("args", {})
        logger.info(f"🔧 Starting tool: {tool_name} with args: {args}")
    
    async def _handle_tool_complete(self, event: Event):
        """Handle tool complete"""
        tool_name = event.data.get("tool", "unknown")
        success = event.data.get("success", False)
        status = "✅" if success else "❌"
        logger.info(f"{status} Tool completed: {tool_name}")
    
    async def _handle_tool_error(self, event: Event):
        """Handle tool error"""
        tool_name = event.data.get("tool", "unknown")
        error_msg = event.data.get("error", "Unknown error")
        logger.error(f"❌ Tool error [{tool_name}]: {error_msg}")
    
    async def _handle_photo_start(self, event: Event):
        """Handle photo start"""
        tool_name = event.data.get("tool", "photo")
        logger.info(f"📸 Capturing photo...")
    
    async def _handle_photo_complete(self, event: Event):
        """Handle photo complete"""
        success = event.data.get("success", False)
        status = "✅" if success else "❌"
        logger.info(f"📸 {status} Photo capture completed")
    
    async def _handle_video_start(self, event: Event):
        """Handle video start"""
        duration = event.data.get("duration", 3)
        logger.info(f"🎥 Starting video recording ({duration}s)...")
    
    async def _handle_video_complete(self, event: Event):
        """Handle video complete"""
        duration = event.data.get("duration", 3)
        success = event.data.get("success", False)
        status = "✅" if success else "❌"
        logger.info(f"🎥 {status} Video recording completed ({duration}s)")
    
    async def _handle_recording_start(self, event: Event):
        """Handle recording start"""
        duration = event.data.get("duration", 3)
        logger.debug(f"🎬 Recording frames for {duration}s...")
    
    async def _handle_recording_complete(self, event: Event):
        """Handle recording complete"""
        duration = event.data.get("duration", 3)
        file_size = event.data.get("file_size", 0)
        logger.debug(f"🎬 Recording completed: {duration}s, {file_size} bytes")
    
    # Voice control handlers
    async def _handle_voice_dictation_toggled(self, event: Event):
        """Handle voice dictation toggle"""
        enabled = event.data.get("enabled", False)
        status = "🎤" if enabled else "🔇"
        logger.info(f"{status} Voice dictation {'enabled' if enabled else 'disabled'}")
    
    # Camera control handlers
    async def _handle_camera_capture_toggled(self, event: Event):
        """Handle camera capture toggle"""
        enabled = event.data.get("enabled", False)
        status = "📸" if enabled else "🔇"
        logger.info(f"{status} Camera capture {'enabled' if enabled else 'disabled'}")
    
    # TTS control handlers
    async def _handle_tts_toggled(self, event: Event):
        """Handle TTS toggle"""
        enabled = event.data.get("enabled", False)
        status = "🔊" if enabled else "🔇"
        logger.info(f"{status} TTS {'enabled' if enabled else 'disabled'}")
    
    # Error handler
    async def _handle_error(self, event: Event):
        """Handle error events"""
        error_msg = event.data.get("error", "Unknown error")
        action = event.action
        logger.error(f"❌ Error [{action}]: {error_msg}")
    
    # Object detection event handlers
    async def _handle_object_detection_started(self, event: Event):
        """Handle object detection started"""
        video_id = event.data.get("video_id", "unknown")
        logger.info(f"🎯 Object detection started for video: {video_id}")
    
    async def _handle_object_detection_result(self, event: Event):
        """Handle object detection result"""
        detected_objects = event.data.get("detectedObjects", "none")
        latency = event.data.get("latency", 0)
        logger.info(f"🎯 Objects detected: {detected_objects} (latency: {latency:.2f}ms)")
    
    async def _handle_object_detection_completed(self, event: Event):
        """Handle object detection completed"""
        total_frames = event.data.get("total_frames", 0)
        avg_latency = event.data.get("avg_latency", 0)
        logger.info(f"✅ Object detection completed: {total_frames} frames, avg latency: {avg_latency:.2f}ms")
    
    async def _handle_object_detection_stopped(self, event: Event):
        """Handle object detection stopped"""
        logger.info("🛑 Object detection stopped")
    
    async def _handle_object_detection_error(self, event: Event):
        """Handle object detection error"""
        error_msg = event.data.get("error", "Unknown error")
        logger.error(f"❌ Object detection error: {error_msg}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            "is_running": self.is_running,
            "audio_listening": self.audio_processor.is_listening if self.audio_processor else False,
            "voice_dictation_enabled": self.audio_processor.is_voice_dictation_enabled() if self.audio_processor else False,
            "vision_capturing": self.vision_processor.is_capturing if self.vision_processor else False,
            "camera_capture_enabled": self.vision_processor.is_camera_capture_enabled() if self.vision_processor else False,
            "llm_processing": self.llm_processor.is_processing if self.llm_processor else False,
            "whisper_loaded": self.audio_processor.whisper_model_loaded if self.audio_processor else False,
            "tts_enabled": self.llm_processor.is_tts_enabled() if self.llm_processor else False,
        }

    # ================== CONTROL METHODS ==================
    # These methods provide centralized control through the task manager
    
    async def toggle_voice_dictation(self, enabled: bool) -> Dict[str, Any]:
        """Toggle voice dictation through task manager"""
        try:
            await self.audio_processor.set_voice_dictation_enabled(enabled)
            
            # Publish control event
            await event_bus.publish(Event(
                type=EventType.VOICE_CONTROL,
                action="dictation_toggled",
                data={"enabled": enabled, "message": f"Voice dictation {'enabled' if enabled else 'disabled'}"}
            ))
            
            return {
                "success": True,
                "enabled": enabled,
                "message": f"Voice dictation {'enabled' if enabled else 'disabled'}"
            }
        except Exception as e:
            logger.error(f"Failed to toggle voice dictation: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to toggle voice dictation: {e}"
            }
    
    async def toggle_camera_capture(self, enabled: bool) -> Dict[str, Any]:
        """Toggle camera capture through task manager"""
        try:
            await self.vision_processor.set_camera_capture_enabled(enabled)
            
            # Publish control event
            await event_bus.publish(Event(
                type=EventType.CAMERA_CONTROL,
                action="capture_toggled",
                data={"enabled": enabled, "message": f"Camera capture {'enabled' if enabled else 'disabled'}"}
            ))
            
            return {
                "success": True,
                "enabled": enabled,
                "message": f"Camera capture {'enabled' if enabled else 'disabled'}"
            }
        except Exception as e:
            logger.error(f"Failed to toggle camera capture: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to toggle camera capture: {e}"
            }
    
    async def toggle_tts(self, enabled: bool) -> Dict[str, Any]:
        """Toggle TTS through task manager"""
        try:
            await self.llm_processor.set_tts_enabled(enabled)
            
            # Publish control event
            await event_bus.publish(Event(
                type=EventType.TTS_CONTROL,
                action="tts_toggled",
                data={"enabled": enabled, "message": f"TTS {'enabled' if enabled else 'disabled'}"}
            ))
            
            return {
                "success": True,
                "enabled": enabled,
                "message": f"TTS {'enabled' if enabled else 'disabled'}"
            }
        except Exception as e:
            logger.error(f"Failed to toggle TTS: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to toggle TTS: {e}"
            }
    
    async def update_audio_device(self, device_id: int) -> Dict[str, Any]:
        """Update audio device through task manager"""
        try:
            # Import here to avoid circular imports
            from ..config import settings
            
            # Update settings
            settings.audio_device_index = device_id
            
            # Restart audio processor if currently listening
            was_listening = self.audio_processor.is_listening
            if was_listening:
                await self.audio_processor.stop_listening()
                await self.audio_processor.start_listening()
            
            # Publish event
            await event_bus.publish(Event(
                type=EventType.SYSTEM_STATUS,
                action="audio_device_changed",
                data={
                    "device_id": device_id,
                    "was_listening": was_listening,
                    "message": f"Audio device changed to device {device_id}"
                }
            ))
            
            return {
                "success": True,
                "device_id": device_id,
                "message": f"Audio device updated successfully"
            }
        except Exception as e:
            logger.error(f"Failed to update audio device: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to update audio device: {e}"
            }
    
    async def update_video_device(self, device_id: int) -> Dict[str, Any]:
        """Update video device through task manager"""
        try:
            # Import here to avoid circular imports
            from ..config import settings
            
            # Update settings
            settings.camera_index = device_id
            
            # Restart vision processor if currently capturing
            was_capturing = self.vision_processor.is_capturing
            if was_capturing:
                await self.vision_processor.stop_capture()
                await self.vision_processor.start_capture()
            
            # Publish event
            await event_bus.publish(Event(
                type=EventType.SYSTEM_STATUS,
                action="video_device_changed",
                data={
                    "device_id": device_id,
                    "was_capturing": was_capturing,
                    "message": f"Video device changed to device {device_id}"
                }
            ))
            
            return {
                "success": True,
                "device_id": device_id,
                "message": f"Video device updated successfully"
            }
        except Exception as e:
            logger.error(f"Failed to update video device: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to update video device: {e}"
            }


# Remove global instance - will be handled by dependency injection
