import asyncio
import logging
from typing import Optional, Dict, Callable, Any

from ..events import Event, EventType, event_bus

logger = logging.getLogger(__name__)


class TaskManager:
    def __init__(self, audio_processor, vision_processor, llm_processor):
        # Dependency injection - no more global imports
        self.audio_processor = audio_processor
        self.vision_processor = vision_processor
        self.llm_processor = llm_processor
        
        self.is_running = False
        self.event_handler_task: Optional[asyncio.Task] = None
        
        # Lookup table for event handlers - much cleaner than if/elif chain
        self.event_handlers: Dict[EventType, Dict[str, Callable]] = {
            EventType.SYSTEM_STATUS: {
                "listening": self._handle_system_listening,
                "camera_active": self._handle_system_camera_active,
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
            # Start all processors
            await self.llm_processor.start()
            await self.vision_processor.start_capture()
            await self.audio_processor.start_listening()
            
            # Start event handler
            self.event_handler_task = asyncio.create_task(self._event_handler())
            
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
    
    # Error handler
    async def _handle_error(self, event: Event):
        """Handle error events"""
        error_msg = event.data.get("error", "Unknown error")
        action = event.action
        logger.error(f"❌ Error [{action}]: {error_msg}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            "is_running": self.is_running,
            "audio_listening": self.audio_processor.is_listening if self.audio_processor else False,
            "vision_capturing": self.vision_processor.is_capturing if self.vision_processor else False,
            "llm_processing": self.llm_processor.is_processing if self.llm_processor else False,
            "whisper_loaded": self.audio_processor.whisper_model_loaded if self.audio_processor else False,
        }


# Remove global instance - will be handled by dependency injection
