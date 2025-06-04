import asyncio
import logging
import numpy as np
import sounddevice as sd
import mlx_whisper
from typing import Optional, List
import threading
import time
from collections import deque
import concurrent.futures
import io

from ..config import settings
from ..events import Event, EventType, event_bus

logger = logging.getLogger(__name__)


class AudioProcessor:
    def __init__(self, cpu_pool: concurrent.futures.ThreadPoolExecutor, loop: asyncio.AbstractEventLoop):
        # Dependency injection
        self.cpu_pool = cpu_pool
        self.loop = loop
        
        # State management
        self.is_listening = False
        self.is_transcribing = False
        self.is_recording = False
        
        # Context accumulation state
        self.context_mode = False  # Whether we're accumulating context after wake word
        self.context_buffer: List[str] = []  # Accumulated context after wake word
        self.last_speech_time = 0  # Track when last speech was detected
        
        # Audio buffers
        self.audio_buffer = deque(maxlen=settings.audio_sample_rate * 5)  # 5 seconds buffer
        self.recording_buffer = []
        
        # MLX Whisper model configuration
        self.whisper_model_path = "mlx-community/whisper-large-v3-turbo"
        self.whisper_model_loaded = False
        
        # Threading
        self.audio_thread: Optional[threading.Thread] = None
        self.should_stop = False
        
        # Wake word keywords (case-insensitive)
        self.wake_words = ["osmo", "hey osmo", "testing"]
        
        # Audio chunking settings - now 10 seconds
        self.chunk_duration = 10.0  # Process every 10 seconds
        self.chunk_frames = int(settings.audio_sample_rate * self.chunk_duration)
        
        # Context accumulation settings
        self.silence_threshold = 2.0  # 2 seconds of silence to end context
    
    async def initialize(self):
        """Initialize the audio processor"""
        await self._init_whisper()
    
    async def _init_whisper(self):
        """Initialize MLX Whisper model"""
        try:
            await event_bus.publish(Event(
                type=EventType.SYSTEM_STATUS,
                action="whisper_loading",
                data={"message": "Loading MLX Whisper model..."}
            ))
            
            def load_whisper():
                try:
                    # Test load the model without creating temp files
                    test_audio = np.random.randint(-1000, 1000, 16000, dtype=np.int16)
                    
                    # Convert to float32 for MLX Whisper (no temp file needed)
                    audio_float = test_audio.astype(np.float32) / 32768.0
                    
                    # Test transcription
                    result = mlx_whisper.transcribe(
                        audio_float, 
                        path_or_hf_repo=self.whisper_model_path
                    )
                    
                    if "text" in result:
                        logger.info(f"✅ MLX Whisper model loaded: {self.whisper_model_path}")
                        return True
                    return False
                except Exception as e:
                    logger.error(f"MLX Whisper loading failed: {e}")
                    return False
            
            success = await self.loop.run_in_executor(self.cpu_pool, load_whisper)
            
            if success:
                self.whisper_model_loaded = True
                await event_bus.publish(Event(
                    type=EventType.SYSTEM_STATUS,
                    action="whisper_ready",
                    data={"model": self.whisper_model_path}
                ))
            else:
                await event_bus.publish(Event(
                    type=EventType.ERROR,
                    action="whisper_failed",
                    data={"error": "Model verification failed"}
                ))
                
        except Exception as e:
            logger.error(f"Failed to initialize MLX Whisper: {e}")
            await event_bus.publish(Event(
                type=EventType.ERROR,
                action="whisper_failed", 
                data={"error": str(e)}
            ))

    async def start_listening(self):
        """Start continuous audio processing"""
        if self.is_listening:
            return
        
        self.is_listening = True
        self.should_stop = False
        
        self.audio_thread = threading.Thread(target=self._audio_loop, daemon=True)
        self.audio_thread.start()
        
        logger.info("Audio processor started")
        await event_bus.publish(Event(
            type=EventType.SYSTEM_STATUS,
            action="listening",
            data={"message": "Listening for speech"}
        ))

    async def stop_listening(self):
        """Stop audio processing"""
        self.should_stop = True
        self.is_listening = False
        
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=2.0)

    def _audio_loop(self):
        """Main audio processing loop"""
        try:
            with sd.InputStream(
                channels=settings.audio_channels,
                samplerate=settings.audio_sample_rate,
                dtype=np.int16,
                blocksize=settings.audio_chunk_size,
                callback=self._audio_callback
            ):
                while not self.should_stop:
                    time.sleep(0.1)
        except Exception as e:
            logger.error(f"Audio loop error: {e}")
            if self.loop:
                asyncio.run_coroutine_threadsafe(
                    event_bus.publish(Event(
                        type=EventType.ERROR,
                        action="audio_loop_error",
                        data={"error": str(e)}
                    )),
                    self.loop
                )

    def _audio_callback(self, indata, frames, time, status):
        """Audio callback for sounddevice"""
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        audio_data = indata.flatten().astype(np.int16)
        self.audio_buffer.extend(audio_data)
        
        if not self.is_transcribing:
            self.recording_buffer.extend(audio_data)
            
            # Process in 10-second chunks
            if len(self.recording_buffer) >= self.chunk_frames:
                if self.loop:
                    asyncio.run_coroutine_threadsafe(
                        self._process_audio_chunk(),
                        self.loop
                    )

    async def _process_audio_chunk(self):
        """Process 10-second audio chunk"""
        if self.is_transcribing or not self.recording_buffer:
            return
        
        chunk_data = self.recording_buffer.copy()
        self.recording_buffer = []
        self.is_recording = True
        
        await self._transcribe_audio_chunk(chunk_data)

    async def _transcribe_audio_chunk(self, audio_data: List[int]):
        """Transcribe audio chunk without temporary files"""
        if not self.whisper_model_loaded:
            return
        
        self.is_transcribing = True
        
        try:
            await event_bus.publish(Event(
                type=EventType.AUDIO_EVENT,
                action="transcription_start",
                data={}
            ))
            
            def transcribe_audio():
                try:
                    # Convert to numpy array and normalize to float32
                    audio_array = np.array(audio_data, dtype=np.int16)
                    audio_float = audio_array.astype(np.float32) / 32768.0
                    
                    # Direct transcription without temp files
                    result = mlx_whisper.transcribe(
                        audio_float, 
                        path_or_hf_repo=self.whisper_model_path
                    )
                    
                    return result["text"].strip()
                except Exception as e:
                    logger.error(f"Transcription error: {e}")
                    return None
            
            transcript = await self.loop.run_in_executor(self.cpu_pool, transcribe_audio)
            
            if transcript is not None:
                print(f"[TRANSCRIPTION] {transcript}")
                
                await event_bus.publish(Event(
                    type=EventType.AUDIO_EVENT,
                    action="raw_transcript",
                    data={"transcript": transcript}
                ))
                
                # Handle wake word and context logic
                await self._handle_transcript(transcript)
                
        except Exception as e:
            logger.error(f"Transcription error: {e}")
        finally:
            await event_bus.publish(Event(
                type=EventType.AUDIO_EVENT,
                action="transcription_end",
                data={}
            ))
            self.is_transcribing = False
            self.is_recording = False

    async def _handle_transcript(self, transcript: str):
        """Handle transcript with wake word detection and context accumulation"""
        if not transcript:
            return
        
        current_time = time.time()
        
        # Check for wake word
        if self._contains_wake_word(transcript):
            logger.info(f"Wake word detected: {transcript}")
            
            # Start context accumulation mode
            self.context_mode = True
            self.context_buffer = []
            self.last_speech_time = current_time
            
            await event_bus.publish(Event(
                type=EventType.AUDIO_EVENT,
                action="wake_word_detected",
                data={"transcript": transcript}
            ))
            
            # Extract command after wake word
            cleaned_transcript = self._clean_wake_word_from_transcript(transcript)
            if cleaned_transcript:
                self.context_buffer.append(cleaned_transcript)
            
        elif self.context_mode:
            # We're in context accumulation mode
            self.context_buffer.append(transcript)
            self.last_speech_time = current_time
            
        # Check if we should end context accumulation (silence timeout)
        if self.context_mode and (current_time - self.last_speech_time) > self.silence_threshold:
            await self._finalize_context()

    async def _finalize_context(self):
        """Finalize accumulated context and send to LLM"""
        if not self.context_buffer:
            self.context_mode = False
            return
        
        # Combine all context into a single query
        full_context = " ".join(self.context_buffer).strip()
        
        logger.info(f"Context finalized: '{full_context}'")
        
        await event_bus.publish(Event(
            type=EventType.AUDIO_EVENT,
            action="context_ready",
            data={"transcript": full_context}
        ))
        
        # Reset context accumulation
        self.context_mode = False
        self.context_buffer = []

    def _contains_wake_word(self, transcript: str) -> bool:
        """Check if transcript contains wake word"""
        transcript_lower = transcript.lower()
        return any(wake_word in transcript_lower for wake_word in self.wake_words)
    
    def _clean_wake_word_from_transcript(self, transcript: str) -> str:
        """Remove wake word from transcript"""
        cleaned = transcript.lower()
        
        # Remove wake words
        for wake_word in self.wake_words:
            if cleaned.startswith(wake_word):
                cleaned = cleaned[len(wake_word):].strip()
                break
        
        for wake_word in self.wake_words:
            cleaned = cleaned.replace(wake_word, "").strip()
        
        return " ".join(cleaned.split())


# Remove global instance - now handled by dependency injection
