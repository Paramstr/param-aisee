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
import webrtcvad

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
        
        # VAD state management
        self.vad_state = "IDLE"  # IDLE, SPEAKING, SILENCE_DETECTED
        self.speech_frames = []  # Accumulated speech frames
        self.silence_frame_count = 0
        self.speech_frame_count = 0
        
        # Context accumulation state
        self.context_mode = False  # Whether we're accumulating context after wake word
        self.context_buffer: List[str] = []  # Accumulated context after wake word
        self.last_speech_time = 0  # Track when last speech was detected
        
        # Audio buffers
        self.audio_buffer = deque(maxlen=settings.audio_sample_rate * 5)  # 5 seconds buffer
        
        # MLX Whisper model configuration
        self.whisper_model_path = "mlx-community/whisper-large-v3-turbo"
        self.whisper_model_loaded = False
        
        # VAD configuration
        self.vad = webrtcvad.Vad(settings.vad_aggressiveness)  # Use configurable aggressiveness
        self.frame_duration_ms = 30  # 30ms frames for VAD
        self.frame_size = int(settings.audio_sample_rate * self.frame_duration_ms / 1000)
        
        # VAD thresholds
        self.speech_threshold = 3  # Frames of speech to trigger recording
        self.silence_threshold_frames = 40  # Frames of silence to end recording (~1.2 seconds)
        self.min_speech_duration_frames = 10  # Minimum speech duration to transcribe
        
        # Threading
        self.audio_thread: Optional[threading.Thread] = None
        self.should_stop = False
        
        # Wake word keywords (case-insensitive)
        self.wake_words = ["hey", "osmo", "hey osmo", "chat", "hey chat", "hey aisee"]
        
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
        
        logger.info("Audio processor started with VAD")
        await event_bus.publish(Event(
            type=EventType.SYSTEM_STATUS,
            action="listening",
            data={"message": "Listening for speech with VAD"}
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
                blocksize=self.frame_size,  # Use VAD frame size
                callback=self._audio_callback
            ):
                while not self.should_stop:
                    time.sleep(0.01)  # Shorter sleep for more responsive VAD
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
        """Audio callback for sounddevice with VAD processing"""
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        audio_data = indata.flatten().astype(np.int16)
        self.audio_buffer.extend(audio_data)
        
        if not self.is_transcribing:
            # Process audio frame through VAD
            if self.loop:
                asyncio.run_coroutine_threadsafe(
                    self._process_vad_frame(audio_data),
                    self.loop
                )

    async def _process_vad_frame(self, audio_frame: np.ndarray):
        """Process audio frame through VAD state machine"""
        # Ensure frame is the right size for VAD
        if len(audio_frame) != self.frame_size:
            return
        
        # Convert to bytes for VAD
        audio_bytes = audio_frame.tobytes()
        
        try:
            # Check if frame contains speech
            is_speech = self.vad.is_speech(audio_bytes, settings.audio_sample_rate)
            
            if is_speech:
                self.speech_frame_count += 1
                self.silence_frame_count = 0
                
                # Always add speech frames to buffer
                self.speech_frames.extend(audio_frame)
                
                # Transition to SPEAKING state if we have enough speech
                if self.vad_state == "IDLE" and self.speech_frame_count >= self.speech_threshold:
                    self.vad_state = "SPEAKING"
                    logger.debug("VAD: Started speaking")
                    
                    await event_bus.publish(Event(
                        type=EventType.AUDIO_EVENT,
                        action="speech_start",
                        data={"message": "Speech detected"}
                    ))
                
            else:
                # No speech detected
                self.speech_frame_count = 0
                
                if self.vad_state == "SPEAKING":
                    self.silence_frame_count += 1
                    
                    # Still add some silence frames to capture end of words
                    if self.silence_frame_count <= 10:  # Add up to 300ms of silence
                        self.speech_frames.extend(audio_frame)
                    
                    # Check if we've had enough silence to end recording
                    if self.silence_frame_count >= self.silence_threshold_frames:
                        self.vad_state = "SILENCE_DETECTED"
                        logger.debug("VAD: Silence detected, ending speech")
                        
                        await event_bus.publish(Event(
                            type=EventType.AUDIO_EVENT,
                            action="speech_end",
                            data={"message": "Silence detected"}
                        ))
                        
                        # Process the accumulated speech
                        await self._process_speech_segment()
                
                elif self.vad_state == "IDLE":
                    # Reset any accumulated frames during idle
                    self.speech_frames = []
        
        except Exception as e:
            logger.error(f"VAD processing error: {e}")

    async def _process_speech_segment(self):
        """Process accumulated speech segment"""
        if not self.speech_frames or len(self.speech_frames) < self.min_speech_duration_frames * self.frame_size:
            # Reset state
            self._reset_vad_state()
            return
        
        # Copy the speech data
        speech_data = self.speech_frames.copy()
        
        # Reset VAD state
        self._reset_vad_state()
        
        # Transcribe the speech segment
        await self._transcribe_speech_segment(speech_data)

    def _reset_vad_state(self):
        """Reset VAD state machine"""
        self.vad_state = "IDLE"
        self.speech_frames = []
        self.silence_frame_count = 0
        self.speech_frame_count = 0

    async def _transcribe_speech_segment(self, audio_data: List[int]):
        """Transcribe speech segment without temporary files"""
        if not self.whisper_model_loaded:
            return
        
        self.is_transcribing = True
        
        try:
            await event_bus.publish(Event(
                type=EventType.AUDIO_EVENT,
                action="transcription_start",
                data={"duration_ms": len(audio_data) * 1000 // settings.audio_sample_rate}
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

    async def _handle_transcript(self, transcript: str):
        """Handle transcript with wake word detection and context accumulation"""
        if not transcript:
            return
        
        current_time = time.time()
        transcript_lower = transcript.lower() # For checking "enter"
        
        # Check for wake word
        if self._contains_wake_word(transcript): # Pass original transcript for wake word check
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
            
            # Add the full transcript including wake word
            # Don't add if it's just the wake word and nothing else useful.
            # Or, decide if the "enter" command itself should be part of the context.
            # For now, let's assume "enter" itself is not part of the context sent to LLM.
            # If the transcript *is* "enter", it will be handled below.
            # If the transcript *contains* a wake word but isn't *just* "enter", add it.
            if transcript_lower != "enter.": # common ASR artifact for "enter"
                 # also check for "enter" without the period.
                if transcript_lower != "enter":
                    self.context_buffer.append(transcript)
            
        elif self.context_mode:
            # We're in context accumulation mode
            # Check if the command is "enter"
            if transcript_lower == "enter." or transcript_lower == "enter":
                logger.info(f'"Enter" command detected. Finalizing context.')
                # Don't add "enter" to the buffer
                await self._finalize_context()
                return # Context finalized, no further processing in this handler for this transcript

            # Not "enter", so add to context buffer
            self.context_buffer.append(transcript)
            self.last_speech_time = current_time
            
        # Check if we should end context accumulation (silence timeout)
        # This check should only happen if context mode is still active (i.e., "enter" wasn't said)
        if self.context_mode and (current_time - self.last_speech_time) > self.silence_threshold:
            logger.info(f"Silence threshold reached. Finalizing context.")
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


# Remove global instance - now handled by dependency injection
