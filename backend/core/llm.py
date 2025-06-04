import asyncio
import logging
import aiohttp
import json
import subprocess
import threading
from typing import Optional, AsyncGenerator, Dict, Any
import base64
import concurrent.futures

from ..config import settings
from ..events import Event, EventType, event_bus
from .conversation import conversation_storage

logger = logging.getLogger(__name__)


class LLMProcessor:
    def __init__(self, io_pool: concurrent.futures.ThreadPoolExecutor):
        # Dependency injection
        self.io_pool = io_pool
        
        self.session: Optional[aiohttp.ClientSession] = None
        self.is_processing = False
        
        # TTS state
        self.tts_process: Optional[subprocess.Popen] = None
        
    async def start(self):
        """Initialize the LLM processor"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=60)
            self.session = aiohttp.ClientSession(timeout=timeout)
        logger.info("LLM processor started")
    
    async def stop(self):
        """Stop the LLM processor"""
        if self.session:
            await self.session.close()
            self.session = None
        
        # Stop any running TTS
        if self.tts_process:
            self.tts_process.terminate()
            self.tts_process = None
        
        logger.info("LLM processor stopped")
    
    async def process_query(self, transcript: str, vision_processor=None):
        """Process a user query with current camera frame"""
        if self.is_processing:
            logger.warning("LLM already processing, skipping request")
            return
        
        self.is_processing = True
        
        try:
            # Store user message
            conversation_storage.add_message("user", transcript)
            
            # Get current frame for context
            image_base64 = None
            if vision_processor:
                image_base64 = vision_processor.capture_frame_for_llm()
            
            # Build the prompt with conversation context
            prompt = self._build_prompt(transcript, image_base64 is not None)
            
            await event_bus.publish(Event(
                type=EventType.LLM_EVENT,
                action="response_start",
                data={
                    "transcript": transcript,
                    "has_image": image_base64 is not None
                }
            ))
            
            # Call LLM API
            response_text = ""
            async for chunk in self._call_llm_api(prompt, image_base64):
                response_text += chunk
                await event_bus.publish(Event(
                    type=EventType.LLM_EVENT,
                    action="response_chunk",
                    data={"chunk": chunk}
                ))
            
            # Store assistant response
            if response_text.strip():
                conversation_storage.add_message("assistant", response_text)
            
            await event_bus.publish(Event(
                type=EventType.LLM_EVENT,
                action="response_end",
                data={"full_response": response_text}
            ))
            
            # Start TTS
            if response_text.strip():
                await self._speak_text(response_text)
            
        except Exception as e:
            logger.error(f"LLM processing error: {e}")
            await event_bus.publish(Event(
                type=EventType.ERROR,
                action="llm_processing_failed",
                data={"error": f"LLM processing failed: {e}"}
            ))
        finally:
            self.is_processing = False
    
    def _build_prompt(self, transcript: str, has_image: bool) -> str:
        """Build the system prompt for the LLM"""
        base_prompt = """You are Osmo, an AI assistant with vision and speech capabilities. You can see through a camera and respond to user queries about what you observe.

Key traits:
- Be conversational and helpful
- Keep responses concise but informative (1-3 sentences typically)
- If you can see something in the image, describe it naturally
- If asked about something not visible, explain what you can see instead
- Be friendly and engaging
- Remember previous parts of our conversation for context

"""
        
        # Add conversation context (optimized)
        conversation_context = conversation_storage.get_conversation_context(limit=5)
        if conversation_context:
            base_prompt += f"Recent conversation:\n{conversation_context}\n\n"
        
        if has_image:
            base_prompt += "You can see the current camera view. "
        else:
            base_prompt += "Camera view is not available. "
        
        base_prompt += f"User said: \"{transcript}\""
        
        return base_prompt
    
    async def _call_llm_api(self, prompt: str, image_base64: Optional[str]) -> AsyncGenerator[str, None]:
        """Call OpenRouter API and stream response"""
        if not self.session:
            raise Exception("LLM session not initialized")
        
        url = "https://openrouter.ai/api/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "Osmo Assistant"
        }
        
        # Build messages
        messages = []
        
        if image_base64:
            # Vision-enabled message
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            })
        else:
            # Text-only message
            messages.append({
                "role": "user",
                "content": prompt
            })
        
        payload = {
            "model": settings.openrouter_model,
            "messages": messages,
            "max_tokens": settings.max_tokens,
            "temperature": settings.temperature,
            "stream": True
        }
        
        try:
            async with self.session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API error {response.status}: {error_text}")
                
                # Stream the response
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    
                    if line.startswith('data: '):
                        data = line[6:]  # Remove 'data: ' prefix
                        
                        if data == '[DONE]':
                            break
                        
                        try:
                            chunk_data = json.loads(data)
                            delta = chunk_data.get('choices', [{}])[0].get('delta', {})
                            content = delta.get('content', '')
                            
                            if content:
                                yield content
                                
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            logger.error(f"API call error: {e}")
            raise
    
    async def _speak_text(self, text: str):
        """Convert text to speech using macOS 'say' command"""
        try:
            await event_bus.publish(Event(
                type=EventType.TTS_EVENT,
                action="start",
                data={"text": text}
            ))
            
            # Clean text for TTS
            clean_text = self._clean_text_for_tts(text)
            
            # Start TTS in background thread
            tts_thread = threading.Thread(target=self._tts_worker, args=(clean_text,), daemon=True)
            tts_thread.start()
            
        except Exception as e:
            logger.error(f"TTS error: {e}")
            await event_bus.publish(Event(
                type=EventType.ERROR,
                action="tts_failed",
                data={"error": f"TTS failed: {e}"}
            ))
    
    def _clean_text_for_tts(self, text: str) -> str:
        """Clean text for better TTS pronunciation"""
        # Remove markdown and special characters
        text = text.replace('*', '').replace('_', '').replace('#', '')
        text = text.replace('\n', ' ').replace('\r', ' ')
        
        # Replace common abbreviations
        replacements = {
            'AI': 'A I',
            'API': 'A P I',
            'URL': 'U R L',
            'USB': 'U S B',
            'CPU': 'C P U',
            'GPU': 'G P U'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text.strip()
    
    def _tts_worker(self, text: str):
        """TTS worker function (runs in thread)"""
        try:
            # Use macOS say command
            self.tts_process = subprocess.Popen(
                ["say", text],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # Wait for completion
            self.tts_process.wait()
            
            # Publish completion event
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(
                    event_bus.publish(Event(
                        type=EventType.TTS_EVENT,
                        action="end",
                        data={}
                    ))
                )
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"TTS worker error: {e}")
        finally:
            self.tts_process = None


# Remove global instance - will be handled by dependency injection
