from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # OpenRouter API configuration
    openrouter_api_key: str
    openrouter_model: str = "qwen/qwen2.5-vl-72b-instruct:free"

    
    # Audio configuration
    audio_sample_rate: int = 16000
    audio_channels: int = 1
    audio_chunk_size: int = 1024
    audio_device_index: Optional[int] = None  # None = use system default device
    vad_aggressiveness: int = 3  # WebRTC VAD aggressiveness (0-3)
    silence_duration_threshold: float = 2.0  # seconds of silence to stop recording
    
    # Vision configuration
    camera_index: int = 0  # Default to first available camera
    camera_width: int = 640   # Start with VGA resolution (widely supported)
    camera_height: int = 480  # VGA height
    camera_fps: int = 30      # 30fps is widely supported (60fps often fails)
    
    # LLM configuration
    max_tokens: int = 1000
    temperature: float = 0.7
    
    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["http://localhost:3000"]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()
