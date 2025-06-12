import asyncio
import logging
from typing import Dict, Set, Optional
from enum import Enum

logger = logging.getLogger(__name__)

class ServiceMode(Enum):
    OSMO = "osmo"
    OBJECT_DEMO = "object_demo"

class ServiceRegistry:
    """Manages which services are active based on current mode"""
    
    def __init__(self):
        # Define service categories
        self.service_tags: Dict[str, Set[str]] = {
            "osmo": {
                "audio_processor",
                "vision_processor", 
                "llm_processor",
                "tool_registry"
            },
            "object_demo": {
                "object_detection_manager",
                "vision_processor"  # Need camera for real-time detection
            }
        }
        
        # Current active mode
        self.active_mode: Optional[ServiceMode] = None
        self.active_services: Set[str] = set()
        
        # Event type filtering by mode
        self.allowed_events: Dict[str, Set[str]] = {
            "osmo": {
                "system_status", "audio_event", "llm_event", 
                "tts_event", "tool_event", "voice_control", 
                "camera_control", "error"
            },
            "object_demo": {
                "system_status", "object_demo", "camera_control", "error"
            }
        }
    
    async def set_mode(self, mode: ServiceMode) -> Dict:
        """Switch to a specific mode, enabling/disabling services accordingly"""
        if self.active_mode == mode:
            return {"message": f"Already in {mode.value} mode", "mode": mode.value}
        
        old_mode = self.active_mode
        self.active_mode = mode
        self.active_services = self.service_tags[mode.value].copy()
        
        logger.info(f"Switched from {old_mode.value if old_mode else 'none'} to {mode.value} mode")
        logger.info(f"Active services: {self.active_services}")
        
        return {
            "message": f"Switched to {mode.value} mode", 
            "mode": mode.value,
            "active_services": list(self.active_services)
        }
    
    def is_service_active(self, service_name: str) -> bool:
        """Check if a service is currently active"""
        return service_name in self.active_services
    
    def should_process_event(self, event_type: str) -> bool:
        """Check if an event type should be processed in current mode"""
        if not self.active_mode:
            return True  # Default to allowing all events if no mode set
            
        return event_type in self.allowed_events[self.active_mode.value]
    
    def get_status(self) -> Dict:
        """Get current registry status"""
        return {
            "active_mode": self.active_mode.value if self.active_mode else None,
            "active_services": list(self.active_services),
            "available_modes": [mode.value for mode in ServiceMode]
        }

# Global service registry instance
service_registry = ServiceRegistry() 