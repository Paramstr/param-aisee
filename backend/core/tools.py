import asyncio
import logging
import re
import tempfile
import os
from typing import Dict, Callable, Any, Optional
from abc import ABC, abstractmethod

from ..events import Event, EventType, event_bus

logger = logging.getLogger(__name__)


class Tool(ABC):
    """Base class for all tools"""
    
    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool with given parameters"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name for identification"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for LLM prompt"""
        pass


class PhotoTool(Tool):
    """Tool for capturing single photos"""
    
    def __init__(self, vision_processor):
        self.vision_processor = vision_processor
    
    @property
    def name(self) -> str:
        return "get_photo"
    
    @property
    def description(self) -> str:
        return "<get_photo/>                     → capture one still image"
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Capture a single photo"""
        try:
            await event_bus.publish(Event(
                type=EventType.TOOL_EVENT,
                action="photo_start",
                data={"tool": self.name}
            ))
            
            # Capture photo using vision processor
            photo_base64 = self.vision_processor.capture_frame_for_llm()
            
            if photo_base64 is None:
                raise Exception("Failed to capture photo - no camera frame available")
            
            await event_bus.publish(Event(
                type=EventType.TOOL_EVENT,
                action="photo_complete",
                data={
                    "tool": self.name,
                    "photo_base64": photo_base64,
                    "success": True
                }
            ))
            
            return {
                "success": True,
                "photo_base64": photo_base64,
                "message": "Photo captured successfully"
            }
            
        except Exception as e:
            logger.error(f"Photo capture failed: {e}")
            await event_bus.publish(Event(
                type=EventType.TOOL_EVENT,
                action="photo_failed",
                data={
                    "tool": self.name,
                    "error": str(e),
                    "success": False
                }
            ))
            
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to capture photo"
            }


class VideoTool(Tool):
    """Tool for capturing video clips"""
    
    def __init__(self, video_recorder):
        self.video_recorder = video_recorder
    
    @property
    def name(self) -> str:
        return "get_video"
    
    @property
    def description(self) -> str:
        return "<get_video duration=\"N\"/>        → capture N-second video"
    
    async def execute(self, duration: int = 3, **kwargs) -> Dict[str, Any]:
        """Capture a video clip"""
        try:
            # Validate duration
            duration = max(1, min(300, int(duration)))  # Clamp between 1-300 seconds (5 minutes)
            
            await event_bus.publish(Event(
                type=EventType.TOOL_EVENT,
                action="video_start",
                data={
                    "tool": self.name,
                    "duration": duration
                }
            ))
            
            # Record video
            video_result = await self.video_recorder.record_video(duration)
            
            if not video_result["success"]:
                raise Exception(video_result.get("error", "Unknown video recording error"))
            
            await event_bus.publish(Event(
                type=EventType.TOOL_EVENT,
                action="video_complete",
                data={
                    "tool": self.name,
                    "duration": duration,
                    "video_base64": video_result.get("video_base64"),
                    "frames_recorded": video_result.get("frames_recorded"),
                    "file_size": video_result.get("file_size"),
                    "success": True
                }
            ))
            
            return {
                "success": True,
                "video_base64": video_result.get("video_base64"),
                "file_path": video_result.get("file_path"),
                "duration": duration,
                "frames_recorded": video_result.get("frames_recorded"),
                "message": f"Video recorded successfully ({duration}s, {video_result.get('frames_recorded', 0)} frames)"
            }
            
        except Exception as e:
            logger.error(f"Video capture failed: {e}")
            await event_bus.publish(Event(
                type=EventType.TOOL_EVENT,
                action="video_failed",
                data={
                    "tool": self.name,
                    "duration": duration,
                    "error": str(e),
                    "success": False
                }
            ))
            
            return {
                "success": False,
                "error": str(e),
                "duration": duration,
                "message": f"Failed to record video: {e}"
            }


class ToolRegistry:
    """Registry for managing available tools"""
    
    def __init__(self, vision_processor=None, video_recorder=None):
        self.tools: Dict[str, Tool] = {}
        self.vision_processor = vision_processor
        self.video_recorder = video_recorder
        self._initialize_tools()
    
    def _initialize_tools(self):
        """Initialize available tools"""
        if self.vision_processor:
            self.tools["get_photo"] = PhotoTool(self.vision_processor)
        
        if self.video_recorder:
            self.tools["get_video"] = VideoTool(self.video_recorder)
        
        logger.info(f"Initialized {len(self.tools)} tools: {list(self.tools.keys())}")
    
    def get_tool_definitions(self) -> str:
        """Get tool definitions for LLM prompt"""
        if not self.tools:
            return ""
        
        definitions = "════════ 3. TOOLS ═════════════════\n"
        for tool in self.tools.values():
            definitions += f"{tool.description}\n"
        
        # Add usage instructions
        definitions += " • When <get_video> runs, UI auto-counts \"3-2-1-Begin\", shows timer, then says \"Finished recording.\"\n"
        
        return definitions
    
    async def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool by name"""
        if tool_name not in self.tools:
            error_msg = f"Unknown tool: {tool_name}"
            logger.error(error_msg)
            await event_bus.publish(Event(
                type=EventType.TOOL_EVENT,
                action="tool_error",
                data={
                    "tool": tool_name,
                    "error": error_msg,
                    "success": False
                }
            ))
            return {"success": False, "error": error_msg}
        
        tool = self.tools[tool_name]
        logger.info(f"Executing tool: {tool_name} with args: {kwargs}")
        
        try:
            await event_bus.publish(Event(
                type=EventType.TOOL_EVENT,
                action="tool_start",
                data={
                    "tool": tool_name,
                    "args": kwargs
                }
            ))
            
            result = await tool.execute(**kwargs)
            
            await event_bus.publish(Event(
                type=EventType.TOOL_EVENT,
                action="tool_complete",
                data={
                    "tool": tool_name,
                    "result": result,
                    "success": result.get("success", False)
                }
            ))
            
            return result
            
        except Exception as e:
            error_msg = f"Tool execution failed: {e}"
            logger.error(error_msg)
            await event_bus.publish(Event(
                type=EventType.TOOL_EVENT,
                action="tool_error",
                data={
                    "tool": tool_name,
                    "error": error_msg,
                    "success": False
                }
            ))
            return {"success": False, "error": error_msg}
    
    def parse_tool_calls(self, text: str) -> list:
        """Parse tool calls from LLM response text"""
        tool_calls = []
        
        # Pattern for <get_photo/>
        photo_pattern = r'<get_photo\s*/>'
        photo_matches = re.findall(photo_pattern, text, re.IGNORECASE)
        for match in photo_matches:
            tool_calls.append({"tool": "get_photo", "args": {}})
        
        # Pattern for <get_video duration="N"/>
        video_pattern = r'<get_video\s+duration="(\d+)"\s*/>'
        video_matches = re.findall(video_pattern, text, re.IGNORECASE)
        for duration_str in video_matches:
            duration = int(duration_str)
            tool_calls.append({"tool": "get_video", "args": {"duration": duration}})
        
        # Pattern for <get_video/> (default duration)
        video_default_pattern = r'<get_video\s*/>'
        video_default_matches = re.findall(video_default_pattern, text, re.IGNORECASE)
        for match in video_default_matches:
            tool_calls.append({"tool": "get_video", "args": {"duration": 3}})
        
        return tool_calls
    
    def clean_response_text(self, text: str) -> str:
        """Remove tool call XML tags from response text"""
        # Remove all tool call patterns
        text = re.sub(r'<get_photo\s*/>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<get_video\s+duration="[^"]*"\s*/>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<get_video\s*/>', '', text, flags=re.IGNORECASE)
        
        # Clean up extra whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Multiple newlines to double
        text = text.strip()
        
        return text 