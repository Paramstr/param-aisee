import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class ConversationMessage:
    id: str
    type: str  # 'user' | 'assistant' | 'system'
    content: str
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None

class ConversationStorage:
    def __init__(self, storage_path: str = "conversation_history.json"):
        self.storage_path = Path(storage_path)
        self.messages: List[ConversationMessage] = []
        self.load_conversation()
    
    def add_message(self, message_type: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> ConversationMessage:
        """Add a new message to the conversation"""
        message = ConversationMessage(
            id=f"{time.time():.6f}",
            type=message_type,
            content=content.strip(),
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        self.messages.append(message)
        self.save_conversation()
        
        logger.info(f"Added {message_type} message: {content[:50]}...")
        return message
    
    def get_recent_messages(self, limit: int = 10) -> List[ConversationMessage]:
        """Get recent messages for context"""
        return self.messages[-limit:] if limit > 0 else self.messages
    
    def get_conversation_context(self, limit: int = 5) -> str:
        """Get formatted conversation context for LLM - optimized string operations"""
        recent_messages = self.get_recent_messages(limit)
        
        if not recent_messages:
            return ""
        
        # Use list comprehension and join for better performance
        context_parts = []
        for msg in recent_messages:
            if msg.type == 'user':
                context_parts.append(f"User: {msg.content}")
            elif msg.type == 'assistant':
                context_parts.append(f"Assistant: {msg.content}")
        
        # Single join operation instead of multiple concatenations
        return "\n".join(context_parts)
    
    def clear_conversation(self):
        """Clear all conversation history"""
        self.messages = []
        self.save_conversation()
        logger.info("Conversation history cleared")
    
    def save_conversation(self):
        """Save conversation to disk"""
        try:
            data = {
                "messages": [asdict(msg) for msg in self.messages],
                "last_updated": time.time()
            }
            
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Failed to save conversation: {e}")
    
    def load_conversation(self):
        """Load conversation from disk"""
        try:
            if self.storage_path.exists():
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.messages = [
                    ConversationMessage(**msg_data) 
                    for msg_data in data.get("messages", [])
                ]
                
                logger.info(f"Loaded {len(self.messages)} conversation messages")
            else:
                logger.info("No existing conversation file found, starting fresh")
                
        except Exception as e:
            logger.error(f"Failed to load conversation: {e}")
            self.messages = []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get conversation statistics"""
        user_messages = sum(1 for msg in self.messages if msg.type == 'user')
        assistant_messages = sum(1 for msg in self.messages if msg.type == 'assistant')
        
        return {
            "total_messages": len(self.messages),
            "user_messages": user_messages,
            "assistant_messages": assistant_messages,
            "first_message": self.messages[0].timestamp if self.messages else None,
            "last_message": self.messages[-1].timestamp if self.messages else None
        }

# Global conversation storage instance
conversation_storage = ConversationStorage() 