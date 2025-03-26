from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from bson import ObjectId
from enum import Enum

from app.models.user import PyObjectId

class MessageRole(str, Enum):
    """Message role enum."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class ChatMessage(BaseModel):
    """Chat message model."""
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    conversation_id: str
    role: MessageRole
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {
            ObjectId: str
        }

class ChatMessageCreate(BaseModel):
    """Chat message creation model."""
    conversation_id: str
    role: MessageRole
    content: str
    metadata: Optional[Dict[str, Any]] = None

class Conversation(BaseModel):
    """Conversation model."""
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    user_id: str
    title: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {
            ObjectId: str
        }

class ConversationCreate(BaseModel):
    """Conversation creation model."""
    user_id: str
    title: str = "New Conversation" 
    initial_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class ConversationUpdate(BaseModel):
    """Conversation update model."""
    title: Optional[str] = None
    is_active: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None

class ConversationSummary(BaseModel):
    """Conversation summary model."""
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0
    
    class Config:
        json_encoders = {
            ObjectId: str
        } 