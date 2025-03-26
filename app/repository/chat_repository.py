from typing import List, Optional, Dict, Any
from bson import ObjectId
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.models.chat import ChatMessage, ChatMessageCreate, Conversation, ConversationCreate, ConversationUpdate, ConversationSummary


class ChatRepository:
    """Repository for chat-related database operations."""
    
    def __init__(self, database: AsyncIOMotorDatabase):
        """Initialize with database connection."""
        self.db = database
        self.messages_collection = database.chat_messages
        self.conversations_collection = database.conversations
    
    async def create_indexes(self):
        """Create necessary indexes."""
        await self.messages_collection.create_index("conversation_id")
        await self.messages_collection.create_index("created_at")
        await self.conversations_collection.create_index("user_id")
        await self.conversations_collection.create_index("created_at")
    
    # Conversation methods
    
    async def create_conversation(self, data: ConversationCreate, user_id: str) -> Conversation:
        """Create a new conversation."""
        now = datetime.utcnow()
        conversation = Conversation(
            _id=ObjectId(),
            user_id=user_id,
            title=data.title,
            created_at=now,
            updated_at=now,
            is_active=True,
            metadata=data.metadata or {}
        )
        
        await self.conversations_collection.insert_one(conversation.dict(by_alias=True))
        return conversation
    
    async def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID."""
        if not ObjectId.is_valid(conversation_id):
            return None
            
        result = await self.conversations_collection.find_one({"_id": ObjectId(conversation_id)})
        if result:
            return Conversation(**result)
        return None
    
    async def update_conversation(self, conversation_id: str, data: ConversationUpdate) -> Optional[Conversation]:
        """Update a conversation."""
        if not ObjectId.is_valid(conversation_id):
            return None
            
        conversation = await self.get_conversation(conversation_id)
        if not conversation:
            return None
            
        update_data = data.dict(exclude_unset=True)
        if update_data:
            update_data["updated_at"] = datetime.utcnow()
            await self.conversations_collection.update_one(
                {"_id": ObjectId(conversation_id)},
                {"$set": update_data}
            )
            
        return await self.get_conversation(conversation_id)
    
    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and all its messages."""
        if not ObjectId.is_valid(conversation_id):
            return False
            
        # Delete the conversation
        result = await self.conversations_collection.delete_one({"_id": ObjectId(conversation_id)})
        
        # Delete all messages in this conversation
        await self.messages_collection.delete_many({"conversation_id": conversation_id})
        
        return result.deleted_count > 0
    
    async def list_user_conversations(self, user_id: str, skip: int = 0, limit: int = 20) -> List[ConversationSummary]:
        """List conversations for a user with pagination."""
        cursor = self.conversations_collection.find({"user_id": user_id}).sort("updated_at", -1).skip(skip).limit(limit)
        conversations = await cursor.to_list(length=limit)
        
        result = []
        for conv in conversations:
            # Get message count for each conversation
            count = await self.messages_collection.count_documents({"conversation_id": str(conv["_id"])})
            conv_obj = Conversation(**conv)
            result.append(ConversationSummary(
                id=conv_obj.id,
                title=conv_obj.title,
                created_at=conv_obj.created_at,
                updated_at=conv_obj.updated_at,
                message_count=count
            ))
            
        return result
    
    # Message methods
    
    async def create_message(self, data: ChatMessageCreate) -> ChatMessage:
        """Create a new chat message."""
        now = datetime.utcnow()
        message = ChatMessage(
            _id=ObjectId(),
            conversation_id=data.conversation_id,
            role=data.role,
            content=data.content,
            created_at=now,
            metadata=data.metadata or {}
        )
        
        # Update conversation's updated_at timestamp
        if ObjectId.is_valid(data.conversation_id):
            await self.conversations_collection.update_one(
                {"_id": ObjectId(data.conversation_id)},
                {"$set": {"updated_at": now}}
            )
        
        await self.messages_collection.insert_one(message.dict(by_alias=True))
        return message
    
    async def get_message(self, message_id: str) -> Optional[ChatMessage]:
        """Get a message by ID."""
        if not ObjectId.is_valid(message_id):
            return None
            
        result = await self.messages_collection.find_one({"_id": ObjectId(message_id)})
        if result:
            return ChatMessage(**result)
        return None
    
    async def get_conversation_messages(self, conversation_id: str, skip: int = 0, limit: int = 50) -> List[ChatMessage]:
        """Get messages for a conversation with pagination."""
        cursor = self.messages_collection.find({"conversation_id": conversation_id}).sort("created_at", 1).skip(skip).limit(limit)
        messages = await cursor.to_list(length=limit)
        return [ChatMessage(**msg) for msg in messages]
    
    async def count_conversation_messages(self, conversation_id: str) -> int:
        """Count messages in a conversation."""
        return await self.messages_collection.count_documents({"conversation_id": conversation_id})
    
    async def delete_message(self, message_id: str) -> bool:
        """Delete a message."""
        if not ObjectId.is_valid(message_id):
            return False
            
        result = await self.messages_collection.delete_one({"_id": ObjectId(message_id)})
        return result.deleted_count > 0
    
    async def get_conversation_context(self, conversation_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the most recent messages from a conversation as context for AI."""
        cursor = self.messages_collection.find({"conversation_id": conversation_id}).sort("created_at", -1).limit(limit)
        messages = await cursor.to_list(length=limit)
        
        # Reverse to get chronological order
        messages.reverse()
        
        # Convert to the format expected by AI
        context = []
        for msg in messages:
            context.append({
                "role": msg["role"].lower(),
                "content": msg["content"]
            })
            
        return context 