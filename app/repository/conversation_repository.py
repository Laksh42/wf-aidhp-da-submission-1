from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase
from datetime import datetime
from typing import List, Optional, Dict
import logging

from app.models.conversation import Conversation, Message, MessageRole
from app.models.chat import ConversationCreate, ConversationUpdate
from app.database.mongodb import get_database

# Setup logging
logger = logging.getLogger(__name__)

class ConversationRepository:
    """Repository for conversation operations."""
    
    def __init__(self, db: AsyncIOMotorDatabase = None):
        """Initialize with database connection."""
        self.db = db
        self.collection_name = "conversations"
    
    async def setup(self):
        """Setup repository with database connection if not initialized."""
        if not self.db:
            self.db = await get_database()
        
        # Ensure indexes
        await self.db[self.collection_name].create_index("user_id")
        await self.db[self.collection_name].create_index("created_at")
    
    async def create(self, conv_create: ConversationCreate) -> Conversation:
        """Create a new conversation."""
        await self.setup()
        
        # Generate new ID
        conv_id = str(ObjectId())
        
        # Create initial message if provided
        messages = []
        if conv_create.initial_message:
            messages.append(Message(
                role=MessageRole.USER,
                content=conv_create.initial_message,
                timestamp=datetime.utcnow()
            ))
        
        # Create conversation object
        conversation = Conversation(
            _id=conv_id,
            user_id=conv_create.user_id,
            title=conv_create.title,
            messages=messages,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            is_active=True
        )
        
        # Insert into database
        await self.db[self.collection_name].insert_one(conversation.dict(by_alias=True))
        
        return conversation
    
    async def get(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID."""
        await self.setup()
        
        conv_dict = await self.db[self.collection_name].find_one({"_id": conversation_id})
        if conv_dict:
            return Conversation(**conv_dict)
        return None
    
    async def list_by_user(self, user_id: str, skip: int = 0, limit: int = 20) -> List[Conversation]:
        """List conversations for a user with pagination."""
        await self.setup()
        
        cursor = self.db[self.collection_name].find(
            {"user_id": user_id}
        ).sort("updated_at", -1).skip(skip).limit(limit)
        
        conversations = []
        async for conv_dict in cursor:
            conversations.append(Conversation(**conv_dict))
        
        return conversations
    
    async def add_message(self, conversation_id: str, message: Message) -> Optional[Conversation]:
        """Add a message to a conversation."""
        await self.setup()
        
        # Get the conversation
        conversation = await self.get(conversation_id)
        if not conversation:
            return None
        
        # Add the message
        result = await self.db[self.collection_name].update_one(
            {"_id": conversation_id},
            {
                "$push": {"messages": message.dict()},
                "$set": {"updated_at": datetime.utcnow()}
            }
        )
        
        if result.modified_count > 0:
            return await self.get(conversation_id)
        return None
    
    async def update(self, conversation_id: str, update_data: ConversationUpdate) -> Optional[Conversation]:
        """Update a conversation."""
        await self.setup()
        
        # Get the conversation
        conversation = await self.get(conversation_id)
        if not conversation:
            return None
        
        # Prepare update data
        update_dict = update_data.dict(exclude_unset=True)
        if not update_dict:
            return conversation
        
        update_dict["updated_at"] = datetime.utcnow()
        
        # Update the conversation
        result = await self.db[self.collection_name].update_one(
            {"_id": conversation_id},
            {"$set": update_dict}
        )
        
        if result.modified_count > 0:
            return await self.get(conversation_id)
        return conversation
    
    async def set_meta_prompt(self, conversation_id: str, meta_prompt: str) -> Optional[Conversation]:
        """Set the meta prompt for a conversation."""
        await self.setup()
        
        result = await self.db[self.collection_name].update_one(
            {"_id": conversation_id},
            {
                "$set": {
                    "meta_prompt": meta_prompt,
                    "updated_at": datetime.utcnow()
                }
            }
        )
        
        if result.modified_count > 0:
            return await self.get(conversation_id)
        return None
    
    async def delete(self, conversation_id: str) -> bool:
        """Delete a conversation."""
        await self.setup()
        
        result = await self.db[self.collection_name].delete_one({"_id": conversation_id})
        return result.deleted_count > 0
    
    async def count_by_user(self, user_id: str) -> int:
        """Count conversations for a user."""
        await self.setup()
        
        count = await self.db[self.collection_name].count_documents({"user_id": user_id})
        return count 