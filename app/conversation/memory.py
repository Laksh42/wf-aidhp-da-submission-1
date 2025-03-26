from typing import List, Dict, Any, Optional
import json
from datetime import datetime, timedelta
from motor.motor_asyncio import AsyncIOMotorDatabase
from app.config import settings

class ConversationMemory:
    def __init__(self, db: AsyncIOMotorDatabase):
        """Initialize with database connection."""
        self.db = db
        
    async def store_interaction(
        self,
        user_id: str,
        message: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store a conversation interaction in MongoDB."""
        interaction = {
            "user_id": user_id,
            "message": message,
            "response": response,
            "timestamp": datetime.utcnow(),
            "metadata": metadata or {}
        }
        
        # Store in MongoDB for long-term storage
        await self.db.conversations.insert_one(interaction)
        
    async def get_recent_interactions(
        self,
        user_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve recent interactions from MongoDB."""
        interactions = await self.db.conversations.find(
            {"user_id": user_id}
        ).sort("timestamp", -1).limit(limit).to_list(length=limit)
        
        return interactions
    
    async def get_user_context(
        self,
        user_id: str,
        time_window: timedelta = timedelta(days=30)
    ) -> Dict[str, Any]:
        """Get user context including preferences, sentiment trends, and financial behavior."""
        cutoff_date = datetime.utcnow() - time_window
        
        # Get recent interactions
        interactions = await self.db.conversations.find({
            "user_id": user_id,
            "timestamp": {"$gte": cutoff_date}
        }).sort("timestamp", -1).limit(50).to_list(length=None)
        
        # Get user preferences
        preferences = await self.db.user_preferences.find_one({"user_id": user_id})
        
        return {
            "interactions": interactions or [],
            "preferences": preferences or {},
        }
    
    async def update_user_preferences(
        self,
        user_id: str,
        preferences: Dict[str, Any]
    ) -> None:
        """Update user preferences based on interactions and feedback."""
        await self.db.user_preferences.update_one(
            {"user_id": user_id},
            {"$set": preferences},
            upsert=True
        )
    
    async def store_feedback(
        self,
        user_id: str,
        interaction_id: str,
        rating: int,
        feedback_text: Optional[str] = None
    ) -> None:
        """Store user feedback for RLHF."""
        feedback = {
            "user_id": user_id,
            "interaction_id": interaction_id,
            "rating": rating,
            "feedback_text": feedback_text,
            "timestamp": datetime.utcnow()
        }
        await self.db.feedback.insert_one(feedback)
    
    async def get_feedback_stats(
        self,
        user_id: str,
        time_window: timedelta = timedelta(days=30)
    ) -> Dict[str, Any]:
        """Get feedback statistics for model improvement."""
        cutoff_date = datetime.utcnow() - time_window
        
        feedback = await self.db.feedback.find({
            "user_id": user_id,
            "timestamp": {"$gte": cutoff_date}
        }).to_list(length=None)
        
        return {
            "total_feedback": len(feedback),
            "average_rating": sum(f["rating"] for f in feedback) / len(feedback) if feedback else 0,
            "feedback_texts": [f["feedback_text"] for f in feedback if f.get("feedback_text")]
        } 