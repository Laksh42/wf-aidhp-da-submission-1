from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging
from passlib.context import CryptContext
from pymongo.errors import DuplicateKeyError

from app.models.user import UserCreate, UserInDB, User, UserUpdate
from app.database.mongodb import get_database

# Setup logging
logger = logging.getLogger(__name__)

# Password context for hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class UserRepository:
    """Repository for user-related database operations."""
    
    def __init__(self, database: AsyncIOMotorDatabase):
        """Initialize with database connection."""
        self.db = database
        self.collection = database.users
    
    async def create_indexes(self):
        """Create necessary indexes."""
        await self.collection.create_index("email", unique=True, sparse=True)
        await self.collection.create_index("user_id", unique=True)
    
    async def get_by_id(self, user_id: str) -> Optional[UserInDB]:
        """Get a user by MongoDB ObjectID."""
        if not ObjectId.is_valid(user_id):
            return None
            
        user = await self.collection.find_one({"_id": ObjectId(user_id)})
        if user:
            return UserInDB(**user)
        return None
    
    async def get_by_user_id(self, user_id: str) -> Optional[UserInDB]:
        """Get a user by CSV user_id."""
        user = await self.collection.find_one({"user_id": user_id})
        if user:
            return UserInDB(**user)
        return None
    
    async def get_by_email(self, email: str) -> Optional[UserInDB]:
        """Get a user by email."""
        if not email:
            return None
            
        user = await self.collection.find_one({"email": email})
        if user:
            return UserInDB(**user)
        return None
    
    async def create(self, user_data: UserCreate) -> UserInDB:
        """Create a new user."""
        # Check if user already exists
        existing_user = await self.get_by_user_id(user_data.user_id)
        if existing_user:
            raise ValueError(f"User with user_id {user_data.user_id} already exists")
            
        # Hash password
        hashed_password = self._hash_password(user_data.password)
        
        # Convert to dict for storage
        user_dict = user_data.dict(exclude={"password"})
        user_dict["hashed_password"] = hashed_password
        user_dict["created_at"] = datetime.utcnow()
        user_dict["is_active"] = True
        
        try:
            result = await self.collection.insert_one(user_dict)
            created_user = await self.get_by_id(str(result.inserted_id))
            return created_user
        except DuplicateKeyError as e:
            logger.error(f"Duplicate key error: {e}")
            if "email" in str(e) and user_data.email:
                raise ValueError(f"User with email {user_data.email} already exists")
            else:
                raise ValueError(f"User with user_id {user_data.user_id} already exists")
    
    async def update(self, user_id: str, user_data: UserUpdate) -> Optional[UserInDB]:
        """Update a user."""
        if not ObjectId.is_valid(user_id):
            return None
            
        user = await self.get_by_id(user_id)
        if not user:
            return None
            
        update_data = user_data.dict(exclude_unset=True)
        
        if "password" in update_data:
            update_data["hashed_password"] = self._hash_password(update_data.pop("password"))
        
        if update_data:
            await self.collection.update_one(
                {"_id": ObjectId(user_id)},
                {"$set": update_data}
            )
            
        return await self.get_by_id(user_id)
    
    async def delete(self, user_id: str) -> bool:
        """Delete a user."""
        if not ObjectId.is_valid(user_id):
            return False
            
        result = await self.collection.delete_one({"_id": ObjectId(user_id)})
        return result.deleted_count > 0
    
    async def list(self, skip: int = 0, limit: int = 100) -> List[UserInDB]:
        """List users with pagination."""
        cursor = self.collection.find().skip(skip).limit(limit)
        users = await cursor.to_list(length=limit)
        return [UserInDB(**user) for user in users]
    
    async def count(self) -> int:
        """Count total users."""
        return await self.collection.count_documents({})
    
    async def update_last_login(self, user_id: str) -> None:
        """Update user's last login timestamp."""
        if ObjectId.is_valid(user_id):
            # Update by MongoDB ID
            await self.collection.update_one(
                {"_id": ObjectId(user_id)},
                {"$set": {"last_login": datetime.utcnow()}}
            )
        else:
            # Update by user_id (CSV ID)
            await self.collection.update_one(
                {"user_id": user_id},
                {"$set": {"last_login": datetime.utcnow()}}
            )

    def _hash_password(self, password: str) -> str:
        """Hash a password for storing."""
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a stored password against a provided password."""
        return pwd_context.verify(plain_password, hashed_password)
    
    async def get_user_data(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user data from various collections."""
        # Basic user info
        user = await self.get_by_user_id(user_id)
        if not user:
            return {}
            
        user_data = user.dict()
        
        # Demographic data
        demographics = await self.db.demographic_data.find_one({"user_id": user_id})
        if demographics:
            # Remove _id field from the result
            demographics.pop("_id", None)
            user_data["demographics"] = demographics
            
        # Account data
        account = await self.db.account_data.find_one({"user_id": user_id})
        if account:
            account.pop("_id", None)
            user_data["account"] = account
            
        # Credit history
        credit = await self.db.credit_history.find_one({"user_id": user_id})
        if credit:
            credit.pop("_id", None)
            user_data["credit"] = credit
            
        # Investments
        investments_cursor = self.db.investment_data.find({"user_id": user_id})
        investments = await investments_cursor.to_list(length=100)
        if investments:
            for inv in investments:
                inv.pop("_id", None)
            user_data["investments"] = investments
            
        # Transactions
        transactions_cursor = self.db.transaction_data.find({"user_id": user_id}).sort("date", -1).limit(20)
        transactions = await transactions_cursor.to_list(length=20)
        if transactions:
            for tx in transactions:
                tx.pop("_id", None)
            user_data["recent_transactions"] = transactions
            
        return user_data
    
    async def list_all(self, skip: int = 0, limit: int = 100) -> List[User]:
        """List all users with pagination."""
        cursor = self.collection.find().skip(skip).limit(limit)
        users = []
        
        async for user_dict in cursor:
            user = UserInDB(**user_dict)
            users.append(User(
                id=user.id,
                user_id=user.user_id,
                email=user.email,
                full_name=user.full_name,
                is_active=user.is_active,
                created_at=user.created_at,
                last_login=user.last_login
            ))
        
        return users 