import logging
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import ConnectionFailure
from app.config import settings
import asyncio
from typing import Dict, List, Any, Optional
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global database connection
db_client: AsyncIOMotorClient = None
db: AsyncIOMotorDatabase = None
mock_db: Dict[str, List[Dict[str, Any]]] = None

class MockCollection:
    def __init__(self, name: str, data: List[Dict[str, Any]] = None):
        self.name = name
        self.data = data or []
    
    async def find_one(self, query: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        if not query:
            return self.data[0] if self.data else None
        
        for item in self.data:
            match = True
            for key, value in query.items():
                if key not in item or item[key] != value:
                    match = False
                    break
            if match:
                return item
        return None
    
    async def find(self, query: Dict[str, Any] = None):
        results = []
        if not query:
            return self.data
        
        for item in self.data:
            match = True
            for key, value in query.items():
                if key not in item or item[key] != value:
                    match = False
                    break
            if match:
                results.append(item)
        
        class MockCursor:
            def __init__(self, items):
                self.items = items
            
            def sort(self, *args, **kwargs):
                # Simple sorting could be implemented here
                return self
            
            async def to_list(self, length=None):
                return self.items[:length] if length else self.items
        
        return MockCursor(results)
    
    async def insert_one(self, document: Dict[str, Any]):
        self.data.append(document)
        class MockInsertResult:
            def __init__(self, inserted_id):
                self.inserted_id = inserted_id
        return MockInsertResult(document.get('_id', 'mock_id'))
    
    async def update_one(self, query: Dict[str, Any], update: Dict[str, Any], upsert: bool = False):
        item = await self.find_one(query)
        if item:
            # Handle $set operator
            if '$set' in update:
                for key, value in update['$set'].items():
                    item[key] = value
            # Handle direct updates
            else:
                for key, value in update.items():
                    item[key] = value
            class MockUpdateResult:
                def __init__(self, matched_count, modified_count, upserted_id=None):
                    self.matched_count = matched_count
                    self.modified_count = modified_count
                    self.upserted_id = upserted_id
            return MockUpdateResult(1, 1)
        elif upsert:
            # Upsert: create if not exists
            new_doc = {**query}
            if '$set' in update:
                new_doc.update(update['$set'])
            else:
                new_doc.update(update)
            await self.insert_one(new_doc)
            return MockUpdateResult(0, 0, 'mock_id')
        else:
            return MockUpdateResult(0, 0)

class MockDatabase:
    def __init__(self):
        self.collections = {
            "users": MockCollection("users", [
                {"_id": "1", "user_id": "testuser", "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW", "full_name": "Test User", "email": "test@example.com"}
            ]),
            "account_data": MockCollection("account_data"),
            "credit_history": MockCollection("credit_history"),
            "demographic_data": MockCollection("demographic_data"),
            "investment_data": MockCollection("investment_data"),
            "transaction_data": MockCollection("transaction_data"),
            "products": MockCollection("products"),
        }
    
    def __getitem__(self, name):
        if name not in self.collections:
            self.collections[name] = MockCollection(name)
        return self.collections[name]
    
    async def list_collection_names(self):
        return list(self.collections.keys())
    
    async def command(self, command):
        # Mock ping command
        if command == "ping":
            return {"ok": 1}
        return {"ok": 0}

async def connect_to_mongo() -> AsyncIOMotorDatabase:
    """
    Create a MongoDB connection pool and connect to the database.
    Returns the database instance.
    """
    global db_client, db, mock_db
    
    # If mock data is enabled, use the mock database
    if settings.ENABLE_MOCK_DATA:
        logger.info("Mock data enabled: Using in-memory database")
        mock_db = MockDatabase()
        return mock_db
    
    try:
        # Create client using the MongoDB URL from settings
        mongodb_url = settings.MONGODB_URL
        logger.info(f"Connecting to MongoDB at {mongodb_url}")
        
        # Set a shorter server selection timeout for faster startup
        db_client = AsyncIOMotorClient(mongodb_url, serverSelectionTimeoutMS=5000)
        
        # Get database
        db = db_client[settings.MONGODB_DB]
        
        # Test connection
        await db.command("ping")
        logger.info("Connected to MongoDB")
        
        # List collections to verify database is fully accessible
        collections = await db.list_collection_names()
        logger.info(f"Available collections: {', '.join(collections) if collections else 'None'}")
        
        if not collections:
            logger.warning("MongoDB database is empty - no collections found. Data may need to be imported.")
        
        # Check specific collections needed for financial data
        required_collections = ["account_data", "credit_history", "demographic_data", "investment_data", "transaction_data", "products"]
        missing_collections = [coll for coll in required_collections if coll not in collections]
        
        if missing_collections:
            logger.warning(f"Missing required collections: {', '.join(missing_collections)}")
            logger.warning("Some financial data may not be available")
        
        return db
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {str(e)}")
        if settings.ENABLE_MOCK_DATA:
            logger.info("Using mock database instead")
            mock_db = MockDatabase()
            return mock_db
        # Return None instead of raising to allow app to start without DB
        db = None
        return None

async def close_mongo_connection():
    """Close MongoDB connection."""
    global db_client
    if db_client:
        db_client.close()
        logger.info("MongoDB connection closed")

async def get_database():
    """
    Get the database instance. Used as a dependency.
    """
    global db, mock_db
    if settings.ENABLE_MOCK_DATA and mock_db is not None:
        return mock_db
    if db is None:
        db = await connect_to_mongo()
    return db 