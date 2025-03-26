from motor.motor_asyncio import AsyncIOMotorClient
from app.config import settings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB client
mongo_client = None
db = None

async def connect_to_mongo():
    """Connect to MongoDB and initialize the database client."""
    global mongo_client, db
    
    try:
        mongo_client = AsyncIOMotorClient(settings.MONGODB_URI)
        db = mongo_client[settings.DATABASE_NAME]
        
        # Print MongoDB connection info for verification
        server_info = await mongo_client.server_info()
        logger.info(f"Connected to MongoDB version {server_info.get('version')}")
        
        # Create collections if they don't exist
        if "users" not in await db.list_collection_names():
            await db.create_collection("users")
            await db["users"].create_index("username", unique=True)
        
        if "meta_prompts" not in await db.list_collection_names():
            await db.create_collection("meta_prompts")
            await db["meta_prompts"].create_index("user_id")
        
        if "conversations" not in await db.list_collection_names():
            await db.create_collection("conversations")
            await db["conversations"].create_index("user_id")
        
        if "recommendations" not in await db.list_collection_names():
            await db.create_collection("recommendations")
            await db["recommendations"].create_index("user_id")
        
        logger.info("MongoDB collections initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        # If we're in development mode, use a mock DB or in-memory solution
        if settings.DEBUG:
            logger.warning("Using in-memory database for development")
        else:
            raise

async def close_mongo_connection():
    """Close the MongoDB connection."""
    global mongo_client
    if mongo_client:
        mongo_client.close()
        logger.info("MongoDB connection closed")

# Database access functions
async def get_database():
    """Get the database instance for dependency injection."""
    return db 