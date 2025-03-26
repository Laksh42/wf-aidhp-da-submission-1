# Database package

import logging
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import ConnectionFailure
from typing import Optional

from app.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global MongoDB client
client: Optional[AsyncIOMotorClient] = None
database: Optional[AsyncIOMotorDatabase] = None

async def connect_to_mongo():
    """Connect to MongoDB."""
    global client, database
    try:
        # Use the connection string from settings instead of just the URL
        connection_string = settings.MONGODB_CONNECTION_STRING
        logger.info(f"Connecting to MongoDB using connection string (without credentials): {connection_string.split('@')[-1]}")
        
        client = AsyncIOMotorClient(connection_string)
        # Use the correct database name from settings
        database = client[settings.MONGODB_DB]
        
        # Verify connection
        await database.command("ping")
        logger.info(f"Connected to MongoDB database: {settings.MONGODB_DB}")
        
        return database
        
    except ConnectionFailure as e:
        logger.error(f"Failed to connect to MongoDB: {str(e)}")
        # Don't raise the exception, to allow app to function without DB
        logger.warning("Application will run with limited functionality (no personalized financial data)")
        database = None
        return None
    
    return database

async def close_mongo_connection():
    """Close MongoDB connection."""
    global client
    if client:
        client.close()
        logger.info("Closed MongoDB connection")

def get_database() -> Optional[AsyncIOMotorDatabase]:
    """Get the database instance."""
    if database is None:
        logger.warning("Database not initialized. Financial data will not be available.")
    return database

__all__ = ["connect_to_mongo", "close_mongo_connection", "get_database"] 