import pandas as pd
import os
from pathlib import Path
import logging
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo.errors import BulkWriteError
import json

from app.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Utility for loading data from CSV files into MongoDB collections.
    Used for initializing the database with sample data.
    """
    
    def __init__(self, db: AsyncIOMotorDatabase):
        """Initialize with database connection."""
        self.db = db
        self.data_dir = Path(settings.DATA_DIR)
    
    async def load_all_datasets(self):
        """Load all datasets into MongoDB collections."""
        logger.info("Starting to load all datasets into MongoDB...")
        
        # Ensure data directory exists
        if not os.path.exists(self.data_dir):
            logger.warning(f"Data directory {self.data_dir} not found. Creating it.")
            os.makedirs(self.data_dir, exist_ok=True)
            return False
        
        # Load each dataset
        await self.load_dataset("demographic_data.csv", "demographics")
        await self.load_dataset("account_data.csv", "accounts")
        await self.load_dataset("credit_history.csv", "credit_history")
        await self.load_dataset("investment_data.csv", "investments")
        await self.load_dataset("transaction_data.csv", "transactions")
        await self.load_dataset("social_media_sentiment.csv", "social_media")
        await self.load_dataset("products.csv", "products")
        
        logger.info("Finished loading all datasets into MongoDB.")
        return True
    
    async def load_dataset(self, filename: str, collection_name: str):
        """
        Load a specific dataset into a MongoDB collection.
        
        Args:
            filename: The CSV file name
            collection_name: The MongoDB collection name
        """
        file_path = self.data_dir / filename
        
        if not os.path.exists(file_path):
            logger.warning(f"Dataset file {file_path} not found. Skipping.")
            return False
        
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} records from {filename}")
            
            # Convert to list of dictionaries for MongoDB
            records = df.to_dict("records")
            
            # Clear existing collection
            await self.db[collection_name].delete_many({})
            
            # Insert data
            if records:
                result = await self.db[collection_name].insert_many(records)
                logger.info(f"Inserted {len(result.inserted_ids)} records into {collection_name} collection")
            
            # Create indexes based on collection
            if collection_name == "demographics" or collection_name == "accounts" or collection_name == "credit_history":
                await self.db[collection_name].create_index("user_id")
            
            elif collection_name == "investments" or collection_name == "transactions" or collection_name == "social_media":
                await self.db[collection_name].create_index([("user_id", 1)])
            
            elif collection_name == "products":
                await self.db[collection_name].create_index("product_id")
            
            return True
        
        except Exception as e:
            logger.error(f"Error loading dataset {filename} into collection {collection_name}: {str(e)}")
            return False
    
    async def get_user_data(self, user_id: str):
        """
        Retrieve all data for a specific user from all collections.
        
        Args:
            user_id: The user ID to retrieve data for
            
        Returns:
            Dictionary with all user data
        """
        result = {
            "demographics": None,
            "account": None,
            "credit": None,
            "investments": [],
            "transactions": [],
            "social_media": []
        }
        
        # Get demographic data
        demographics = await self.db["demographics"].find_one({"user_id": user_id})
        if demographics:
            result["demographics"] = demographics
        
        # Get account data
        account = await self.db["accounts"].find_one({"user_id": user_id})
        if account:
            result["account"] = account
        
        # Get credit history
        credit = await self.db["credit_history"].find_one({"user_id": user_id})
        if credit:
            result["credit"] = credit
        
        # Get investment data
        investments_cursor = self.db["investments"].find({"user_id": user_id})
        async for investment in investments_cursor:
            result["investments"].append(investment)
        
        # Get transaction data
        transactions_cursor = self.db["transactions"].find({"user_id": user_id})
        async for transaction in transactions_cursor:
            result["transactions"].append(transaction)
        
        # Get social media data
        social_cursor = self.db["social_media"].find({"user_id": user_id})
        async for social in social_cursor:
            result["social_media"].append(social)
        
        return result
    
    async def list_all_products(self):
        """
        Retrieve all financial products.
        
        Returns:
            List of financial products
        """
        products = []
        products_cursor = self.db["products"].find({})
        async for product in products_cursor:
            products.append(product)
        
        return products 