import logging
import asyncio
import os
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
import pandas as pd
from datetime import datetime, date
from typing import Dict, List, Any, Optional
from pathlib import Path

from app.config import settings
from app.utils.data_loader import DataLoader
from app.database import connect_to_mongo, get_database, close_mongo_connection
from app.repository.user_repository import UserRepository
from app.repository.chat_repository import ChatRepository
from app.repository.document_repository import DocumentRepository
from app.repository.financial_repository import FinancialRepository

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def connect_to_mongodb() -> AsyncIOMotorDatabase:
    """
    Connect to MongoDB and return the database instance.
    
    Returns:
        AsyncIOMotorDatabase: The MongoDB database instance
    """
    try:
        # Create a MongoDB client
        client = AsyncIOMotorClient(settings.MONGODB_URL)
        
        # Access the database
        db = client[settings.MONGODB_DB_NAME]
        
        # Verify connection
        await db.command("ping")
        logger.info(f"Connected to MongoDB: {settings.MONGODB_URL}, database: {settings.MONGODB_DB_NAME}")
        
        return db
    
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {str(e)}")
        raise

class DataLoader:
    """Utility class to load data from CSV files into MongoDB."""
    
    def __init__(self, db):
        """Initialize with database connection."""
        self.db = db
        self.data_dir = Path("./data")
        self.financial_repo = FinancialRepository(db)
        
    async def load_data(self):
        """Load all datasets into MongoDB."""
        await self.load_demographic_data()
        await self.load_account_data()
        await self.load_transaction_data()
        await self.load_credit_history()
        await self.load_investment_data()
        await self.load_product_data()
        logger.info("Completed loading all datasets")
        
    async def load_demographic_data(self):
        """Load demographic data from CSV."""
        try:
            csv_path = self.data_dir / "demographic_data.csv"
            if not csv_path.exists():
                logger.warning(f"Demographic data file not found at {csv_path}")
                return
                
            df = pd.read_csv(csv_path)
            data = df.to_dict('records')
            
            # Preprocess data for MongoDB
            for item in data:
                item["_id"] = item.get("_id", None)
            
            count = await self.financial_repo.bulk_load_demographics(data)
            logger.info(f"Loaded {count} demographic records")
            
        except Exception as e:
            logger.error(f"Error loading demographic data: {str(e)}")
    
    async def load_account_data(self):
        """Load account data from CSV."""
        try:
            csv_path = self.data_dir / "account_data.csv"
            if not csv_path.exists():
                logger.warning(f"Account data file not found at {csv_path}")
                return
                
            df = pd.read_csv(csv_path)
            
            # Convert date strings to date objects
            for date_col in ['account_opening_date']:
                if date_col in df.columns:
                    df[date_col] = pd.to_datetime(df[date_col]).dt.date
                    
            data = df.to_dict('records')
            
            # Preprocess data for MongoDB
            for item in data:
                item["_id"] = item.get("_id", None)
            
            count = await self.financial_repo.bulk_load_accounts(data)
            logger.info(f"Loaded {count} account records")
            
        except Exception as e:
            logger.error(f"Error loading account data: {str(e)}")
    
    async def load_transaction_data(self):
        """Load transaction data from CSV."""
        try:
            csv_path = self.data_dir / "transaction_data.csv"
            if not csv_path.exists():
                logger.warning(f"Transaction data file not found at {csv_path}")
                return
                
            df = pd.read_csv(csv_path)
            
            # Convert date strings to date objects
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date']).dt.date
                    
            data = df.to_dict('records')
            
            # Preprocess data for MongoDB
            for item in data:
                item["_id"] = item.get("_id", None)
            
            count = await self.financial_repo.bulk_load_transactions(data)
            logger.info(f"Loaded {count} transaction records")
            
        except Exception as e:
            logger.error(f"Error loading transaction data: {str(e)}")
    
    async def load_credit_history(self):
        """Load credit history data from CSV."""
        try:
            csv_path = self.data_dir / "credit_history.csv"
            if not csv_path.exists():
                logger.warning(f"Credit history file not found at {csv_path}")
                return
                
            df = pd.read_csv(csv_path)
            data = df.to_dict('records')
            
            # Preprocess data for MongoDB
            for item in data:
                item["_id"] = item.get("_id", None)
            
            count = await self.financial_repo.bulk_load_credit_history(data)
            logger.info(f"Loaded {count} credit history records")
            
        except Exception as e:
            logger.error(f"Error loading credit history data: {str(e)}")
    
    async def load_investment_data(self):
        """Load investment data from CSV."""
        try:
            csv_path = self.data_dir / "investment_data.csv"
            if not csv_path.exists():
                logger.warning(f"Investment data file not found at {csv_path}")
                return
                
            df = pd.read_csv(csv_path)
            
            # Convert date strings to date objects
            if 'start_date' in df.columns:
                df['start_date'] = pd.to_datetime(df['start_date']).dt.date
                    
            data = df.to_dict('records')
            
            # Preprocess data for MongoDB
            for item in data:
                item["_id"] = item.get("_id", None)
            
            count = await self.financial_repo.bulk_load_investments(data)
            logger.info(f"Loaded {count} investment records")
            
        except Exception as e:
            logger.error(f"Error loading investment data: {str(e)}")
    
    async def load_product_data(self):
        """Load financial product data from CSV."""
        try:
            csv_path = self.data_dir / "products.csv"
            if not csv_path.exists():
                logger.warning(f"Products data file not found at {csv_path}")
                return
                
            df = pd.read_csv(csv_path)
            data = df.to_dict('records')
            
            # Preprocess data for MongoDB
            for item in data:
                item["_id"] = item.get("_id", None)
            
            count = await self.financial_repo.bulk_load_products(data)
            logger.info(f"Loaded {count} product records")
            
        except Exception as e:
            logger.error(f"Error loading product data: {str(e)}")

async def create_indexes(db):
    """Create indexes for all collections."""
    logger.info("Creating database indexes...")
    
    # Initialize repositories
    user_repo = UserRepository(db)
    chat_repo = ChatRepository(db)
    document_repo = DocumentRepository(db)
    financial_repo = FinancialRepository(db)
    
    # Create indexes
    await user_repo.create_indexes()
    await chat_repo.create_indexes()
    await document_repo.create_indexes()
    await financial_repo.create_indexes()
    
    logger.info("Database indexes created successfully")

async def initialize_database():
    """Initialize the database with sample data."""
    logger.info("Initializing database...")
    
    # Connect to MongoDB
    await connect_to_mongo()
    db = get_database()
    
    try:
        # Create indexes
        await create_indexes(db)
        
        # Load data from CSV files
        loader = DataLoader(db)
        await loader.load_data()
        
        logger.info("Database initialization completed successfully")
        
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        
    finally:
        # Close MongoDB connection
        await close_mongo_connection()

if __name__ == "__main__":
    asyncio.run(initialize_database()) 