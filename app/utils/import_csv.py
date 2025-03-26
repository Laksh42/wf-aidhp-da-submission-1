#!/usr/bin/env python3

import os
import csv
import asyncio
import logging
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import BulkWriteError

from app.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("csv_import")

# Path to CSV data files
DATA_DIR = "data"

async def connect_to_mongo():
    """Connect to MongoDB."""
    try:
        client = AsyncIOMotorClient(settings.MONGODB_URL)
        db = client[settings.MONGODB_DB]
        
        # Verify connection
        await db.command("ping")
        logger.info(f"Connected to MongoDB: {settings.MONGODB_URL}, database: {settings.MONGODB_DB}")
        
        return db
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {str(e)}")
        raise

def csv_to_dict(csv_file):
    """Convert CSV file to list of dictionaries."""
    data = []
    with open(csv_file, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            # Convert numeric values
            for key, value in row.items():
                try:
                    if '.' in value:
                        row[key] = float(value)
                    else:
                        row[key] = int(value)
                except (ValueError, TypeError):
                    pass  # Keep as string if not convertible
            data.append(row)
    return data

async def import_csv_to_collection(db, collection_name, file_name):
    """Import CSV data into MongoDB collection."""
    csv_path = os.path.join(DATA_DIR, file_name)
    if not os.path.exists(csv_path):
        logger.warning(f"CSV file not found: {csv_path}")
        return 0
    
    data = csv_to_dict(csv_path)
    if not data:
        logger.warning(f"No data found in {csv_path}")
        return 0
    
    # Check if collection already has data
    count = await db[collection_name].count_documents({})
    if count > 0:
        logger.info(f"Collection {collection_name} already has {count} documents. Skipping import.")
        return count
    
    # Create index on user_id for better lookup performance
    await db[collection_name].create_index("user_id")
    
    try:
        result = await db[collection_name].insert_many(data)
        logger.info(f"Imported {len(result.inserted_ids)} records into {collection_name}")
        return len(result.inserted_ids)
    except BulkWriteError as e:
        logger.error(f"Bulk write error: {str(e)}")
        return 0
    except Exception as e:
        logger.error(f"Error importing data: {str(e)}")
        return 0

async def main():
    """Main function to import all CSV files."""
    db = await connect_to_mongo()
    
    # Define CSV files to import
    csv_files = {
        "demographic_data": "demographic_data.csv",
        "account_data": "account_data.csv",
        "credit_history": "credit_history.csv",
        "investment_data": "investment_data.csv",
        "transaction_data": "transaction_data.csv",
        "products": "products.csv",
        "social_media_sentiment": "social_media_sentiment.csv"
    }
    
    total_imported = 0
    for collection, file_name in csv_files.items():
        count = await import_csv_to_collection(db, collection, file_name)
        total_imported += count
    
    logger.info(f"Total records imported: {total_imported}")

if __name__ == "__main__":
    asyncio.run(main()) 