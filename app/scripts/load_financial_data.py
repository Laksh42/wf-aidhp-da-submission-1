"""Script to load financial data into the database and generate sample data files."""

import asyncio
import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from motor.motor_asyncio import AsyncIOMotorClient
import sys

from app.config import settings
from app.models.financial import InvestmentType
from app.utils.data_loader import DataLoader

# Path to data directory
DATA_DIR = Path(settings.DATA_DIR)

# Synthetic financial products data
FINANCIAL_PRODUCTS = [
    {
        "name": "Blue Chip Growth Fund",
        "category": "mutual_funds",
        "interest_rate": 5.25,
        "term_years": 0,
        "minimum_investment": 1000.0,
        "description": "A high-growth mutual fund focusing on established, large-cap companies.",
        "risk_level": "medium",
        "suitable_for": "Long-term investors seeking growth with some stability"
    },
    {
        "name": "Government Bond Index",
        "category": "bonds",
        "interest_rate": 3.0,
        "term_years": 10,
        "minimum_investment": 500.0,
        "description": "A bond fund tracking U.S. government securities.",
        "risk_level": "low",
        "suitable_for": "Conservative investors seeking stable income"
    }
]

# Save sample data to JSON files
def save_sample_data():
    """Save sample financial data to JSON files."""
    os.makedirs(DATA_DIR / "sample", exist_ok=True)
    products_file = DATA_DIR / "sample" / "products.json"
    
    with open(products_file, 'w') as f:
        json.dump(FINANCIAL_PRODUCTS, f, indent=2)
    
    print(f"Saved sample data to {products_file}")

async def main():
    """Main function to load all financial data from CSV files."""
    print("Starting financial data loading process...")
    print(f"Python version: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    print(f"MONGODB_URL: {settings.MONGODB_URL}")
    print(f"MONGODB_DB_NAME: {settings.MONGODB_DB_NAME}")
    print(f"Data directory: {settings.DATA_DIR}")
    
    # Check if data files exist
    data_dir = Path(settings.DATA_DIR)
    print(f"Data directory exists: {os.path.exists(data_dir)}")
    if os.path.exists(data_dir):
        print(f"Data directory contents: {os.listdir(data_dir)}")
        
        # Check for investment data specifically
        investment_data_path = data_dir / "investment_data.csv"
        print(f"Investment data exists: {os.path.exists(investment_data_path)}")
    
    # Connect to MongoDB
    try:
        print("Connecting to MongoDB...")
        client = AsyncIOMotorClient(settings.MONGODB_URL)
        db = client[settings.MONGODB_DB_NAME]
        
        # Ping the database
        print("Pinging MongoDB...")
        await db.command("ping")
        print("MongoDB connection successful!")
        
        # Initialize the data loader
        print("Initializing DataLoader...")
        loader = DataLoader(db)
        
        # Load all datasets
        print("Loading datasets...")
        success = await loader.load_all_datasets()
        
        if success:
            print("Successfully loaded all financial data into the database.")
        else:
            print("Some datasets could not be loaded. Check the logs for details.")
    
    except Exception as e:
        print(f"Error loading financial data: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Close MongoDB connection
        if 'client' in locals():
            client.close()
            print("MongoDB connection closed")

if __name__ == "__main__":
    asyncio.run(main()) 