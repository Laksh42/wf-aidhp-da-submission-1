import pandas as pd
import os
from pathlib import Path
import logging
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import BulkWriteError

from app.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def initialize_database():
    """
    Initialize the MongoDB database with sample data from CSV files.
    This function loads data from CSV files and creates necessary collections.
    It also ensures that user IDs in the data match those in the users collection.
    """
    logger.info("Initializing database with sample data...")
    
    # Connect to MongoDB
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    db = client[settings.MONGODB_DB]
    
    # Get existing collections
    collections = await db.list_collection_names()
    logger.info(f"Available collections: {', '.join(collections)}")
    
    # Load data from CSV files
    data_files = {
        'demographic_data': 'demographic_data.csv',
        'account_data': 'account_data.csv',
        'credit_history': 'credit_history.csv',
        'investment_data': 'investment_data.csv',
        'transaction_data': 'transaction_data.csv',
        'social_media_sentiment': 'social_media_sentiment.csv',
        'products': 'products.csv'
    }
    
    data_dir = Path(settings.DATA_DIR)
    total_imported = 0
    
    # Import each CSV file
    for collection_name, filename in data_files.items():
        file_path = data_dir / filename
        
        if not os.path.exists(file_path):
            logger.warning(f"File {filename} not found. Skipping import for {collection_name}.")
            continue
        
        # Check if collection already has data
        existing_count = await db[collection_name].count_documents({})
        if existing_count > 0:
            logger.info(f"Collection {collection_name} already has {existing_count} documents. Skipping import.")
            total_imported += existing_count
            continue
        
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            records = df.to_dict('records')
            
            if records:
                # Insert into collection
                result = await db[collection_name].insert_many(records)
                logger.info(f"Imported {len(result.inserted_ids)} records into {collection_name}")
                total_imported += len(result.inserted_ids)
                
                # Create indexes for collections
                if collection_name in ['demographic_data', 'account_data', 'credit_history']:
                    await db[collection_name].create_index("user_id")
                elif collection_name in ['investment_data', 'transaction_data', 'social_media_sentiment']:
                    await db[collection_name].create_index("user_id")
                elif collection_name == 'products':
                    await db[collection_name].create_index("product_id")
                    
        except Exception as e:
            logger.error(f"Error importing {filename} to {collection_name}: {str(e)}")
    
    # Check if there are user records
    users_count = await db.users.count_documents({})
    if users_count == 0:
        logger.warning("No users found in the database. Creating test user.")
        
        # Create a test user
        from passlib.context import CryptContext
        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
        test_user = {
            "user_id": "testuser",
            "email": "test@example.com",
            "hashed_password": pwd_context.hash("password"),
            "is_active": True
        }
        
        await db.users.insert_one(test_user)
        logger.info("Created test user 'testuser' with password 'password'")
    
    # Generate meta-prompts for existing users
    user_ids = []
    async for user in db.users.find({}, {"user_id": 1}):
        user_ids.append(user["user_id"])
    
    logger.info(f"Generating meta-prompts for {len(user_ids)} users")
    
    # Import the MetaPromptGenerator class
    from app.models.meta_prompt_generator import MetaPromptGenerator
    
    # Create an instance of MetaPromptGenerator
    meta_prompt_generator = MetaPromptGenerator(db)
    
    # Generate meta-prompts for each user
    for user_id in user_ids:
        meta_prompt = await meta_prompt_generator.generate_meta_prompt(user_id)
        
        # Store meta-prompt in the database
        await db.meta_prompts.update_one(
            {"user_id": user_id},
            {"$set": {"user_id": user_id, "prompt_text": meta_prompt, "updated_at": pd.Timestamp.now()}},
            upsert=True
        )
        
        logger.info(f"Generated meta-prompt for user {user_id}")
    
    logger.info(f"Database initialization complete. Total records imported: {total_imported}")
    
    # Return statistics
    return {
        "records_imported": total_imported,
        "users": users_count,
        "collections": len(collections)
    }

# Function to add additional field to demographic data with synthetic information
async def add_synthetic_data():
    """
    Add synthetic financial goal data to demographic records to enhance personalization.
    """
    # Connect to MongoDB
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    db = client[settings.MONGODB_DB]
    
    # Check if demographic_data collection exists
    if "demographic_data" not in await db.list_collection_names():
        logger.warning("demographic_data collection not found. Skipping synthetic data addition.")
        return
    
    # Define possible financial goals based on different user segments
    financial_goals = {
        "young_professional": ["Build emergency fund", "Pay off student loans", "Save for home down payment", "Start retirement planning"],
        "mid_career": ["Maximize retirement contributions", "College savings for children", "Investment diversification", "Pay off mortgage early"],
        "pre_retirement": ["Retirement planning", "Estate planning", "Healthcare cost planning", "Downsize home"],
        "retired": ["Income generation", "Estate management", "Healthcare expense management", "Legacy planning"],
        "business_owner": ["Business expansion", "Succession planning", "Tax optimization", "Business asset diversification"]
    }
    
    risk_tolerance_mapping = {
        "20-30": "Aggressive",
        "31-40": "Moderately Aggressive",
        "41-50": "Moderate",
        "51-60": "Moderately Conservative",
        "61+": "Conservative"
    }
    
    # Get all demographic records
    demographic_records = []
    async for record in db.demographic_data.find({}):
        demographic_records.append(record)
    
    # Update each record with synthetic data
    for record in demographic_records:
        user_id = record["user_id"]
        age = record.get("age", 35)
        occupation = record.get("occupation", "").lower()
        
        # Determine user segment based on age and occupation
        segment = "mid_career"  # default
        
        if age < 30:
            segment = "young_professional"
        elif age >= 30 and age < 45:
            segment = "mid_career"
        elif age >= 45 and age < 60:
            segment = "pre_retirement"
        elif age >= 60:
            segment = "retired"
        
        if "business" in occupation or "owner" in occupation or "entrepreneur" in occupation:
            segment = "business_owner"
        
        # Assign financial goals based on segment
        import random
        goals = random.sample(financial_goals[segment], min(3, len(financial_goals[segment])))
        
        # Determine risk tolerance based on age
        age_bracket = "20-30" if age < 30 else "31-40" if age < 40 else "41-50" if age < 50 else "51-60" if age < 60 else "61+"
        risk_tolerance = risk_tolerance_mapping[age_bracket]
        
        # Add random financial values
        annual_expenses = round(float(record.get("annual_income", 70000)) * random.uniform(0.5, 0.8), 2)
        
        # Update the record with synthetic data
        await db.demographic_data.update_one(
            {"user_id": user_id},
            {"$set": {
                "financial_goals": goals,
                "risk_tolerance": risk_tolerance,
                "annual_expenses": annual_expenses
            }}
        )
        
        logger.info(f"Added synthetic data for user {user_id}")
    
    logger.info(f"Added synthetic data to {len(demographic_records)} demographic records")

if __name__ == "__main__":
    # Run the initialization function
    asyncio.run(initialize_database())
    
    # Add synthetic data
    asyncio.run(add_synthetic_data()) 