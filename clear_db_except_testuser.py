from motor.motor_asyncio import AsyncIOMotorClient
import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def clear_db_except_testuser():
    try:
        # Connect to the MongoDB database using connection string from .env
        mongodb_url = os.getenv('MONGODB_URL', 'mongodb://localhost:27017')
        mongodb_db = os.getenv('MONGODB_DB', 'financial_advisor')
        
        print(f"Connecting to MongoDB at {mongodb_url}, database: {mongodb_db}")
        client = AsyncIOMotorClient(mongodb_url)
        db = client[mongodb_db]
        
        # Count users before deletion
        total_users = await db.users.count_documents({})
        print(f"Total users before deletion: {total_users}")
        
        # Delete all users except testuser
        result = await db.users.delete_many({'user_id': {'$ne': 'testuser'}})
        
        print(f'Deleted {result.deleted_count} users. Only testuser should remain in the database.')
        
        # Verify remaining users
        remaining_users = await db.users.find().to_list(length=100)
        print(f'Remaining users: {len(remaining_users)}')
        for user in remaining_users:
            print(f"  User ID: {user.get('user_id')}, Email: {user.get('email', 'None')}")
    
    except Exception as e:
        print(f'Error: {e}')

# Run the async function
if __name__ == "__main__":
    asyncio.run(clear_db_except_testuser()) 