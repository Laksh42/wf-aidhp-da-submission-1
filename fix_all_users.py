import asyncio
import sys
from motor.motor_asyncio import AsyncIOMotorClient
from passlib.context import CryptContext
from app.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

async def fix_all_users():
    # Connect to MongoDB
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    db = client[settings.MONGODB_DB]
    
    # Get all users
    users = await db.users.find().to_list(length=100)
    
    if not users:
        print("No users found in the database")
        return
    
    print(f"Found {len(users)} users in the database")
    
    # Default test password
    test_password = "password"
    
    # Fix each user
    for user in users:
        user_id = user.get('user_id')
        if not user_id:
            print(f"Skipping user with no user_id: {user.get('_id')}")
            continue
            
        print(f"\nFixing user: {user_id}")
        
        # Create a new hash and update user
        new_hash = pwd_context.hash(test_password)
        print(f"New hash: {new_hash[:10]}...")
        
        # Update the user
        await db.users.update_one(
            {"user_id": user_id},
            {"$set": {"hashed_password": new_hash}}
        )
        
        print(f"Updated user {user_id} with new password hash")
    
    print("\nAll users have been updated with the password 'password'")

if __name__ == "__main__":
    asyncio.run(fix_all_users()) 