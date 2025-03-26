import asyncio
import sys
from motor.motor_asyncio import AsyncIOMotorClient
from passlib.context import CryptContext
from app.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

async def fix_user_auth():
    # Connect to MongoDB
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    db = client[settings.MONGODB_DB]
    
    # Get testuser
    user = await db.users.find_one({"user_id": "testuser2"})
    
    if not user:
        print("User 'testuser2' not found")
        return
    
    print(f"Found user: {user['user_id']}")
    
    # Test password verification against existing hash
    test_password = "password"
    hashed_password = user.get('hashed_password')
    
    print(f"Existing hash: {hashed_password}")
    
    # Try verification
    try:
        is_valid = pwd_context.verify(test_password, hashed_password)
        print(f"Password 'password' verification: {is_valid}")
    except Exception as e:
        print(f"Verification error: {str(e)}")
    
    # Create a new hash and update user
    new_hash = pwd_context.hash(test_password)
    print(f"New hash: {new_hash}")
    
    # Update the user with both fields to be safe
    await db.users.update_one(
        {"user_id": "testuser2"},
        {"$set": {"hashed_password": new_hash}}
    )
    
    print("Updated user with new password hash")
    
    # Verify the new hash
    user = await db.users.find_one({"user_id": "testuser2"})
    updated_hash = user.get('hashed_password')
    
    try:
        is_valid = pwd_context.verify(test_password, updated_hash)
        print(f"Verification with new hash: {is_valid}")
    except Exception as e:
        print(f"New hash verification error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(fix_user_auth()) 