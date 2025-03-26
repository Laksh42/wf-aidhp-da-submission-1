import asyncio
import sys
from motor.motor_asyncio import AsyncIOMotorClient
from passlib.context import CryptContext
from app.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

async def debug_users():
    # Connect to MongoDB
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    db = client[settings.MONGODB_DB]
    
    # Get all users
    users = await db.users.find().to_list(length=100)
    
    if not users:
        print("No users found in the database")
        return
    
    print(f"Found {len(users)} users in the database")
    
    # Print user details
    for user in users:
        print("\nUser details:")
        print(f"  _id: {user.get('_id')}")
        print(f"  user_id: {user.get('user_id')}")
        print(f"  email: {user.get('email')}")
        
        # Check for password fields
        hashed_password = user.get('hashed_password')
        password_hash = user.get('password_hash') 
        
        if hashed_password:
            print(f"  hashed_password: {hashed_password[:10]}...")
        else:
            print("  hashed_password: Not found")
        
        if password_hash:
            print(f"  password_hash: {password_hash[:10]}...")
        else:
            print("  password_hash: Not found")
            
        # Test password verification
        test_password = "password"
        if hashed_password:
            is_valid = pwd_context.verify(test_password, hashed_password)
            print(f"  Password 'password' is valid against hashed_password: {is_valid}")
        if password_hash:
            is_valid = pwd_context.verify(test_password, password_hash)
            print(f"  Password 'password' is valid against password_hash: {is_valid}")

if __name__ == "__main__":
    asyncio.run(debug_users()) 