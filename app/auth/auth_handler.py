from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from motor.motor_asyncio import AsyncIOMotorDatabase
import logging

from app.models.user import User, UserCreate, UserInDB
from app.repository.user_repository import UserRepository
from app.auth.security import create_access_token, decode_access_token
from app.database.mongodb import get_database
from app.config import settings

# Setup logging
logger = logging.getLogger(__name__)

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/token")

class AuthHandler:
    """Handler for authentication operations."""
    
    def __init__(self, db: AsyncIOMotorDatabase = None):
        """Initialize with database connection."""
        self.user_repo = UserRepository(db)
    
    async def register_user(self, user_data: UserCreate) -> User:
        """Register a new user."""
        try:
            # Create user via repository
            user = await self.user_repo.create(user_data)
            return user
        except ValueError as e:
            # User already exists
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        except Exception as e:
            logger.error(f"Error registering user: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error during user registration"
            )
    
    async def authenticate_user(self, username_or_email: str, password: str) -> Dict[str, str]:
        """Authenticate a user and return access token."""
        try:
            # Try to get user by email
            user = await self.user_repo.get_by_email(username_or_email)
            
            # If not found, try by username
            if not user:
                user = await self.user_repo.get_by_username(username_or_email)
            
            # If still not found or password doesn't match
            if not user or not self.user_repo.verify_password(password, user.hashed_password):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid credentials",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            # Create access token
            token_data = {
                "sub": user.id,
                "username": user.username,
                "email": user.email
            }
            
            # Set token expiration based on settings
            access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
            access_token = create_access_token(
                data=token_data, 
                expires_delta=access_token_expires
            )
            
            return {
                "access_token": access_token,
                "token_type": "bearer"
            }
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            logger.error(f"Error authenticating user: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error during authentication"
            )
    
    async def get_current_user(self, token: str = Depends(oauth2_scheme)) -> User:
        """Get the current authenticated user from token."""
        try:
            # Decode the token
            payload = decode_access_token(token)
            if payload is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication credentials",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            # Extract user ID from token
            user_id = payload.get("sub")
            if user_id is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token data",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            # Get user from database
            user_in_db = await self.user_repo.get(user_id)
            if user_in_db is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            # Check if user is active
            if not user_in_db.is_active:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Inactive user",
                )
            
            # Return User model (not UserInDB with hashed password)
            return User(
                id=user_in_db.id,
                username=user_in_db.username,
                email=user_in_db.email,
                full_name=user_in_db.full_name,
                is_active=user_in_db.is_active,
                is_premium=user_in_db.is_premium,
                created_at=user_in_db.created_at,
                updated_at=user_in_db.updated_at,
                preferences=user_in_db.preferences
            )
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            logger.error(f"Error getting current user: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error"
            )

# Dependencies for FastAPI
async def get_auth_handler():
    """Get an AuthHandler instance."""
    db = await get_database()
    return AuthHandler(db)

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    auth_handler: AuthHandler = Depends(get_auth_handler)
) -> User:
    """Dependency for getting the current authenticated user."""
    return await auth_handler.get_current_user(token) 