from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from typing import Dict, Any
import logging
from datetime import timedelta

from app.models.user import UserCreate, User, UserInDB
from app.database.user_db import get_user_by_email, create_user
from app.core.security import create_access_token, verify_password, get_password_hash
from app.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/login", response_model=Dict[str, Any])
async def login(
    form_data: OAuth2PasswordRequestForm = Depends()
) -> Dict[str, Any]:
    """
    Login endpoint for user authentication.
    
    Args:
        form_data: OAuth2 form with username (email) and password
        
    Returns:
        Dict with access token and token type
    """
    try:
        user = await get_user_by_email(form_data.username)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        if not verify_password(form_data.password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Create access token
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.email}, expires_delta=access_token_expires
        )
        
        logger.info(f"User {user.email} logged in successfully")
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user_id": str(user.id),
            "email": user.email,
            "full_name": user.full_name
        }
        
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login error: {str(e)}",
        )

@router.post("/register", response_model=Dict[str, Any])
async def register(user_data: UserCreate) -> Dict[str, Any]:
    """
    Register a new user.
    
    Args:
        user_data: User registration data
        
    Returns:
        Dict with user details (without password)
    """
    try:
        # Check if user with this email already exists
        existing_user = await get_user_by_email(user_data.email)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered",
            )
        
        # Hash the password
        hashed_password = get_password_hash(user_data.password)
        
        # Create user in DB
        user_in_db = UserInDB(
            email=user_data.email,
            full_name=user_data.full_name,
            hashed_password=hashed_password,
        )
        
        created_user = await create_user(user_in_db)
        
        logger.info(f"New user registered: {created_user.email}")
        
        return {
            "user_id": str(created_user.id),
            "email": created_user.email,
            "full_name": created_user.full_name,
            "message": "User registered successfully"
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration error: {str(e)}",
        )