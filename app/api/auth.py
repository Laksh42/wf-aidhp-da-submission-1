from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Body, Form
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
import logging

from app.models.user import User, UserInDB, UserCreate, Token, TokenData, UserData
from app.repository.user_repository import UserRepository
from app.database.mongodb import get_database
from app.config import settings

# Security configurations
SECRET_KEY = settings.SECRET_KEY
ALGORITHM = settings.JWT_ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")

router = APIRouter()

logger = logging.getLogger(__name__)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

async def get_user(user_id: str):
    db = await get_database()
    user_repo = UserRepository(db)
    user = await user_repo.get_by_user_id(user_id)
    if user:
        return user
    return None

async def authenticate_user(user_id: str, password: str):
    user = await get_user(user_id)
    if not user:
        logger.warning(f"User not found: {user_id}")
        return False
    # Debug the user object
    logger.debug(f"Authenticating user: {user_id}, User object has hashed_password: {hasattr(user, 'hashed_password')}")
    if not verify_password(password, user.hashed_password):
        logger.warning(f"Invalid password for user: {user_id}")
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        token_data = TokenData(sub=user_id, user_id=user_id)
    except JWTError:
        raise credentials_exception
    user = await get_user(user_id=token_data.user_id)
    if user is None:
        raise credentials_exception
    return user

@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    # In OAuth2PasswordRequestForm, username field is used for the user_id
    logger.info(f"Login attempt for user: {form_data.username}")
    user = await authenticate_user(form_data.username, form_data.password)
    if not user:
        logger.warning(f"Failed login attempt for user: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect user ID or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Update last login time
    db = await get_database()
    user_repo = UserRepository(db)
    await user_repo.update_last_login(user.user_id)
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.user_id}, expires_delta=access_token_expires
    )
    logger.info(f"Successful login for user: {user.user_id}")
    return {"access_token": access_token, "token_type": "bearer", "user_id": user.user_id}

@router.post("/register", response_model=User)
async def register_user(user_data: UserCreate = Body(...)):
    logger.info(f"Registration attempt for user: {user_data.user_id}")
    db = await get_database()
    user_repo = UserRepository(db)
    
    # Check if user already exists
    existing_user = await user_repo.get_by_user_id(user_data.user_id)
    if existing_user:
        logger.info(f"User already exists: {user_data.user_id}")
        # User exists, return the existing user
        return existing_user
    
    try:
        # Create new user
        created_user = await user_repo.create(user_data)
        logger.info(f"Successfully registered new user: {created_user.user_id}")
        return created_user
    except ValueError as e:
        logger.error(f"Error registering user {user_data.user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.get("/verify", response_model=dict)
async def verify_token(current_user: User = Depends(get_current_user)):
    """
    Verify if the current token is valid.
    This is used by the frontend to check if a stored token is still valid.
    """
    return {"valid": True, "user_id": current_user.user_id}

@router.get("/me", response_model=User)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """
    Get current user information.
    This is used by the frontend to get user details after login or token verification.
    """
    return current_user

@router.get("/user-data", response_model=UserData)
async def get_user_data(current_user: User = Depends(get_current_user)):
    """
    Get comprehensive user data from various CSV sources.
    """
    db = await get_database()
    user_repo = UserRepository(db)
    raw_user_data = await user_repo.get_user_data(current_user.user_id)
    
    if not raw_user_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User data not found"
        )
    
    # Use the UserData model to validate and serialize the data
    try:
        user_data = UserData.model_validate({
            "user_id": current_user.user_id,
            "demographic_data": raw_user_data.get("demographics"),
            "account_data": raw_user_data.get("account"),
            "credit_history": raw_user_data.get("credit"),
            "investment_data": raw_user_data.get("investments"),
            "recent_transactions": raw_user_data.get("recent_transactions")
        })
        return user_data
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing user data: {str(e)}"
        )

@router.get("/validate")
async def validate_token(current_user: User = Depends(get_current_user)):
    """
    Validate the current token and return user information if valid
    """
    return {
        "valid": True,
        "user_id": current_user.user_id,
        "email": current_user.email
    } 