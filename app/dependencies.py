from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from typing import Optional
from datetime import datetime, timedelta

from app.database.mongodb import get_database
from app.config import settings
from app.chatbot.enhanced_chatbot import EnhancedChatbot
from app.conversation.memory import ConversationMemory
from app.recommendations.engine import RecommendationEngine
from app.repository.chat_repository import ChatRepository
from app.repository.user_repository import UserRepository
from app.repository.document_repository import DocumentRepository
from app.models.user import User, TokenData

# OAuth2 setup
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# User authentication dependencies
async def get_user_repository():
    """Dependency to get the user repository."""
    db = await get_database()
    return UserRepository(db)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Get the current authenticated user from the JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(sub=username, user_id=username)
    except JWTError:
        raise credentials_exception
    
    user_repo = await get_user_repository()
    user = await user_repo.get_by_user_id(token_data.user_id)
    if user is None:
        raise credentials_exception
    
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    """Get the current active user."""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# Document repository dependency
async def get_document_repository():
    """Dependency to get the document repository."""
    db = await get_database()
    return DocumentRepository(db)

# Chat repository dependency
async def get_chat_repository():
    """Dependency to get the chat repository."""
    db = await get_database()
    return ChatRepository(db)

# Chatbot and related dependencies
async def get_conversation_memory():
    """Dependency to get the conversation memory."""
    db = await get_database()
    return ConversationMemory(db)

async def get_recommendation_engine():
    """Dependency to get the recommendation engine."""
    db = await get_database()
    return RecommendationEngine(db)

async def get_chatbot(
    memory: ConversationMemory = Depends(get_conversation_memory),
    recommendation_engine: RecommendationEngine = Depends(get_recommendation_engine)
):
    """Dependency to get the enhanced chatbot."""
    return EnhancedChatbot(memory, recommendation_engine) 