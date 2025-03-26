from fastapi import APIRouter, Depends, HTTPException, status, Body, Request, Response
from typing import Dict, Any, Optional
import logging
from uuid import uuid4
from pydantic import BaseModel
from app.api.dependencies import get_current_user
from app.chatbot.enhanced_chatbot import EnhancedChatbot
from app.models.user import User
from app.config import settings
import time

router = APIRouter()
logger = logging.getLogger(__name__)

enhanced_chatbot = EnhancedChatbot()

class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatMessageResponse(BaseModel):
    response: str
    session_id: str
    recommendations: list = []

@router.post("/send", response_model=ChatMessageResponse)
async def send_message(
    message_data: ChatMessage,
    current_user: User = Depends(get_current_user)
):
    """
    Send a message to the chatbot and get a response
    """
    start_time = time.time()
    
    # Log incoming request
    logger.info(f"Processing chat request from user: {current_user.user_id}")
    
    # Validate session_id or create new one
    session_id = message_data.session_id if message_data.session_id else str(uuid4())
    
    try:
        # Get response from chatbot
        response_text, recommendations = enhanced_chatbot.generate_response(
            message_data.message,
            session_id=session_id,
            user_id=current_user.user_id
        )
        
        # Log successful processing
        processing_time = time.time() - start_time
        logger.info(f"Chat request processed in {processing_time:.2f}s for user: {current_user.user_id}")
        
        # Return response with session_id and recommendations
        return ChatMessageResponse(
            response=response_text,
            session_id=session_id,
            recommendations=recommendations
        )
    except Exception as e:
        # Log error
        logger.error(f"Error processing chat request: {str(e)}", exc_info=True)
        
        # Return appropriate HTTP exception
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing your message: {str(e)}"
        )

@router.get("/history/{session_id}")
async def get_chat_history(
    session_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get the chat history for a specific session
    """
    try:
        # Get chat history from chatbot
        history = enhanced_chatbot.get_chat_history(session_id)
        
        if not history:
            return {"messages": []}
            
        return {"messages": history}
    except Exception as e:
        logger.error(f"Error retrieving chat history: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve chat history: {str(e)}"
        ) 