from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from app.dependencies import get_current_active_user, get_chatbot
from app.models.user import User
from app.chatbot.enhanced_chatbot import EnhancedChatbot

router = APIRouter()

class ChatMessage(BaseModel):
    message: str
    image_url: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    recommendations: List[Dict[str, Any]] = []

@router.post("/send", response_model=ChatResponse)
async def send_message(
    chat_input: ChatMessage,
    current_user: User = Depends(get_current_active_user),
    chatbot: EnhancedChatbot = Depends(get_chatbot)
) -> ChatResponse:
    """
    Send a message to the chatbot and get a response.
    Requires authentication.
    """
    try:
        # Process the message using the chatbot
        response, recommendations = await chatbot.process_message(
            user_id=str(current_user.id),
            message=chat_input.message,
            image=None  # TODO: Implement image processing if needed
        )
        
        return ChatResponse(
            response=response,
            recommendations=recommendations
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing chat message: {str(e)}"
        )

@router.get("/history", response_model=List[Dict[str, Any]])
async def get_chat_history(
    current_user: User = Depends(get_current_active_user),
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Get chat history for the current user.
    Requires authentication.
    """
    # TODO: Implement chat history retrieval from database
    return []  # Return empty list for now 