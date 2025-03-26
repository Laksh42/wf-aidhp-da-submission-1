from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Optional, Any

from app.models.user import User
from app.models.chat import (
    ChatMessage, ChatMessageCreate, Conversation, 
    ConversationCreate, ConversationUpdate, ConversationSummary
)
from app.repository.chat_repository import ChatRepository
from app.dependencies import get_current_active_user, get_chat_repository
from app.services.llm_service import generate_llm_response  # Import your LLM service

router = APIRouter()

# Conversation endpoints

@router.post("/conversations", response_model=Conversation, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    data: ConversationCreate,
    current_user: User = Depends(get_current_active_user),
    chat_repo: ChatRepository = Depends(get_chat_repository)
) -> Any:
    """
    Create a new conversation.
    """
    # Create the conversation with the current user's ID
    conversation = await chat_repo.create_conversation(data, str(current_user.id))
    return conversation

@router.get("/conversations", response_model=List[ConversationSummary])
async def list_conversations(
    skip: int = 0,
    limit: int = 20,
    current_user: User = Depends(get_current_active_user),
    chat_repo: ChatRepository = Depends(get_chat_repository)
) -> Any:
    """
    List conversations for the current user.
    """
    conversations = await chat_repo.list_user_conversations(str(current_user.id), skip, limit)
    return conversations

@router.get("/conversations/{conversation_id}", response_model=Conversation)
async def get_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_active_user),
    chat_repo: ChatRepository = Depends(get_chat_repository)
) -> Any:
    """
    Get a specific conversation.
    """
    conversation = await chat_repo.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    # Check if user owns the conversation
    if conversation.user_id != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this conversation"
        )
    
    return conversation

@router.put("/conversations/{conversation_id}", response_model=Conversation)
async def update_conversation(
    conversation_id: str,
    data: ConversationUpdate,
    current_user: User = Depends(get_current_active_user),
    chat_repo: ChatRepository = Depends(get_chat_repository)
) -> Any:
    """
    Update a conversation.
    """
    conversation = await chat_repo.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    # Check if user owns the conversation
    if conversation.user_id != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this conversation"
        )
    
    updated_conversation = await chat_repo.update_conversation(conversation_id, data)
    return updated_conversation

@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_active_user),
    chat_repo: ChatRepository = Depends(get_chat_repository)
) -> None:
    """
    Delete a conversation.
    """
    conversation = await chat_repo.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    # Check if user owns the conversation
    if conversation.user_id != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this conversation"
        )
    
    deleted = await chat_repo.delete_conversation(conversation_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete conversation"
        )

# Message endpoints

@router.get("/conversations/{conversation_id}/messages", response_model=List[ChatMessage])
async def get_messages(
    conversation_id: str,
    skip: int = 0,
    limit: int = 50,
    current_user: User = Depends(get_current_active_user),
    chat_repo: ChatRepository = Depends(get_chat_repository)
) -> Any:
    """
    Get messages for a conversation.
    """
    conversation = await chat_repo.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    # Check if user owns the conversation
    if conversation.user_id != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this conversation"
        )
    
    messages = await chat_repo.get_conversation_messages(conversation_id, skip, limit)
    return messages

@router.post("/chat", response_model=ChatMessage)
async def send_message(
    message: ChatMessageCreate,
    current_user: User = Depends(get_current_active_user),
    chat_repo: ChatRepository = Depends(get_chat_repository)
) -> Any:
    """
    Send a message and get an AI response.
    """
    # Check if conversation exists and user has access
    conversation = await chat_repo.get_conversation(message.conversation_id)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    if conversation.user_id != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this conversation"
        )
    
    # Save user message
    user_message = await chat_repo.create_message(message)
    
    try:
        # Get conversation context (for LLM)
        context = await chat_repo.get_conversation_context(message.conversation_id)
        
        # Generate AI response using LLM service
        ai_response = await generate_llm_response(context, str(current_user.id))
        
        # Save AI response
        ai_message = ChatMessageCreate(
            conversation_id=message.conversation_id,
            role="assistant",  # Use lowercase to match MessageRole enum
            content=ai_response,
            metadata={"generated": True}
        )
        assistant_message = await chat_repo.create_message(ai_message)
        
        return assistant_message
    except Exception as e:
        # Log the error
        logger.error(f"Error generating response: {str(e)}")
        
        # Create a fallback message
        fallback_response = "I apologize, but I encountered an error processing your request. Please try again later."
        
        fallback_message = ChatMessageCreate(
            conversation_id=message.conversation_id,
            role="assistant",  # Use lowercase to match MessageRole enum
            content=fallback_response,
            metadata={"error": str(e), "fallback": True}
        )
        
        # Save fallback response
        assistant_message = await chat_repo.create_message(fallback_message)
        return assistant_message 