import logging
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import openai
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.config import settings
from app.models.conversation import Message, MessageRole, Conversation
from app.repository.conversation_repository import ConversationRepository
from app.models.meta_prompt_generator import MetaPromptGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatService:
    """Service for managing chat conversations and generating responses."""
    
    def __init__(self, db: AsyncIOMotorDatabase = None, meta_prompt_generator: MetaPromptGenerator = None):
        """Initialize the chat service."""
        self.conversation_repo = ConversationRepository(db)
        self.meta_prompt_generator = meta_prompt_generator
        self.client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_MODEL
        
    async def generate_response(self, conversation_id: str, user_id: str) -> Optional[Message]:
        """Generate a response to the conversation."""
        try:
            # Get the conversation from the repository
            conversation = await self.conversation_repo.get(conversation_id)
            if not conversation or conversation.user_id != user_id:
                logger.error(f"Conversation not found or does not belong to user: {conversation_id}")
                return None
            
            # Get or create meta-prompt for this user
            meta_prompt = await self._get_or_create_meta_prompt(user_id, conversation)
            
            # Prepare messages for the LLM
            messages = self._prepare_messages(conversation, meta_prompt)
            
            # Call the LLM to generate a response
            response = await self._call_llm(messages)
            if not response:
                return Message(
                    role=MessageRole.ASSISTANT,
                    content="I'm sorry, I'm having trouble generating a response right now. Please try again later.",
                    timestamp=datetime.utcnow()
                )
            
            # Create and store the assistant message
            assistant_message = Message(
                role=MessageRole.ASSISTANT,
                content=response,
                timestamp=datetime.utcnow()
            )
            
            # Add the message to the conversation
            await self.conversation_repo.add_message(conversation_id, assistant_message)
            
            return assistant_message
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return Message(
                role=MessageRole.ASSISTANT,
                content="I encountered an error while processing your request. Please try again.",
                timestamp=datetime.utcnow()
            )
    
    async def _get_or_create_meta_prompt(self, user_id: str, conversation: Conversation) -> str:
        """Get or create a meta-prompt for the user."""
        # If conversation already has a meta-prompt, use that
        if conversation.meta_prompt:
            return conversation.meta_prompt
        
        # Generate a new meta-prompt if we have a generator
        if self.meta_prompt_generator:
            try:
                meta_prompt = await self.meta_prompt_generator.generate_meta_prompt(user_id)
                # Save the meta-prompt to the conversation
                await self.conversation_repo.set_meta_prompt(conversation.id, meta_prompt)
                return meta_prompt
            except Exception as e:
                logger.error(f"Error generating meta-prompt: {str(e)}")
                # Fall back to default meta-prompt
                return "You are a helpful assistant for a financial advisor application."
        
        # Default meta-prompt if no generator
        return "You are a helpful assistant for a financial advisor application."
    
    def _prepare_messages(self, conversation: Conversation, meta_prompt: str) -> List[Dict[str, Any]]:
        """Prepare the messages for the LLM."""
        messages = []
        
        # Add system message with user context (meta-prompt)
        messages.append({
            "role": "system",
            "content": meta_prompt
        })
        
        # Add the last 10 messages from the conversation to maintain context
        recent_messages = conversation.messages[-10:] if len(conversation.messages) > 10 else conversation.messages
        
        for msg in recent_messages:
            messages.append({
                "role": msg.role.value,
                "content": msg.content
            })
        
        return messages
    
    async def _call_llm(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """Call the LLM to generate a response."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            # Extract the assistant's response
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content
            return None
        except Exception as e:
            logger.error(f"Error calling LLM: {str(e)}")
            return None 