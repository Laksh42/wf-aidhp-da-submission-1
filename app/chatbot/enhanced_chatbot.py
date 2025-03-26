from typing import List, Dict, Any, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from PIL import Image
import numpy as np
from app.conversation.memory import ConversationMemory
from app.recommendations.engine import RecommendationEngine
from app.config import settings
import logging
import os

logger = logging.getLogger(__name__)

class EnhancedChatbot:
    def __init__(
        self,
        memory: ConversationMemory,
        recommendation_engine: RecommendationEngine
    ):
        self.memory = memory
        self.recommendation_engine = recommendation_engine
        
        # Check for GPU availability for embedding models only
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("Using GPU for inference")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU for inference - no GPU available")
        
        # Initialize embedding model for semantic search
        try:
            self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
            self.embedding_model.to(self.device)
            logger.info(f"Successfully loaded embedding model: {settings.EMBEDDING_MODEL}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            try:
                # Fallback to a smaller embedding model
                self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                self.embedding_model.to(self.device)
                logger.info("Successfully loaded embedding model: sentence-transformers/all-MiniLM-L6-v2")
            except Exception as e2:
                logger.error(f"Failed to load fallback embedding model: {e2}")
                self.embedding_model = None
        
        # Initialize image analysis model
        try:
            self.image_model = SentenceTransformer("clip-ViT-B-32")
            self.image_model.to(self.device)
            logger.info("Successfully loaded image analysis model")
        except Exception as e:
            logger.error(f"Failed to load image model: {e}")
            self.image_model = None
            logger.warning("Image analysis disabled due to model loading failure")
        
    async def process_message(
        self,
        user_id: str,
        message: str,
        image: Optional[Image.Image] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Process user message and generate response with recommendations."""
        # Get user context
        user_context = await self.memory.get_user_context(user_id)
        
        # Process image if provided and image model is available
        image_embedding = None
        if image and self.image_model is not None:
            try:
                image_embedding = self._process_image(image)
            except Exception as e:
                logger.error(f"Failed to process image: {e}")
        
        # Generate response
        response = await self._generate_response(
            message,
            user_context,
            image_embedding,
            context
        )
        
        # Get personalized recommendations
        recommendations = await self.recommendation_engine.get_personalized_recommendations(
            user_id,
            context or {}
        )
        
        # Store interaction
        await self.memory.store_interaction(
            user_id,
            message,
            response,
            {
                "image_embedding": image_embedding.tolist() if image_embedding is not None else None,
                "context": context,
                "recommendations": recommendations
            }
        )
        
        return response, recommendations
    
    async def _generate_response(
        self,
        message: str,
        user_context: Dict[str, Any],
        image_embedding: Optional[np.ndarray] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate response using the LLM service."""
        # Prepare prompt with context
        prompt = self._prepare_prompt(
            message,
            user_context,
            image_embedding,
            context
        )
        
        try:
            # Check if embedding models were properly loaded
            if self.embedding_model is None:
                logger.error("Embedding model not properly initialized")
                return "I apologize, but my language model is currently unavailable. Please try again later."
                
            # Generate response using the LLM service
            response = await self._generate_response_from_llm(prompt)
            
            # Clean up response
            response = self._clean_response(response, prompt)
            
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            return "I apologize, but I encountered an error while processing your request. Please try again with a simpler question."
    
    def _process_image(self, image: Image.Image) -> np.ndarray:
        """Process image and extract embeddings."""
        if self.image_model is None:
            raise RuntimeError("Image model not available")
            
        # Resize image if needed
        image = image.resize((224, 224))
        
        # Convert to tensor and normalize
        image_tensor = torch.tensor(np.array(image)).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
        
        # Get image embedding
        with torch.no_grad():
            image_embedding = self.image_model.encode(image_tensor)
        
        return image_embedding
    
    def _prepare_prompt(
        self,
        message: str,
        user_context: Dict[str, Any],
        image_embedding: Optional[np.ndarray] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Prepare prompt for the model with proper context."""
        # Create a simple prompt structure - actual formatting will happen in LLMService
        return message
    
    def _clean_response(self, response: str, prompt: str) -> str:
        """Clean up generated response."""
        try:
            # Remove prompt from response
            if prompt in response:
                response = response.replace(prompt, "").strip()
            
            # Remove any remaining special tokens
            response = response.replace("<s>", "").replace("</s>", "").strip()
            
            # Extract just the assistant's response after the [/INST] tag
            if "[/INST]" in response:
                response = response.split("[/INST]")[1].strip()
            
            # Ensure response is not empty
            if not response:
                response = "I apologize, but I couldn't generate a proper response. Could you please rephrase your question?"
            
            return response
        except Exception as e:
            logger.error(f"Error in clean_response: {e}")
            return "I apologize, but I encountered an error processing your request. How else can I help you with your financial needs?"
    
    async def process_feedback(
        self,
        user_id: str,
        interaction_id: str,
        rating: int,
        feedback_text: Optional[str] = None
    ) -> None:
        """Process user feedback for RLHF."""
        # Store feedback
        await self.memory.store_feedback(
            user_id,
            interaction_id,
            rating,
            feedback_text
        )
        
        # Get feedback statistics
        feedback_stats = await self.memory.get_feedback_stats(user_id)
        
        # Update model behavior based on feedback
        if feedback_stats["total_feedback"] > 0:
            self._adjust_model_behavior(feedback_stats)
    
    def _adjust_model_behavior(self, feedback_stats: Dict[str, Any]) -> None:
        """Adjust model behavior based on feedback statistics."""
        # Implement model fine-tuning or parameter adjustment based on feedback
        # This is a placeholder for actual RLHF implementation
        pass
    
    async def _generate_response_from_llm(self, prompt: str) -> str:
        """Generate response using the LLM service."""
        try:
            # Set Google API key directly in the environment
            # This ensures it's available to the LLM service
            os.environ["GOOGLE_API_KEY"] = "AIzaSyDbtwa0HfpRqmFUV3D1vEUIyYHBBHWLy6M"
            
            # Use the LLM service instead of local models
            from app.services.llm_service import LLMService
            llm_service = LLMService()
            
            # Prepare messages for the API call
            messages = [
                {"role": "system", "content": "You are a helpful financial advisor assistant."},
                {"role": "user", "content": prompt}
            ]
            
            # Generate response using API
            logger.info(f"Using {llm_service.provider} API with model {llm_service.model} for chat")
            response = await llm_service.generate_response(messages)
            
            return response
                
        except Exception as e:
            logger.error(f"Error in LLM service: {e}", exc_info=True)
            return "I apologize, but I encountered an error while processing your request. Please try again with a simpler question." 