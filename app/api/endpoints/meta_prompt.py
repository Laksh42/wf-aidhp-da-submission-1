from fastapi import APIRouter, Depends, HTTPException, status, Body
from typing import Dict, Any, Optional
import logging
from datetime import datetime
from bson.objectid import ObjectId

from app.models.user import User
from app.utils.data_loader import DataLoader
from app.utils.data_processor import DataProcessor
from app.utils.prompt_generator import PromptGenerator
from app.api.deps import get_current_user

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/generate", response_model=Dict[str, Any])
async def generate_meta_prompt(
    query: Dict[str, str] = Body(...),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Generate a meta-prompt for the financial advisor chatbot based on user data.
    
    Args:
        query: Dictionary containing the user's query
        current_user: Authenticated user object
        
    Returns:
        Dict with the generated meta-prompt and additional context
    """
    try:
        user_query = query.get("query", "")
        if not user_query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        logger.info(f"Generating meta-prompt for user {current_user.email}")
        
        # Load the user's data
        data_loader = DataLoader()
        user_data = await data_loader.get_user_data(str(current_user.id))
        
        # Process the data to extract insights
        data_processor = DataProcessor()
        
        # Process transaction data
        if "transactions" in user_data:
            transaction_insights = data_processor.extract_transaction_insights(user_data["transactions"])
            user_data["transaction_insights"] = transaction_insights
        
        # Process social media data
        if "social_media" in user_data:
            sentiment_insights = data_processor.extract_sentiment_insights(user_data["social_media"])
            user_data["sentiment_insights"] = sentiment_insights
        
        # Process investment data
        if "investments" in user_data:
            investment_analysis = data_processor.analyze_investment_portfolio(user_data["investments"])
            user_data["investment_analysis"] = investment_analysis
        
        # Generate the meta-prompt
        prompt_generator = PromptGenerator()
        meta_prompt = prompt_generator.generate_meta_prompt(user_data, user_query)
        
        return {
            "meta_prompt": meta_prompt,
            "user_id": str(current_user.id),
            "timestamp": datetime.now().isoformat(),
            "data_points": {
                "demographic": bool(user_data.get("demographic")),
                "account": bool(user_data.get("account")),
                "credit": bool(user_data.get("credit")),
                "investments": bool(user_data.get("investments")),
                "transactions": bool(user_data.get("transactions")),
                "social_media": bool(user_data.get("social_media"))
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating meta-prompt: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating meta-prompt: {str(e)}")

@router.post("/product-recommendations", response_model=Dict[str, Any])
async def generate_product_recommendation_prompt(
    query: Dict[str, str] = Body(...),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Generate a product recommendation meta-prompt based on user data.
    
    Args:
        query: Dictionary containing the user's query
        current_user: Authenticated user object
        
    Returns:
        Dict with the generated recommendation prompt
    """
    try:
        user_query = query.get("query", "")
        if not user_query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        logger.info(f"Generating product recommendation prompt for user {current_user.email}")
        
        # Load the user's data and available products
        data_loader = DataLoader()
        user_data = await data_loader.get_user_data(str(current_user.id))
        products = await data_loader.list_all_products()
        
        # Process investment data if available
        data_processor = DataProcessor()
        if "investments" in user_data:
            investment_analysis = data_processor.analyze_investment_portfolio(user_data["investments"])
            user_data["investment_analysis"] = investment_analysis
        
        # Generate the recommendation prompt
        prompt_generator = PromptGenerator()
        recommendation_prompt = prompt_generator.generate_product_recommendation_prompt(
            user_data, products, user_query
        )
        
        return {
            "meta_prompt": recommendation_prompt,
            "user_id": str(current_user.id),
            "timestamp": datetime.now().isoformat(),
            "products_count": len(products)
        }
        
    except Exception as e:
        logger.error(f"Error generating product recommendation prompt: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating recommendation: {str(e)}")

@router.post("/investment-analysis", response_model=Dict[str, Any])
async def generate_investment_analysis_prompt(
    query: Dict[str, str] = Body(...),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Generate an investment analysis meta-prompt based on user data.
    
    Args:
        query: Dictionary containing the user's query
        current_user: Authenticated user object
        
    Returns:
        Dict with the generated investment analysis prompt
    """
    try:
        user_query = query.get("query", "")
        if not user_query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        logger.info(f"Generating investment analysis prompt for user {current_user.email}")
        
        # Load the user's data
        data_loader = DataLoader()
        user_data = await data_loader.get_user_data(str(current_user.id))
        
        # Process investment data
        data_processor = DataProcessor()
        if "investments" in user_data:
            investment_analysis = data_processor.analyze_investment_portfolio(user_data["investments"])
            user_data["investment_analysis"] = investment_analysis
        else:
            raise HTTPException(status_code=400, detail="No investment data available for analysis")
        
        # Generate the investment analysis prompt
        prompt_generator = PromptGenerator()
        analysis_prompt = prompt_generator.generate_investment_analysis_prompt(
            user_data, user_query
        )
        
        return {
            "meta_prompt": analysis_prompt,
            "user_id": str(current_user.id),
            "timestamp": datetime.now().isoformat(),
            "investment_count": len(user_data.get("investments", []))
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error generating investment analysis prompt: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating analysis: {str(e)}") 