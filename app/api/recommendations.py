from fastapi import APIRouter, Depends, HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel

from app.database.connection import get_database
from app.database.models import Recommendations, ProductRecommendation, UserInDB
from app.api.auth import get_current_user
from app.models.recommendation_engine import RecommendationEngine
from app.models.user import User
from app.dependencies import get_current_active_user

# Remove the prefix so it doesn't cause double prefixing
router = APIRouter()

class ProductRecommendation(BaseModel):
    """Product recommendation model."""
    name: str
    description: str
    reason: str
    score: float

class RecommendationsResponse(BaseModel):
    """Recommendations response model."""
    products: List[ProductRecommendation]

@router.get("/", response_model=RecommendationsResponse)
async def get_recommendations(
    current_user: User = Depends(get_current_active_user),
    db: AsyncIOMotorDatabase = Depends(get_database)
) -> RecommendationsResponse:
    """
    Get personalized financial product recommendations for the current user.
    Uses the recommendation engine to generate personalized recommendations based on the user's profile.
    """
    try:
        # Initialize the recommendation engine
        recommendation_engine = RecommendationEngine(db)
        
        # Get personalized recommendations for the user
        recommendations = await recommendation_engine.generate_recommendations(current_user.user_id)
        
        # Convert to API response format
        product_recommendations = [
            ProductRecommendation(
                name=rec.name,
                description=rec.description,
                reason=rec.reason,
                score=rec.score
            ) for rec in recommendations
        ]
        
        # Store the recommendations in history
        await db.recommendations.insert_one({
            "user_id": current_user.user_id,
            "recommendations": [rec.dict() for rec in product_recommendations],
            "created_at": datetime.utcnow()
        })
        
        return RecommendationsResponse(products=product_recommendations)
    except Exception as e:
        # Log the error
        print(f"Error generating recommendations: {str(e)}")
        
        # Return fallback recommendations
        fallback_recommendations = [
            ProductRecommendation(
                name="High-Yield Savings Account",
                description="A savings account with competitive interest rates and no monthly fees.",
                reason="Recommended as a safe option for all users",
                score=90.0
            ),
            ProductRecommendation(
                name="Investment Portfolio",
                description="A diversified investment portfolio tailored to your goals.",
                reason="General recommendation for long-term growth",
                score=85.0
            ),
            ProductRecommendation(
                name="Travel Rewards Credit Card",
                description="Earn points on purchases and get travel benefits.",
                reason="Popular choice for most customers",
                score=75.0
            )
        ]
        
        return RecommendationsResponse(products=fallback_recommendations)

@router.get("/test", response_model=RecommendationsResponse)
async def get_test_recommendations() -> RecommendationsResponse:
    """
    Get test recommendations without authentication (for debugging).
    """
    sample_recommendations = [
        ProductRecommendation(
            name="Test Savings Account",
            description="A test savings account for debugging.",
            reason="This is a test recommendation",
            score=95.0
        ),
        ProductRecommendation(
            name="Test Investment",
            description="A test investment recommendation.",
            reason="This is another test recommendation",
            score=88.0
        )
    ]
    
    return RecommendationsResponse(products=sample_recommendations)

@router.get("/history", response_model=List[Recommendations])
async def get_recommendation_history(
    limit: int = 5,
    skip: int = 0,
    current_user: UserInDB = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Get recommendation history for the current user."""
    cursor = db.recommendations.find({"user_id": current_user.user_id})
    cursor.sort("created_at", -1).skip(skip).limit(limit)
    
    recommendations_history = []
    async for rec_data in cursor:
        recommendations_history.append(Recommendations(**rec_data))
    
    return recommendations_history

@router.post("/feedback", status_code=status.HTTP_200_OK)
async def provide_feedback(
    recommendation_id: str,
    product_name: str,
    feedback: str,
    rating: int,
    current_user: UserInDB = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """
    Provide feedback on a specific product recommendation.
    This helps improve future recommendations.
    """
    # Check if recommendation exists
    recommendation_data = await db.recommendations.find_one({"_id": recommendation_id, "user_id": current_user.user_id})
    if not recommendation_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recommendation not found"
        )
    
    # Save feedback to database
    await db.recommendation_feedback.insert_one({
        "user_id": current_user.user_id,
        "recommendation_id": recommendation_id,
        "product_name": product_name,
        "feedback": feedback,
        "rating": rating,
        "created_at": datetime.utcnow()
    })
    
    return {"status": "success", "message": "Feedback recorded successfully"} 