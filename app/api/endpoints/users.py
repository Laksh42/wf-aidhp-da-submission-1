from fastapi import APIRouter, Depends, HTTPException, status, Body
from typing import Dict, Any, List
import logging
from datetime import datetime
from bson.objectid import ObjectId

from app.models.user import User, UserUpdate
from app.database.user_db import get_user_by_id, update_user
from app.api.deps import get_current_user

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/me", response_model=Dict[str, Any])
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get current logged-in user information.
    
    Args:
        current_user: Currently authenticated user
        
    Returns:
        Dict with user details
    """
    try:
        logger.info(f"Retrieved user info for {current_user.email}")
        
        return {
            "user_id": str(current_user.id),
            "email": current_user.email,
            "full_name": current_user.full_name,
            "created_at": current_user.created_at,
            "last_login": current_user.last_login
        }
        
    except Exception as e:
        logger.error(f"Error retrieving user info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving user info: {str(e)}",
        )

@router.put("/me", response_model=Dict[str, Any])
async def update_current_user(
    user_update: UserUpdate = Body(...),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Update current logged-in user information.
    
    Args:
        user_update: User data to update
        current_user: Currently authenticated user
        
    Returns:
        Dict with updated user details
    """
    try:
        # Create updated user with existing data
        updated_data = current_user.dict()
        
        # Update with new data
        if user_update.full_name:
            updated_data["full_name"] = user_update.full_name
            
        # Update the user in database
        updated_user = await update_user(str(current_user.id), updated_data)
        
        logger.info(f"Updated user info for {updated_user.email}")
        
        return {
            "user_id": str(updated_user.id),
            "email": updated_user.email,
            "full_name": updated_user.full_name,
            "created_at": updated_user.created_at,
            "last_login": updated_user.last_login,
            "message": "User information updated successfully"
        }
        
    except Exception as e:
        logger.error(f"Error updating user info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating user info: {str(e)}",
        )

@router.get("/{user_id}", response_model=Dict[str, Any])
async def get_user_by_id_route(
    user_id: str,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get user information by ID (admin only).
    
    Args:
        user_id: ID of the user to retrieve
        current_user: Currently authenticated user (must be admin)
        
    Returns:
        Dict with user details
    """
    try:
        # Check if current user is admin (simplified for demo)
        # In a real app, you would check admin role
        if str(current_user.id) != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this user's information",
            )
        
        user = await get_user_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )
        
        logger.info(f"Retrieved info for user {user.email}")
        
        return {
            "user_id": str(user.id),
            "email": user.email,
            "full_name": user.full_name,
            "created_at": user.created_at,
            "last_login": user.last_login
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error retrieving user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving user: {str(e)}",
        ) 