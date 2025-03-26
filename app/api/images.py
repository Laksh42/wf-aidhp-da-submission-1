from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Form, status
from fastapi.responses import JSONResponse
from motor.motor_asyncio import AsyncIOMotorDatabase
from typing import List, Optional, Dict, Any
import logging
import json

from app.database.mongodb import get_database
from app.models.image_analyzer import ImageAnalyzer
from app.api.auth import get_current_user
from app.database.models import User

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/images",
    tags=["images"],
    responses={404: {"description": "Not found"}},
)

@router.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    analysis_type: str = Form("general"),
    current_user: User = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """
    Upload and analyze a financial document image.
    
    - **file**: The image file to upload
    - **analysis_type**: Type of analysis to perform (general, receipt, statement, document)
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image"
        )
    
    try:
        # Read file content
        contents = await file.read()
        
        # Initialize image analyzer
        analyzer = ImageAnalyzer()
        
        # Save the uploaded image
        file_path = await analyzer.save_uploaded_image(contents, file.filename)
        
        # Analyze the image
        analysis_result = await analyzer.analyze_image(contents, analysis_type)
        
        # Save the analysis to the database
        analysis_doc = {
            "user_id": str(current_user.id),
            "analysis_type": analysis_type,
            "file_name": file.filename,
            "file_path": file_path,
            "result": analysis_result
        }
        
        result = await db.image_analyses.insert_one(analysis_doc)
        
        # Add the analysis ID to the result
        analysis_result["analysis_id"] = str(result.inserted_id)
        
        return analysis_result
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing image: {str(e)}"
        )
    
@router.get("/analyses")
async def get_analyses(
    limit: int = 10,
    skip: int = 0,
    current_user: User = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """
    Get a list of image analyses for the current user.
    
    - **limit**: Maximum number of analyses to return
    - **skip**: Number of analyses to skip for pagination
    """
    try:
        # Query the database for analyses by this user
        cursor = db.image_analyses.find(
            {"user_id": str(current_user.id)}
        ).sort("_id", -1).skip(skip).limit(limit)
        
        # Convert to list
        analyses = []
        async for doc in cursor:
            # Convert ObjectId to string for JSON serialization
            doc["_id"] = str(doc["_id"])
            
            # Create a simplified view for the list
            analyses.append({
                "analysis_id": doc["_id"],
                "analysis_type": doc["analysis_type"],
                "file_name": doc["file_name"],
                "created_at": doc.get("created_at", None),
                "summary": _get_analysis_summary(doc["result"])
            })
        
        # Return the list
        return analyses
        
    except Exception as e:
        logger.error(f"Error retrieving analyses: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving analyses: {str(e)}"
        )

def _get_analysis_summary(result: Dict[str, Any]) -> str:
    """Generate a short summary of the analysis result."""
    if not result or not isinstance(result, dict):
        return "No analysis available"
    
    try:
        # For receipts, show merchant and total
        if result.get("analysis_type") == "receipt":
            data = result.get("structured_data", {})
            merchant = data.get("merchant", data.get("merchant_name", "Unknown merchant"))
            total = data.get("total_amount", data.get("total", "Unknown amount"))
            return f"Receipt from {merchant} for {total}"
            
        # For statements, show institution and period
        elif result.get("analysis_type") == "statement":
            data = result.get("structured_data", {})
            institution = data.get("institution", data.get("institution_name", "Unknown institution"))
            period = data.get("statement_period", data.get("period", "Unknown period"))
            return f"Statement from {institution} for {period}"
            
        # For documents, show document type
        elif result.get("analysis_type") == "document":
            data = result.get("structured_data", {})
            doc_type = data.get("document_type", "Financial document")
            return f"{doc_type}"
            
        # For general analysis, use the first key-value pair
        else:
            data = result.get("structured_data", {})
            if data:
                first_key = next(iter(data))
                return f"{first_key}: {data[first_key]}"
            else:
                return "General financial document analysis"
                
    except Exception as e:
        logger.error(f"Error generating analysis summary: {str(e)}")
        return "Financial document analysis"

@router.get("/analyses/{analysis_id}")
async def get_analysis(
    analysis_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """
    Get a specific image analysis by ID.
    
    - **analysis_id**: ID of the analysis to retrieve
    """
    try:
        # Query the database for the specific analysis
        analysis = await db.image_analyses.find_one({
            "_id": analysis_id,
            "user_id": str(current_user.id)
        })
        
        if not analysis:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Analysis not found"
            )
        
        # Convert ObjectId to string for JSON serialization
        analysis["_id"] = str(analysis["_id"])
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error retrieving analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving analysis: {str(e)}"
        )

@router.delete("/analyses/{analysis_id}")
async def delete_analysis(
    analysis_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """
    Delete a specific image analysis by ID.
    
    - **analysis_id**: ID of the analysis to delete
    """
    try:
        # Delete the analysis
        result = await db.image_analyses.delete_one({
            "_id": analysis_id,
            "user_id": str(current_user.id)
        })
        
        if result.deleted_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Analysis not found or you don't have permission to delete it"
            )
        
        return {"message": "Analysis deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting analysis: {str(e)}"
        ) 