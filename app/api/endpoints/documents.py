from fastapi import APIRouter, Depends, HTTPException, status, Body, UploadFile, File, Form
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import io

from app.models.user import User
from app.api.deps import get_current_user
from app.multimodal.document_processor import DocumentProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/upload", response_model=Dict[str, Any])
async def upload_document(
    document: UploadFile = File(...),
    document_type: str = Form(...),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Upload and process a financial document.
    
    Args:
        document: The file to upload
        document_type: Type of document (bank_statement, investment_report, tax_document, receipt)
        current_user: Currently authenticated user
        
    Returns:
        Dict with document processing results
    """
    try:
        # Validate document type
        valid_types = ["bank_statement", "investment_report", "tax_document", "receipt"]
        if document_type not in valid_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid document type. Must be one of: {', '.join(valid_types)}",
            )
        
        # Read file content
        content = await document.read()
        if len(content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file",
            )
        
        logger.info(f"Processing {document_type} document upload for user {current_user.email}")
        
        # Process the document
        processor = DocumentProcessor()
        file_path = processor.save_uploaded_file(
            content, document.filename, str(current_user.id)
        )
        
        # Extract information from the document
        extracted_data = processor.process_financial_document(file_path, document_type)
        
        # Generate a human-readable summary
        summary = processor.generate_document_summary(extracted_data)
        
        return {
            "file_name": document.filename,
            "document_type": document_type,
            "file_path": file_path,
            "processed_at": datetime.now().isoformat(),
            "user_id": str(current_user.id),
            "extracted_data": extracted_data,
            "summary": summary
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing document: {str(e)}",
        )

@router.post("/analyze", response_model=Dict[str, Any])
async def analyze_document(
    document_data: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Analyze an already uploaded document with financial insights.
    
    Args:
        document_data: Dictionary with document path and type
        current_user: Currently authenticated user
        
    Returns:
        Dict with analysis results
    """
    try:
        file_path = document_data.get("file_path")
        document_type = document_data.get("document_type")
        
        if not file_path or not document_type:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Both file_path and document_type are required",
            )
        
        logger.info(f"Analyzing {document_type} document for user {current_user.email}")
        
        # This is a placeholder where you would add more sophisticated analysis
        # based on the document type. For now, we'll return some example analysis.
        
        analysis = {
            "file_path": file_path,
            "document_type": document_type,
            "analyzed_at": datetime.now().isoformat(),
            "user_id": str(current_user.id),
        }
        
        # Add different analysis based on document type
        if document_type == "bank_statement":
            analysis["insights"] = [
                "Your spending increased by 15% compared to last month",
                "Consider reducing grocery expenses which are 25% higher than average",
                "You have 3 recurring subscriptions totaling $45 monthly"
            ]
            analysis["recommendations"] = [
                "Review your subscription services to eliminate unused ones",
                "Set up automatic savings transfers based on your cash flow pattern"
            ]
            
        elif document_type == "investment_report":
            analysis["insights"] = [
                "Your portfolio has a 60/40 stock to bond ratio",
                "Your annual return of 7.2% is slightly below the market average",
                "Your portfolio has a relatively high expense ratio of 0.8%"
            ]
            analysis["recommendations"] = [
                "Consider low-cost index funds to reduce expense ratios",
                "Rebalance your portfolio to maintain your target asset allocation"
            ]
            
        elif document_type == "tax_document":
            analysis["insights"] = [
                "You're in the 22% federal tax bracket",
                "You didn't max out your retirement contributions last year",
                "You may be eligible for additional deductions"
            ]
            analysis["recommendations"] = [
                "Increase 401(k) contributions to reduce taxable income",
                "Consider establishing an HSA if you have a high-deductible health plan"
            ]
            
        elif document_type == "receipt":
            analysis["insights"] = [
                "This purchase is categorized as discretionary spending",
                "Similar purchases account for 12% of your monthly expenses"
            ]
            analysis["recommendations"] = [
                "Track these expenses in a dedicated budget category",
                "Consider using a rewards credit card for these purchases"
            ]
        
        return analysis
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error analyzing document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing document: {str(e)}",
        ) 