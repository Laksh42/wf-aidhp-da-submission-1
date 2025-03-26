import os
import shutil
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, BackgroundTasks
from typing import List, Optional, Any
from datetime import datetime

from app.models.user import User
from app.models.document import Document, DocumentCreate, DocumentUpdate, DocumentSummary, DocumentType, ProcessingStatus
from app.repository.document_repository import DocumentRepository
from app.dependencies import get_current_active_user, get_document_repository
from app.services.document_processor import process_document

router = APIRouter()

# Create upload directory if it doesn't exist
UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload", response_model=Document, status_code=status.HTTP_201_CREATED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    document_type: DocumentType = DocumentType.OTHER,
    current_user: User = Depends(get_current_active_user),
    doc_repo: DocumentRepository = Depends(get_document_repository)
) -> Any:
    """
    Upload a financial document for processing.
    """
    # Validate file
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file name"
        )
    
    # Create user directory if it doesn't exist
    user_dir = os.path.join(UPLOAD_DIR, str(current_user.id))
    os.makedirs(user_dir, exist_ok=True)
    
    # Save file
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_filename = f"{timestamp}_{os.path.basename(file.filename)}"
    file_path = os.path.join(user_dir, safe_filename)
    
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # Get file size
    file_size = os.path.getsize(file_path)
    
    # Create document record
    document_data = DocumentCreate(
        user_id=str(current_user.id),
        file_name=file.filename,
        file_path=file_path,
        document_type=document_type,
        mime_type=file.content_type or "application/octet-stream",
        file_size=file_size
    )
    
    document = await doc_repo.create_document(document_data)
    
    # Start background processing
    background_tasks.add_task(
        process_document,
        document_id=str(document.id),
        file_path=file_path
    )
    
    return document

@router.get("/documents", response_model=List[DocumentSummary])
async def list_documents(
    document_type: Optional[DocumentType] = None,
    skip: int = 0,
    limit: int = 20,
    current_user: User = Depends(get_current_active_user),
    doc_repo: DocumentRepository = Depends(get_document_repository)
) -> Any:
    """
    List documents for the current user.
    """
    documents = await doc_repo.list_user_documents(
        str(current_user.id),
        document_type.value if document_type else None,
        skip,
        limit
    )
    return documents

@router.get("/documents/{document_id}", response_model=Document)
async def get_document(
    document_id: str,
    current_user: User = Depends(get_current_active_user),
    doc_repo: DocumentRepository = Depends(get_document_repository)
) -> Any:
    """
    Get a specific document.
    """
    document = await doc_repo.get_document(document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # Check if user owns the document
    if document.user_id != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this document"
        )
    
    return document

@router.put("/documents/{document_id}", response_model=Document)
async def update_document(
    document_id: str,
    data: DocumentUpdate,
    current_user: User = Depends(get_current_active_user),
    doc_repo: DocumentRepository = Depends(get_document_repository)
) -> Any:
    """
    Update a document's metadata.
    """
    document = await doc_repo.get_document(document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # Check if user owns the document
    if document.user_id != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this document"
        )
    
    updated_document = await doc_repo.update_document(document_id, data)
    return updated_document

@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: str,
    current_user: User = Depends(get_current_active_user),
    doc_repo: DocumentRepository = Depends(get_document_repository)
) -> None:
    """
    Delete a document.
    """
    document = await doc_repo.get_document(document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # Check if user owns the document
    if document.user_id != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this document"
        )
    
    # Delete file if it exists
    if os.path.exists(document.file_path):
        try:
            os.remove(document.file_path)
        except OSError:
            # Log error but continue
            pass
    
    # Delete document record
    deleted = await doc_repo.delete_document(document_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete document"
        )

@router.get("/documents/{document_id}/analyses", response_model=List[Document])
async def get_document_analyses(
    document_id: str,
    current_user: User = Depends(get_current_active_user),
    doc_repo: DocumentRepository = Depends(get_document_repository)
) -> Any:
    """
    Get analyses for a document.
    """
    document = await doc_repo.get_document(document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # Check if user owns the document
    if document.user_id != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this document"
        )
    
    analyses = await doc_repo.get_document_analyses(document_id)
    return analyses 