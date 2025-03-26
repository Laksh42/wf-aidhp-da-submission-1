from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from bson import ObjectId
from enum import Enum

from app.models.user import PyObjectId

class DocumentType(str, Enum):
    """Document type enum."""
    BANK_STATEMENT = "bank_statement"
    INVESTMENT_REPORT = "investment_report"
    TAX_DOCUMENT = "tax_document"
    RECEIPT = "receipt"
    OTHER = "other"

class ProcessingStatus(str, Enum):
    """Document processing status enum."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class Document(BaseModel):
    """Document model for financial document uploads."""
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    user_id: str
    file_name: str
    file_path: str
    document_type: DocumentType
    mime_type: str
    file_size: int
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    extracted_data: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {
            ObjectId: str
        }

class DocumentCreate(BaseModel):
    """Document creation model."""
    user_id: str
    file_name: str
    file_path: str
    document_type: DocumentType
    mime_type: str
    file_size: int
    metadata: Optional[Dict[str, Any]] = None

class DocumentUpdate(BaseModel):
    """Document update model."""
    document_type: Optional[DocumentType] = None
    processing_status: Optional[ProcessingStatus] = None
    extracted_data: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

class DocumentSummary(BaseModel):
    """Document summary model."""
    id: str
    file_name: str
    document_type: DocumentType
    uploaded_at: datetime
    processing_status: ProcessingStatus
    
    class Config:
        json_encoders = {
            ObjectId: str
        }

class DocumentAnalysis(BaseModel):
    """Document analysis model."""
    document_id: str
    analysis_type: str
    insights: List[str]
    recommendations: List[str]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            ObjectId: str
        } 