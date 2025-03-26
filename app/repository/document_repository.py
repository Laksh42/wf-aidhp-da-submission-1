from typing import List, Optional, Dict, Any
from bson import ObjectId
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.models.document import Document, DocumentCreate, DocumentUpdate, DocumentSummary, DocumentAnalysis, ProcessingStatus


class DocumentRepository:
    """Repository for document-related database operations."""
    
    def __init__(self, database: AsyncIOMotorDatabase):
        """Initialize with database connection."""
        self.db = database
        self.documents_collection = database.documents
        self.analyses_collection = database.document_analyses
    
    async def create_indexes(self):
        """Create necessary indexes."""
        await self.documents_collection.create_index("user_id")
        await self.documents_collection.create_index("upload_date")
        await self.documents_collection.create_index([("document_type", 1), ("user_id", 1)])
        await self.analyses_collection.create_index("document_id")
        await self.analyses_collection.create_index("created_at")
    
    # Document methods
    
    async def create_document(self, data: DocumentCreate) -> Document:
        """Create a new document record."""
        now = datetime.utcnow()
        document = Document(
            _id=ObjectId(),
            user_id=data.user_id,
            file_name=data.file_name,
            file_path=data.file_path,
            document_type=data.document_type,
            mime_type=data.mime_type,
            file_size=data.file_size,
            upload_date=now,
            processing_status=ProcessingStatus.PENDING,
            extracted_data={},
            metadata=data.metadata or {}
        )
        
        await self.documents_collection.insert_one(document.dict(by_alias=True))
        return document
    
    async def get_document(self, document_id: str) -> Optional[Document]:
        """Get a document by ID."""
        if not ObjectId.is_valid(document_id):
            return None
            
        result = await self.documents_collection.find_one({"_id": ObjectId(document_id)})
        if result:
            return Document(**result)
        return None
    
    async def update_document(self, document_id: str, data: DocumentUpdate) -> Optional[Document]:
        """Update a document."""
        if not ObjectId.is_valid(document_id):
            return None
            
        document = await self.get_document(document_id)
        if not document:
            return None
            
        update_data = data.dict(exclude_unset=True)
        if update_data:
            await self.documents_collection.update_one(
                {"_id": ObjectId(document_id)},
                {"$set": update_data}
            )
            
        return await self.get_document(document_id)
    
    async def update_processing_status(self, document_id: str, status: ProcessingStatus, extracted_data: Optional[Dict[str, Any]] = None) -> Optional[Document]:
        """Update document processing status and extracted data."""
        if not ObjectId.is_valid(document_id):
            return None
            
        update_data = {"processing_status": status}
        if extracted_data is not None:
            update_data["extracted_data"] = extracted_data
            
        await self.documents_collection.update_one(
            {"_id": ObjectId(document_id)},
            {"$set": update_data}
        )
        
        return await self.get_document(document_id)
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and its analyses."""
        if not ObjectId.is_valid(document_id):
            return False
            
        # Delete the document
        result = await self.documents_collection.delete_one({"_id": ObjectId(document_id)})
        
        # Delete all analyses for this document
        await self.analyses_collection.delete_many({"document_id": document_id})
        
        return result.deleted_count > 0
    
    async def list_user_documents(self, user_id: str, document_type: Optional[str] = None, skip: int = 0, limit: int = 20) -> List[DocumentSummary]:
        """List documents for a user with optional filtering."""
        query = {"user_id": user_id}
        if document_type:
            query["document_type"] = document_type
            
        cursor = self.documents_collection.find(query).sort("upload_date", -1).skip(skip).limit(limit)
        documents = await cursor.to_list(length=limit)
        
        return [DocumentSummary(
            id=doc["_id"],
            file_name=doc["file_name"],
            document_type=doc["document_type"],
            upload_date=doc["upload_date"],
            processing_status=doc["processing_status"]
        ) for doc in documents]
    
    async def count_user_documents(self, user_id: str, document_type: Optional[str] = None) -> int:
        """Count documents for a user with optional filtering."""
        query = {"user_id": user_id}
        if document_type:
            query["document_type"] = document_type
            
        return await self.documents_collection.count_documents(query)
    
    # Document analysis methods
    
    async def create_analysis(self, document_id: str, analysis_type: str, insights: Dict[str, Any], recommendations: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> DocumentAnalysis:
        """Create a new document analysis."""
        now = datetime.utcnow()
        analysis = DocumentAnalysis(
            _id=ObjectId(),
            document_id=document_id,
            analysis_type=analysis_type,
            insights=insights,
            recommendations=recommendations,
            created_at=now,
            metadata=metadata or {}
        )
        
        await self.analyses_collection.insert_one(analysis.dict(by_alias=True))
        return analysis
    
    async def get_analysis(self, analysis_id: str) -> Optional[DocumentAnalysis]:
        """Get an analysis by ID."""
        if not ObjectId.is_valid(analysis_id):
            return None
            
        result = await self.analyses_collection.find_one({"_id": ObjectId(analysis_id)})
        if result:
            return DocumentAnalysis(**result)
        return None
    
    async def get_document_analyses(self, document_id: str, analysis_type: Optional[str] = None) -> List[DocumentAnalysis]:
        """Get analyses for a document with optional filtering."""
        query = {"document_id": document_id}
        if analysis_type:
            query["analysis_type"] = analysis_type
            
        cursor = self.analyses_collection.find(query).sort("created_at", -1)
        analyses = await cursor.to_list(length=100)
        
        return [DocumentAnalysis(**analysis) for analysis in analyses]
    
    async def delete_analysis(self, analysis_id: str) -> bool:
        """Delete an analysis."""
        if not ObjectId.is_valid(analysis_id):
            return False
            
        result = await self.analyses_collection.delete_one({"_id": ObjectId(analysis_id)})
        return result.deleted_count > 0 