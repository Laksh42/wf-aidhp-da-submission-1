from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, EmailStr
from bson import ObjectId

class PyObjectId(ObjectId):
    """Custom ObjectId type for Pydantic models."""
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    
    @classmethod
    def validate(cls, v, info=None):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)
    
    @classmethod
    def __get_pydantic_json_schema__(cls, _schema_generator, field_schema):
        field_schema.update(type="string")
        return field_schema

class User(BaseModel):
    id: Optional[PyObjectId] = Field(None, alias="_id")
    user_id: str
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = False
    is_active: Optional[bool] = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    model_config = {
        "populate_by_name": True,
        "arbitrary_types_allowed": True,
        "json_encoders": {ObjectId: str}
    }

class UserCreate(BaseModel):
    """User creation model with password."""
    user_id: str
    password: str
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    
    model_config = {
        "arbitrary_types_allowed": True,
        "json_encoders": {
            ObjectId: str
        },
        "extra": "forbid",
        "populate_by_name": True
    }

class UserUpdate(BaseModel):
    """User update model."""
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None
    password: Optional[str] = None
    
    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "forbid",
        "populate_by_name": True
    }

class UserInDB(User):
    """User model as stored in the database."""
    hashed_password: str

class TokenData(BaseModel):
    """Token data model."""
    user_id: Optional[str] = None
    sub: Optional[str] = None
    
    model_config = {
        "extra": "forbid"
    }

class Token(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str
    user_id: str
    
    model_config = {
        "extra": "forbid"
    }
    
class UserProfile(BaseModel):
    """User profile with basic info."""
    id: str
    user_id: str
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    created_at: datetime
    last_login: Optional[datetime] = None
    
    model_config = {
        "json_encoders": {
            ObjectId: str
        },
        "extra": "forbid",
        "populate_by_name": True
    }

class UserData(BaseModel):
    user_id: str
    demographic_data: Optional[Dict[str, Any]] = None
    account_data: Optional[Dict[str, Any]] = None
    credit_history: Optional[Dict[str, Any]] = None
    investment_data: Optional[Dict[str, Any]] = None
    recent_transactions: Optional[List[Dict[str, Any]]] = None
    
    model_config = {
        "arbitrary_types_allowed": True,
        "json_encoders": {ObjectId: str},
        "json_schema_extra": {
            "examples": [
                {
                    "user_id": "1",
                    "demographic_data": {"age": 30, "occupation": "Engineer"},
                    "account_data": {"balance": 5000},
                    "credit_history": {"score": 750},
                    "investment_data": {"stocks": 10000},
                    "recent_transactions": [{"amount": 100, "type": "debit"}]
                }
            ]
        }
    }

    @classmethod
    def model_validate(cls, obj):
        # Convert any ObjectId to string in nested dictionaries
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, dict):
                    for k, v in value.items():
                        if isinstance(v, ObjectId):
                            value[k] = str(v)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            for k, v in item.items():
                                if isinstance(v, ObjectId):
                                    item[k] = str(v)
        return super().model_validate(obj) 