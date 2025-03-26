from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, date
from bson import ObjectId
from enum import Enum

from app.models.user import PyObjectId

class InvestmentType(str, Enum):
    """Investment type enum."""
    STOCKS = "stocks"
    BONDS = "bonds"
    ETFS = "etfs"
    MUTUAL_FUNDS = "mutual_funds"
    REAL_ESTATE = "real_estate"
    CRYPTOCURRENCY = "cryptocurrency"
    OTHER = "other"

class RiskLevel(str, Enum):
    """Risk level enum."""
    LOW = "low"
    MEDIUM_LOW = "medium_low"
    MEDIUM = "medium"
    MEDIUM_HIGH = "medium_high"
    HIGH = "high"

class Product(BaseModel):
    """Financial product model."""
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    name: str
    category: str
    interest_rate: float
    term_years: int
    minimum_investment: float
    description: str
    risk_level: str
    suitable_for: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {
            ObjectId: str
        }

class ProductCreate(BaseModel):
    """Financial product creation model."""
    name: str
    category: str
    interest_rate: float
    term_years: int
    minimum_investment: float
    description: str
    risk_level: str
    suitable_for: str
    metadata: Optional[Dict[str, Any]] = None

class Investment(BaseModel):
    """Investment model."""
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    user_id: str
    investment_id: int
    investment_type: InvestmentType
    amount: float
    current_value: float
    start_date: date
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {
            ObjectId: str,
            date: lambda d: d.isoformat()
        }

class InvestmentCreate(BaseModel):
    """Investment creation model."""
    user_id: str
    investment_id: int
    investment_type: InvestmentType
    amount: float
    current_value: float
    start_date: date
    metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        json_encoders = {
            date: lambda d: d.isoformat()
        }

class InvestmentUpdate(BaseModel):
    """Investment update model."""
    investment_type: Optional[InvestmentType] = None
    amount: Optional[float] = None
    current_value: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class Transaction(BaseModel):
    """Transaction model."""
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    user_id: str
    transaction_id: int
    date: date
    amount: float
    merchant: str
    category: str
    transaction_type: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {
            ObjectId: str,
            date: lambda d: d.isoformat()
        }

class Account(BaseModel):
    """Account model."""
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    user_id: str
    account_type: str
    account_balance: float
    savings_balance: float
    account_opening_date: date
    checking_account_number: str
    savings_account_number: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {
            ObjectId: str,
            date: lambda d: d.isoformat()
        }

class CreditHistory(BaseModel):
    """Credit history model."""
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    user_id: str
    credit_score: int
    outstanding_debt: float
    credit_utilization: int
    payment_history: str
    credit_age_years: int
    recent_inquiries: int
    delinquencies: int
    total_accounts: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {
            ObjectId: str
        }

class Demographic(BaseModel):
    """Demographic model."""
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    user_id: str
    age: int
    gender: str
    occupation: str
    annual_income: float
    education_level: str
    city: str
    state: str
    marital_status: str
    dependents: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {
            ObjectId: str
        } 