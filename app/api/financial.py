from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import List, Optional, Dict, Any
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.database import get_database
from app.repository.financial_repository import FinancialRepository
from app.models.financial import Product, Investment, InvestmentCreate, InvestmentUpdate
from app.auth import get_current_user
from app.models.user import User

router = APIRouter(
    prefix="/financial",
    tags=["financial"],
    responses={404: {"description": "Not found"}},
)

async def get_financial_repository(db: AsyncIOMotorDatabase = Depends(get_database)) -> FinancialRepository:
    """Get financial repository instance."""
    return FinancialRepository(db)

@router.get("/products", response_model=List[Product])
async def list_products(
    category: Optional[str] = None,
    risk_level: Optional[str] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    repo: FinancialRepository = Depends(get_financial_repository)
):
    """
    Get a list of available financial products.
    
    Optionally filter by category and risk level.
    """
    return await repo.list_products(category, risk_level, skip, limit)

@router.get("/products/{product_id}", response_model=Product)
async def get_product(
    product_id: str,
    repo: FinancialRepository = Depends(get_financial_repository)
):
    """Get details of a specific financial product."""
    product = await repo.get_product(product_id)
    if not product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Product not found"
        )
    return product

@router.get("/investments", response_model=List[Investment])
async def get_user_investments(
    investment_type: Optional[str] = None,
    user: User = Depends(get_current_user),
    repo: FinancialRepository = Depends(get_financial_repository)
):
    """
    Get current user's investments.
    
    Optionally filter by investment type.
    """
    return await repo.get_user_investments(str(user.id), investment_type)

@router.post("/investments", response_model=Investment, status_code=status.HTTP_201_CREATED)
async def create_investment(
    investment_data: InvestmentCreate,
    current_user: User = Depends(get_current_user),
    financial_repo: FinancialRepository = Depends(get_financial_repository)
) -> Any:
    """
    Create a new investment for the current user.
    """
    # Override user_id with current user's ID for security
    investment_data.user_id = str(current_user.id)
    
    investment = await financial_repo.create_investment(investment_data)
    return investment

@router.put("/investments/{investment_id}", response_model=Investment)
async def update_investment(
    investment_id: str,
    investment_data: InvestmentUpdate,
    current_user: User = Depends(get_current_user),
    financial_repo: FinancialRepository = Depends(get_financial_repository)
) -> Any:
    """
    Update an existing investment.
    """
    # Get the investment
    investment = await financial_repo.get_investment(investment_id)
    if not investment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Investment not found"
        )
    
    # Check if user owns the investment
    if investment.user_id != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this investment"
        )
    
    # Update the investment
    updated_investment = await financial_repo.update_investment(investment_id, investment_data)
    return updated_investment

@router.get("/investments/summary", response_model=Dict[str, Any])
async def get_investment_summary(
    user: User = Depends(get_current_user),
    repo: FinancialRepository = Depends(get_financial_repository)
):
    """
    Get a summary of the current user's investments.
    
    Returns total amounts, performance, and breakdowns by type.
    """
    return await repo.get_investment_summary(str(user.id))

@router.get("/transaction-summary", response_model=Dict[str, Any])
async def get_transaction_summary(
    months: int = 3,
    current_user: User = Depends(get_current_user),
    financial_repo: FinancialRepository = Depends(get_financial_repository)
) -> Any:
    """
    Get a summary of the current user's transactions.
    """
    summary = await financial_repo.get_transaction_summary(str(current_user.id), months)
    return summary

@router.get("/account", response_model=Dict[str, Any])
async def get_user_account(
    current_user: User = Depends(get_current_user),
    financial_repo: FinancialRepository = Depends(get_financial_repository)
) -> Any:
    """
    Get the current user's account information.
    """
    account = await financial_repo.get_user_account(str(current_user.id))
    if not account:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Account not found"
        )
    return account

@router.get("/credit-history", response_model=Dict[str, Any])
async def get_user_credit_history(
    current_user: User = Depends(get_current_user),
    financial_repo: FinancialRepository = Depends(get_financial_repository)
) -> Any:
    """
    Get the current user's credit history.
    """
    credit_history = await financial_repo.get_user_credit_history(str(current_user.id))
    if not credit_history:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Credit history not found"
        )
    return credit_history

@router.get("/demographics", response_model=Dict[str, Any])
async def get_user_demographics(
    current_user: User = Depends(get_current_user),
    financial_repo: FinancialRepository = Depends(get_financial_repository)
) -> Any:
    """
    Get the current user's demographic information.
    """
    demographics = await financial_repo.get_user_demographics(str(current_user.id))
    if not demographics:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Demographics not found"
        )
    return demographics

@router.get("/financial-profile", response_model=Dict[str, Any])
async def get_financial_profile(
    current_user: User = Depends(get_current_user),
    financial_repo: FinancialRepository = Depends(get_financial_repository)
) -> Any:
    """
    Get a complete financial profile for the current user.
    """
    user_id = str(current_user.id)
    
    # Get all user financial data
    demographics = await financial_repo.get_user_demographics(user_id)
    account = await financial_repo.get_user_account(user_id)
    credit_history = await financial_repo.get_user_credit_history(user_id)
    investment_summary = await financial_repo.get_investment_summary(user_id)
    transaction_summary = await financial_repo.get_transaction_summary(user_id)
    
    # Combine into a single profile
    profile = {
        "user_id": user_id,
        "demographics": demographics.dict() if demographics else None,
        "account": account.dict() if account else None,
        "credit_history": credit_history.dict() if credit_history else None,
        "investments": investment_summary,
        "transactions": transaction_summary
    }
    
    return profile 