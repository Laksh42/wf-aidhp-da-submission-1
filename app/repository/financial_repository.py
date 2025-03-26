from typing import List, Optional, Dict, Any
from bson import ObjectId
from datetime import date, datetime
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.models.financial import (
    Product, ProductCreate, Investment, InvestmentCreate, InvestmentUpdate,
    Transaction, Account, CreditHistory, Demographic
)


class FinancialRepository:
    """Repository for financial data operations."""
    
    def __init__(self, database: AsyncIOMotorDatabase):
        """Initialize with database connection."""
        self.db = database
        self.products_collection = database.products
        self.investments_collection = database.investment_data
        self.transactions_collection = database.transaction_data
        self.accounts_collection = database.account_data
        self.credit_history_collection = database.credit_history
        self.demographics_collection = database.demographic_data
    
    async def create_indexes(self):
        """Create necessary indexes."""
        # Products indexes
        await self.products_collection.create_index("name")
        await self.products_collection.create_index("category")
        await self.products_collection.create_index("risk_level")
        
        # Investments indexes
        await self.investments_collection.create_index("user_id")
        await self.investments_collection.create_index("investment_type")
        await self.investments_collection.create_index([("user_id", 1), ("investment_type", 1)])
        
        # Transactions indexes
        await self.transactions_collection.create_index("user_id")
        await self.transactions_collection.create_index("date")
        await self.transactions_collection.create_index("category")
        await self.transactions_collection.create_index([("user_id", 1), ("date", -1)])
        
        # Accounts indexes
        await self.accounts_collection.create_index("user_id", unique=True)
        
        # Credit history indexes
        await self.credit_history_collection.create_index("user_id", unique=True)
        
        # Demographics indexes
        await self.demographics_collection.create_index("user_id", unique=True)
    
    # Product methods
    
    async def create_product(self, data: ProductCreate) -> Product:
        """Create a new financial product."""
        product = Product(
            _id=ObjectId(),
            name=data.name,
            category=data.category,
            interest_rate=data.interest_rate,
            term_years=data.term_years,
            minimum_investment=data.minimum_investment,
            description=data.description,
            risk_level=data.risk_level,
            suitable_for=data.suitable_for,
            metadata=data.metadata or {}
        )
        
        await self.products_collection.insert_one(product.dict(by_alias=True))
        return product
    
    async def get_product(self, product_id: str) -> Optional[Product]:
        """Get a product by ID."""
        if not ObjectId.is_valid(product_id):
            return None
            
        result = await self.products_collection.find_one({"_id": ObjectId(product_id)})
        if result:
            return Product(**result)
        return None
    
    async def list_products(self, category: Optional[str] = None, risk_level: Optional[str] = None, 
                          skip: int = 0, limit: int = 20) -> List[Product]:
        """List products with optional filtering."""
        query = {}
        if category:
            query["category"] = category
        if risk_level:
            query["risk_level"] = risk_level
            
        cursor = self.products_collection.find(query).skip(skip).limit(limit)
        products = await cursor.to_list(length=limit)
        
        return [Product(**product) for product in products]
    
    # Investment methods
    
    async def create_investment(self, data: InvestmentCreate) -> Investment:
        """Create a new investment."""
        investment = Investment(
            _id=ObjectId(),
            user_id=data.user_id,
            investment_id=data.investment_id,
            investment_type=data.investment_type,
            amount=data.amount,
            current_value=data.current_value,
            start_date=data.start_date,
            metadata=data.metadata or {}
        )
        
        await self.investments_collection.insert_one(investment.dict(by_alias=True))
        return investment
    
    async def get_investment(self, investment_id: str) -> Optional[Investment]:
        """Get an investment by ID."""
        if not ObjectId.is_valid(investment_id):
            return None
            
        result = await self.investments_collection.find_one({"_id": ObjectId(investment_id)})
        if result:
            return Investment(**result)
        return None
    
    async def update_investment(self, investment_id: str, data: InvestmentUpdate) -> Optional[Investment]:
        """Update an investment."""
        if not ObjectId.is_valid(investment_id):
            return None
            
        investment = await self.get_investment(investment_id)
        if not investment:
            return None
            
        update_data = data.dict(exclude_unset=True)
        if update_data:
            await self.investments_collection.update_one(
                {"_id": ObjectId(investment_id)},
                {"$set": update_data}
            )
            
        return await self.get_investment(investment_id)
    
    async def get_user_investments(self, user_id: str, investment_type: Optional[str] = None) -> List[Investment]:
        """Get investments for a user with optional filtering."""
        query = {"user_id": user_id}
        if investment_type:
            query["investment_type"] = investment_type
            
        cursor = self.investments_collection.find(query)
        investments = await cursor.to_list(length=100)
        
        return [Investment(**investment) for investment in investments]
    
    async def get_investment_summary(self, user_id: str) -> Dict[str, Any]:
        """Get investment summary for a user."""
        investments = await self.get_user_investments(user_id)
        
        if not investments:
            return {
                "total_invested": 0,
                "total_current_value": 0,
                "total_gain_loss": 0,
                "gain_loss_percentage": 0,
                "investment_types": {}
            }
        
        total_invested = sum(inv.amount for inv in investments)
        total_current = sum(inv.current_value for inv in investments)
        
        # Group by investment type
        types = {}
        for inv in investments:
            inv_type = inv.investment_type.value
            if inv_type not in types:
                types[inv_type] = {"amount": 0, "current_value": 0}
            
            types[inv_type]["amount"] += inv.amount
            types[inv_type]["current_value"] += inv.current_value
        
        # Calculate percentages for each type
        for t in types:
            types[t]["percentage"] = round((types[t]["amount"] / total_invested) * 100, 2)
            gain_loss = types[t]["current_value"] - types[t]["amount"]
            types[t]["gain_loss"] = gain_loss
            types[t]["gain_loss_percentage"] = round((gain_loss / types[t]["amount"]) * 100, 2) if types[t]["amount"] > 0 else 0
        
        return {
            "total_invested": total_invested,
            "total_current_value": total_current,
            "total_gain_loss": total_current - total_invested,
            "gain_loss_percentage": round(((total_current - total_invested) / total_invested) * 100, 2) if total_invested > 0 else 0,
            "investment_types": types
        }
    
    # Transaction methods
    
    async def get_user_transactions(self, user_id: str, start_date: Optional[date] = None, 
                                  end_date: Optional[date] = None, category: Optional[str] = None,
                                  limit: int = 100) -> List[Transaction]:
        """Get transactions for a user with optional filtering."""
        query = {"user_id": user_id}
        
        if start_date or end_date:
            date_query = {}
            if start_date:
                date_query["$gte"] = start_date
            if end_date:
                date_query["$lte"] = end_date
            query["date"] = date_query
            
        if category:
            query["category"] = category
            
        cursor = self.transactions_collection.find(query).sort("date", -1).limit(limit)
        transactions = await cursor.to_list(length=limit)
        
        return [Transaction(**tx) for tx in transactions]
    
    async def get_transaction_summary(self, user_id: str, months: int = 3) -> Dict[str, Any]:
        """Get transaction summary for a user for the last X months."""
        from datetime import timedelta
        
        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=30 * months)
        
        transactions = await self.get_user_transactions(user_id, start_date, end_date)
        
        if not transactions:
            return {
                "total_spending": 0,
                "average_monthly": 0,
                "categories": {},
                "largest_transaction": None
            }
        
        # Calculate totals and find largest transaction
        total_spending = 0
        largest_tx = None
        largest_amount = 0
        categories = {}
        
        for tx in transactions:
            if tx.amount > 0:  # Only count outgoing transactions (positive amounts)
                total_spending += tx.amount
                
                if tx.category not in categories:
                    categories[tx.category] = 0
                categories[tx.category] += tx.amount
                
                if tx.amount > largest_amount:
                    largest_amount = tx.amount
                    largest_tx = tx
        
        # Convert categories to percentages
        for cat in categories:
            categories[cat] = {
                "amount": categories[cat],
                "percentage": round((categories[cat] / total_spending) * 100, 2) if total_spending > 0 else 0
            }
        
        return {
            "total_spending": total_spending,
            "average_monthly": round(total_spending / months, 2),
            "categories": categories,
            "largest_transaction": {
                "amount": largest_tx.amount,
                "merchant": largest_tx.merchant,
                "category": largest_tx.category,
                "date": largest_tx.date.isoformat()
            } if largest_tx else None
        }
    
    # Account methods
    
    async def get_user_account(self, user_id: str) -> Optional[Account]:
        """Get account information for a user."""
        result = await self.accounts_collection.find_one({"user_id": user_id})
        if result:
            return Account(**result)
        return None
    
    # Credit history methods
    
    async def get_user_credit_history(self, user_id: str) -> Optional[CreditHistory]:
        """Get credit history for a user."""
        result = await self.credit_history_collection.find_one({"user_id": user_id})
        if result:
            return CreditHistory(**result)
        return None
    
    # Demographics methods
    
    async def get_user_demographics(self, user_id: str) -> Optional[Demographic]:
        """Get demographic information for a user."""
        result = await self.demographics_collection.find_one({"user_id": user_id})
        if result:
            return Demographic(**result)
        return None
    
    # Data loading
    
    async def bulk_load_investments(self, investments: List[Dict[str, Any]]) -> int:
        """Bulk load investment data."""
        if not investments:
            return 0
            
        result = await self.investments_collection.insert_many(investments)
        return len(result.inserted_ids)
    
    async def bulk_load_transactions(self, transactions: List[Dict[str, Any]]) -> int:
        """Bulk load transaction data."""
        if not transactions:
            return 0
            
        result = await self.transactions_collection.insert_many(transactions)
        return len(result.inserted_ids)
    
    async def bulk_load_accounts(self, accounts: List[Dict[str, Any]]) -> int:
        """Bulk load account data."""
        if not accounts:
            return 0
            
        result = await self.accounts_collection.insert_many(accounts)
        return len(result.inserted_ids)
    
    async def bulk_load_credit_history(self, credit_history: List[Dict[str, Any]]) -> int:
        """Bulk load credit history data."""
        if not credit_history:
            return 0
            
        result = await self.credit_history_collection.insert_many(credit_history)
        return len(result.inserted_ids)
    
    async def bulk_load_demographics(self, demographics: List[Dict[str, Any]]) -> int:
        """Bulk load demographic data."""
        if not demographics:
            return 0
            
        result = await self.demographics_collection.insert_many(demographics)
        return len(result.inserted_ids)
    
    async def bulk_load_products(self, products: List[Dict[str, Any]]) -> int:
        """Bulk load product data."""
        if not products:
            return 0
            
        result = await self.products_collection.insert_many(products)
        return len(result.inserted_ids) 