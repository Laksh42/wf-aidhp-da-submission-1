from motor.motor_asyncio import AsyncIOMotorDatabase
from typing import Dict, Any, Optional, List
import pandas as pd
import logging
import os
import json
from pathlib import Path
from datetime import datetime, timedelta

from app.config import settings
from app.utils.data_processor import DataProcessor
from app.utils.meta_prompt_templates import (
    USER_PROFILE_TEMPLATE,
    FINANCIAL_CONTEXT_TEMPLATE,
    TRANSACTION_CONTEXT_TEMPLATE,
    SOCIAL_MEDIA_TEMPLATE,
    FULL_META_PROMPT_TEMPLATE,
    format_financial_goals,
    format_transaction_summary,
    format_investment_summary,
    format_social_media_insights
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetaPromptGenerator:
    """
    Generates personalized meta-prompts for users based on their data.
    These meta-prompts combine demographics, financial profile, transaction history,
    and social media sentiment into a rich context for the LLM.
    """
    
    def __init__(self, db: AsyncIOMotorDatabase):
        """Initialize the meta-prompt generator with database connection"""
        self.db = db
        self.data_processor = DataProcessor()
        self._load_datasets()
    
    def _load_datasets(self):
        """Load the required datasets."""
        data_dir = Path(settings.DATA_DIR)
        
        # Ensure data directory exists
        if not os.path.exists(data_dir):
            logger.warning(f"Data directory {data_dir} not found. Creating it.")
            os.makedirs(data_dir, exist_ok=True)
        
        try:
            # Load demographic data
            demographic_path = data_dir / "demographic_data.csv"
            if os.path.exists(demographic_path):
                self.demographic_df = pd.read_csv(demographic_path)
                logger.info(f"Loaded demographic data with {len(self.demographic_df)} records")
            else:
                logger.warning(f"Demographic data file not found at {demographic_path}")
                self.demographic_df = pd.DataFrame()
            
            # Load account data
            account_path = data_dir / "account_data.csv"
            if os.path.exists(account_path):
                self.account_df = pd.read_csv(account_path)
                logger.info(f"Loaded account data with {len(self.account_df)} records")
            else:
                logger.warning(f"Account data file not found at {account_path}")
                self.account_df = pd.DataFrame()
            
            # Load transaction data
            transaction_path = data_dir / "transaction_data.csv"
            if os.path.exists(transaction_path):
                self.transaction_df = pd.read_csv(transaction_path)
                logger.info(f"Loaded transaction data with {len(self.transaction_df)} records")
            else:
                logger.warning(f"Transaction data file not found at {transaction_path}")
                self.transaction_df = pd.DataFrame()
            
            # Load credit history data
            credit_path = data_dir / "credit_history.csv"
            if os.path.exists(credit_path):
                self.credit_df = pd.read_csv(credit_path)
                logger.info(f"Loaded credit history data with {len(self.credit_df)} records")
            else:
                logger.warning(f"Credit history file not found at {credit_path}")
                self.credit_df = pd.DataFrame()
            
            # Load investment data
            investment_path = data_dir / "investment_data.csv"
            if os.path.exists(investment_path):
                self.investment_df = pd.read_csv(investment_path)
                logger.info(f"Loaded investment data with {len(self.investment_df)} records")
            else:
                logger.warning(f"Investment data file not found at {investment_path}")
                self.investment_df = pd.DataFrame()
            
            # Load social media sentiment data
            sentiment_path = data_dir / "social_media_sentiment.csv"
            if os.path.exists(sentiment_path):
                self.sentiment_df = pd.read_csv(sentiment_path)
                logger.info(f"Loaded social media sentiment data with {len(self.sentiment_df)} records")
            else:
                logger.warning(f"Social media sentiment file not found at {sentiment_path}")
                self.sentiment_df = pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading datasets: {str(e)}")
            # Initialize empty dataframes in case of error
            self.demographic_df = pd.DataFrame()
            self.account_df = pd.DataFrame()
            self.transaction_df = pd.DataFrame()
            self.credit_df = pd.DataFrame()
            self.investment_df = pd.DataFrame()
            self.sentiment_df = pd.DataFrame()
    
    async def generate_meta_prompt(self, user_id: str) -> str:
        """
        Generate a comprehensive meta-prompt for the financial advisor AI
        based on the user's demographic, financial, and behavioral data
        
        Args:
            user_id: The unique identifier for the user
            
        Returns:
            A formatted meta-prompt string
        """
        try:
            # Gather all user data
            user_data = await self._gather_user_data(user_id)
            
            if not user_data:
                logger.warning(f"No data found for user {user_id}, generating generic meta-prompt")
                return self._generate_generic_meta_prompt(user_id)
            
            # Generate the sections of the meta-prompt
            user_profile = self._generate_user_profile(user_data)
            financial_context = self._generate_financial_context(user_data)
            transaction_context = self._generate_transaction_context(user_data)
            social_media_context = self._generate_social_media_context(user_data)
            
            # Combine into the full meta-prompt
            meta_prompt = FULL_META_PROMPT_TEMPLATE.format(
                name=user_data.get('demographics', {}).get('name', user_id),
                age=user_data.get('demographics', {}).get('age', 'unknown age'),
                occupation=user_data.get('demographics', {}).get('occupation', 'professional'),
                location=user_data.get('demographics', {}).get('location', 'unknown location'),
                user_profile=user_profile,
                financial_context=financial_context,
                transaction_context=transaction_context,
                social_media_context=social_media_context,
                risk_tolerance=user_data.get('demographics', {}).get('risk_tolerance', 'Moderate'),
                financial_goals=format_financial_goals(user_data.get('demographics', {}).get('financial_goals', []))
            )
            
            return meta_prompt
            
        except Exception as e:
            logger.error(f"Error generating meta-prompt for user {user_id}: {str(e)}")
            return self._generate_generic_meta_prompt(user_id)
    
    async def _gather_user_data(self, user_id: str) -> Dict[str, Any]:
        """
        Gather all relevant user data from different collections
        
        Args:
            user_id: The unique identifier for the user
            
        Returns:
            Dictionary with all user data organized by category
        """
        user_data = {}
        
        # Get demographic data
        demographics = await self.db.demographic_data.find_one({"user_id": user_id})
        if demographics:
            user_data['demographics'] = demographics
        
        # Get account data
        account = await self.db.account_data.find_one({"user_id": user_id})
        if account:
            user_data['account'] = account
        
        # Get credit history
        credit = await self.db.credit_history.find_one({"user_id": user_id})
        if credit:
            user_data['credit'] = credit
        
        # Get investment data
        investments = []
        async for investment in self.db.investment_data.find({"user_id": user_id}):
            investments.append(investment)
        if investments:
            user_data['investments'] = investments
        
        # Get recent transactions (last 30 days)
        thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        transactions = []
        
        transaction_cursor = self.db.transaction_data.find({
            "user_id": user_id,
            "date": {"$gte": thirty_days_ago}
        }).sort("date", -1)
        
        async for transaction in transaction_cursor:
            transactions.append(transaction)
        
        if transactions:
            user_data['transactions'] = transactions
        
        # Get social media sentiment data (last 30 days)
        social_posts = []
        
        social_cursor = self.db.social_media_sentiment.find({
            "user_id": user_id,
            "date": {"$gte": thirty_days_ago}
        }).sort("date", -1)
        
        async for post in social_cursor:
            social_posts.append(post)
        
        if social_posts:
            user_data['social_media'] = social_posts
        
        return user_data
    
    def _generate_user_profile(self, user_data: Dict[str, Any]) -> str:
        """Generate the user profile section of the meta-prompt"""
        demographics = user_data.get('demographics', {})
        
        try:
            annual_income = float(demographics.get('annual_income', 0))
            annual_expenses = float(demographics.get('annual_expenses', 0))
        except (ValueError, TypeError):
            annual_income = 0
            annual_expenses = 0
        
        financial_goals = demographics.get('financial_goals', [])
        if isinstance(financial_goals, str):
            financial_goals = [financial_goals]
        
        user_profile = USER_PROFILE_TEMPLATE.format(
            name=demographics.get('name', 'User'),
            age=demographics.get('age', 'Unknown'),
            occupation=demographics.get('occupation', 'Unknown'),
            location=demographics.get('location', 'Unknown'),
            annual_income=annual_income,
            annual_expenses=annual_expenses,
            risk_tolerance=demographics.get('risk_tolerance', 'Moderate'),
            financial_goals=format_financial_goals(financial_goals)
        )
        
        return user_profile
    
    def _generate_financial_context(self, user_data: Dict[str, Any]) -> str:
        """Generate the financial context section of the meta-prompt"""
        account = user_data.get('account', {})
        credit = user_data.get('credit', {})
        investments = user_data.get('investments', [])
        
        try:
            checking_balance = float(account.get('account_balance', 0))
            savings_balance = float(account.get('savings_balance', 0))
            total_balance = checking_balance + savings_balance
            
            credit_score = int(credit.get('credit_score', 0))
            outstanding_debt = float(credit.get('outstanding_debt', 0))
            credit_utilization = float(credit.get('credit_utilization', 0))
            
        except (ValueError, TypeError):
            checking_balance = 0
            savings_balance = 0
            total_balance = 0
            credit_score = 0
            outstanding_debt = 0
            credit_utilization = 0
        
        financial_context = FINANCIAL_CONTEXT_TEMPLATE.format(
            account_type=account.get('account_type', 'Standard'),
            checking_balance=checking_balance,
            savings_balance=savings_balance,
            total_balance=total_balance,
            credit_score=credit_score,
            outstanding_debt=outstanding_debt,
            credit_utilization=credit_utilization,
            payment_history=credit.get('payment_history', 'Unknown'),
            credit_age_years=credit.get('credit_age_years', 0),
            investment_summary=format_investment_summary(investments)
        )
        
        return financial_context
    
    def _generate_transaction_context(self, user_data: Dict[str, Any]) -> str:
        """Generate the transaction context section of the meta-prompt"""
        transactions = user_data.get('transactions', [])
        
        if not transactions:
            return "# RECENT TRANSACTIONS\nNo recent transaction data available."
        
        # Find largest expense
        expenses = [t for t in transactions if float(t.get('amount', 0)) < 0]
        largest_expense = "None"
        if expenses:
            largest_expense_txn = min(expenses, key=lambda x: float(x.get('amount', 0)))
            largest_expense = f"${abs(float(largest_expense_txn.get('amount', 0))):,.2f} " + \
                             f"({largest_expense_txn.get('category', 'Unknown')}, {largest_expense_txn.get('merchant', 'Unknown')})"
        
        # Find most frequent category
        categories = {}
        for t in transactions:
            category = t.get('category', 'Unknown')
            if category not in categories:
                categories[category] = 0
            categories[category] += 1
        
        most_frequent_category = "None"
        if categories:
            most_frequent_category = max(categories.items(), key=lambda x: x[1])[0]
        
        # Look for unusual activity (simple approach - transactions > 2x the average)
        avg_amount = sum(abs(float(t.get('amount', 0))) for t in transactions) / len(transactions)
        unusual_txns = [t for t in transactions if abs(float(t.get('amount', 0))) > 2 * avg_amount]
        
        unusual_activity = "None detected"
        if unusual_txns:
            unusual_activity = f"{len(unusual_txns)} transactions significantly above average spending"
        
        transaction_context = TRANSACTION_CONTEXT_TEMPLATE.format(
            recent_transactions=format_transaction_summary(transactions),
            largest_expense=largest_expense,
            most_frequent_category=most_frequent_category,
            unusual_activity=unusual_activity
        )
        
        return transaction_context
    
    def _generate_social_media_context(self, user_data: Dict[str, Any]) -> str:
        """Generate the social media context section of the meta-prompt"""
        social_posts = user_data.get('social_media', [])
        
        if not social_posts:
            return "# SOCIAL MEDIA INSIGHTS\nNo social media data available."
        
        return SOCIAL_MEDIA_TEMPLATE.format(
            topics="See below",
            overall_sentiment="See below",
            financial_interests="See below"
        ) + format_social_media_insights(social_posts)
    
    def _generate_generic_meta_prompt(self, user_id: str) -> str:
        """Generate a generic meta-prompt when user data is unavailable"""
        return f"""
You are a personalized financial wellness assistant. Your client's data is limited, so provide general financial advice that is helpful for a wide audience. Your advice should be:

1. Universal and applicable to people in different financial situations
2. Educational and focused on financial literacy
3. Balanced between short-term financial management and long-term financial goals
4. Respectful of different risk tolerances and financial priorities
5. Clear and jargon-free for accessibility

Avoid making specific assumptions about the user's financial situation, goals, or behaviors without explicit information. When the user provides more details about their situation, incorporate that information into your advice.
""" 