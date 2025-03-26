from motor.motor_asyncio import AsyncIOMotorDatabase
from typing import Dict, Any, Optional, List
import pandas as pd
import logging
import os
import json
from pathlib import Path

from app.config import settings
from app.utils.data_processor import DataProcessor

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
        Generate a meta-prompt for the given user.
        Combines demographic, financial, and sentiment data into a comprehensive context.
        """
        logger.info(f"Generating meta-prompt for user {user_id}")
        
        try:
            # Get user data from datasets
            user_data = self._get_user_data(user_id)
            
            # Process transaction data for insights
            transaction_insights = self.data_processor.extract_transaction_insights(
                user_data.get("transactions", [])
            )
            
            # Process social media sentiment
            sentiment_insights = self.data_processor.extract_sentiment_insights(
                user_data.get("sentiment", [])
            )
            
            # Combine all data into a structured meta-prompt
            meta_prompt = self._format_meta_prompt(
                user_data.get("demographics", {}),
                user_data.get("account", {}),
                user_data.get("credit", {}),
                user_data.get("investments", {}),
                transaction_insights,
                sentiment_insights
            )
            
            return meta_prompt
            
        except Exception as e:
            logger.error(f"Error generating meta-prompt: {str(e)}")
            # Return a basic meta-prompt in case of error
            return "Financial advisor user with limited context information available."
    
    def _get_user_data(self, user_id: str) -> Dict[str, Any]:
        """Retrieve all available data for the user."""
        result = {
            "demographics": {},
            "account": {},
            "credit": {},
            "investments": {},
            "transactions": [],
            "sentiment": []
        }
        
        # Get demographic data
        if not self.demographic_df.empty:
            user_demographics = self.demographic_df[self.demographic_df['user_id'] == user_id]
            if not user_demographics.empty:
                result["demographics"] = user_demographics.iloc[0].to_dict()
        
        # Get account data
        if not self.account_df.empty:
            user_account = self.account_df[self.account_df['user_id'] == user_id]
            if not user_account.empty:
                result["account"] = user_account.iloc[0].to_dict()
        
        # Get credit history
        if not self.credit_df.empty:
            user_credit = self.credit_df[self.credit_df['user_id'] == user_id]
            if not user_credit.empty:
                result["credit"] = user_credit.iloc[0].to_dict()
        
        # Get investment data
        if not self.investment_df.empty:
            user_investments = self.investment_df[self.investment_df['user_id'] == user_id]
            if not user_investments.empty:
                result["investments"] = user_investments.iloc[0].to_dict()
        
        # Get transaction data
        if not self.transaction_df.empty:
            user_transactions = self.transaction_df[self.transaction_df['user_id'] == user_id]
            if not user_transactions.empty:
                result["transactions"] = user_transactions.to_dict('records')
        
        # Get sentiment data
        if not self.sentiment_df.empty:
            user_sentiment = self.sentiment_df[self.sentiment_df['user_id'] == user_id]
            if not user_sentiment.empty:
                result["sentiment"] = user_sentiment.to_dict('records')
        
        return result
    
    def _format_meta_prompt(
        self,
        demographics: Dict[str, Any],
        account: Dict[str, Any],
        credit: Dict[str, Any],
        investments: Dict[str, Any],
        transaction_insights: Dict[str, Any],
        sentiment_insights: Dict[str, Any]
    ) -> str:
        """Format all data into a coherent meta-prompt."""
        
        # Start with basic demographic information
        sections = []
        
        # Demographic section
        if demographics:
            demo_section = [
                "## Demographic Profile",
                f"Age: {demographics.get('age', 'Unknown')}",
                f"Gender: {demographics.get('gender', 'Unknown')}",
                f"Occupation: {demographics.get('occupation', 'Unknown')}",
                f"Annual Income: ${demographics.get('annual_income', 'Unknown')}",
                f"Education: {demographics.get('education_level', 'Unknown')}",
                f"Location: {demographics.get('city', 'Unknown')}, {demographics.get('state', 'Unknown')}"
            ]
            sections.append("\n".join(demo_section))
        
        # Financial profile section
        if account or credit:
            finance_section = ["## Financial Profile"]
            
            if account:
                finance_section.extend([
                    f"Account Type: {account.get('account_type', 'Unknown')}",
                    f"Account Balance: ${account.get('account_balance', 'Unknown')}",
                    f"Savings Balance: ${account.get('savings_balance', 'Unknown')}",
                    f"Account Opened: {account.get('account_opening_date', 'Unknown')}"
                ])
            
            if credit:
                finance_section.extend([
                    f"Credit Score: {credit.get('credit_score', 'Unknown')}",
                    f"Outstanding Debt: ${credit.get('outstanding_debt', 'Unknown')}",
                    f"Credit Utilization: {credit.get('credit_utilization', 'Unknown')}%",
                    f"Payment History: {credit.get('payment_history', 'Unknown')}"
                ])
            
            sections.append("\n".join(finance_section))
        
        # Investment profile section
        if investments:
            invest_section = [
                "## Investment Profile",
                f"Risk Tolerance: {investments.get('risk_tolerance', 'Unknown')}",
                f"Investment Goals: {investments.get('investment_goals', 'Unknown')}",
                f"Current Investments: ${investments.get('current_investments', 'Unknown')}",
                f"Retirement Savings: ${investments.get('retirement_savings', 'Unknown')}",
                f"Investment Preferences: {investments.get('investment_preferences', 'Unknown')}"
            ]
            sections.append("\n".join(invest_section))
        
        # Transaction insights section
        if transaction_insights:
            trans_section = [
                "## Spending Patterns",
                f"Monthly Spending: ${transaction_insights.get('monthly_spending', 'Unknown')}",
                f"Top Spending Categories: {', '.join(transaction_insights.get('top_categories', ['Unknown']))}",
                f"Recent Large Transactions: {transaction_insights.get('large_transactions', 'None')}"
            ]
            if transaction_insights.get('recurring_payments'):
                trans_section.append(f"Recurring Payments: {transaction_insights.get('recurring_payments', 'None')}")
            
            sections.append("\n".join(trans_section))
        
        # Sentiment insights section
        if sentiment_insights:
            sentiment_section = [
                "## Social Media Insights",
                f"Overall Sentiment: {sentiment_insights.get('overall_sentiment', 'Unknown')}",
                f"Financial Interests: {', '.join(sentiment_insights.get('financial_interests', ['Unknown']))}",
                f"Recent Concerns: {sentiment_insights.get('financial_concerns', 'None')}"
            ]
            sections.append("\n".join(sentiment_section))
        
        # Combine all sections
        meta_prompt = "\n\n".join(sections)
        
        # If we have no data, use a generic prompt
        if not meta_prompt:
            meta_prompt = "New financial advisor user with limited context information available."
        
        return meta_prompt 