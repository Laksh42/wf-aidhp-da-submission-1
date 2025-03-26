import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PromptGenerator:
    """
    Utility for generating contextual prompts for the financial advisor chatbot
    based on user data and analysis.
    """
    
    def __init__(self):
        """Initialize the prompt generator."""
        pass
    
    def generate_meta_prompt(self, user_data: Dict[str, Any], user_query: str) -> str:
        """
        Generate a rich contextual meta-prompt for the chatbot based on user data.
        
        Args:
            user_data: Dictionary containing user information from various collections
            user_query: The user's query to the chatbot
            
        Returns:
            str: A formatted prompt with user context
        """
        try:
            # Extract user demographic information
            demographic = user_data.get('demographic', {})
            name = demographic.get('name', 'User')
            age = demographic.get('age', 'unknown age')
            profession = demographic.get('profession', 'unknown profession')
            
            # Extract account information
            account = user_data.get('account', {})
            account_type = account.get('account_type', 'unknown account type')
            balance = account.get('balance', 0)
            
            # Extract credit information
            credit = user_data.get('credit', {})
            credit_score = credit.get('credit_score', 'unknown')
            
            # Extract transaction insights
            transaction_insights = user_data.get('transaction_insights', {})
            monthly_spending = transaction_insights.get('monthly_spending', 'unknown')
            top_categories = transaction_insights.get('top_categories', [])
            top_categories_str = ", ".join(top_categories) if top_categories else "unknown"
            recurring_payments = transaction_insights.get('recurring_payments', 'none identified')
            
            # Extract investment information
            investment_analysis = user_data.get('investment_analysis', {})
            total_invested = investment_analysis.get('total_invested', 0)
            current_value = investment_analysis.get('current_value', 0)
            overall_return = investment_analysis.get('overall_return', 0)
            
            # Asset allocation
            asset_allocation = investment_analysis.get('asset_allocation', {})
            allocation_str = "; ".join([f"{k}: {v}%" for k, v in asset_allocation.items()]) if asset_allocation else "no investments"
            
            # Sentiment insights
            sentiment_insights = user_data.get('sentiment_insights', {})
            overall_sentiment = sentiment_insights.get('overall_sentiment', 'unknown')
            financial_interests = sentiment_insights.get('financial_interests', [])
            financial_interests_str = ", ".join(financial_interests) if financial_interests else "none identified"
            financial_concerns = sentiment_insights.get('financial_concerns', 'none identified')
            
            # Format the meta-prompt with all available information
            meta_prompt = f"""
            You are a sophisticated financial advisor chatbot designed to provide personalized financial advice.
            
            USER PROFILE:
            - Name: {name}
            - Age: {age}
            - Profession: {profession}
            - Account Type: {account_type}
            - Current Balance: ${balance}
            - Credit Score: {credit_score}
            
            FINANCIAL SNAPSHOT:
            - Monthly Spending: ${monthly_spending}
            - Top Spending Categories: {top_categories_str}
            - Recurring Payments: {recurring_payments}
            - Total Investments: ${total_invested}
            - Current Investment Value: ${current_value}
            - Overall Return: {overall_return}%
            - Asset Allocation: {allocation_str}
            
            FINANCIAL BEHAVIOR & INTERESTS:
            - Social Media Sentiment: {overall_sentiment}
            - Financial Interests: {financial_interests_str}
            - Financial Concerns: {financial_concerns}
            
            Provide thoughtful, personalized financial advice based on this user's specific situation.
            Be conversational but professional. Focus on actionable recommendations tailored to their
            circumstances, interests, and concerns.
            
            USER QUERY: {user_query}
            """
            
            return meta_prompt.strip()
            
        except Exception as e:
            logger.error(f"Error generating meta-prompt: {str(e)}")
            # Fallback to a basic prompt
            return f"""
            You are a financial advisor chatbot.
            
            USER QUERY: {user_query}
            """
    
    def generate_product_recommendation_prompt(self, user_data: Dict[str, Any], 
                                              products: List[Dict[str, Any]], 
                                              user_query: str) -> str:
        """
        Generate a prompt for product recommendations based on user data.
        
        Args:
            user_data: Dictionary containing user information
            products: List of financial product dictionaries
            user_query: The user's query to the chatbot
            
        Returns:
            str: A formatted prompt for product recommendations
        """
        try:
            # Extract key user information
            demographic = user_data.get('demographic', {})
            name = demographic.get('name', 'User')
            age = demographic.get('age', 'unknown')
            income = demographic.get('annual_income', 0)
            
            # Credit information
            credit = user_data.get('credit', {})
            credit_score = credit.get('credit_score', 'unknown')
            
            # Extract investment information
            investment_analysis = user_data.get('investment_analysis', {})
            total_invested = investment_analysis.get('total_invested', 0)
            asset_allocation = investment_analysis.get('asset_allocation', {})
            
            # Format product list
            products_summary = ""
            for idx, product in enumerate(products, 1):
                products_summary += f"""
                Product {idx}: {product.get('name', 'Unknown Product')}
                - Type: {product.get('type', 'Unknown')}
                - Description: {product.get('description', 'No description available')}
                - Requirements: {product.get('requirements', 'None specified')}
                - Fees: {product.get('fees', 'None specified')}
                - Interest Rate/Return: {product.get('interest_rate', 'N/A')}
                """
            
            prompt = f"""
            You are a financial product recommendation assistant.
            
            USER PROFILE:
            - Name: {name}
            - Age: {age}
            - Income: ${income}
            - Credit Score: {credit_score}
            - Current Investments: ${total_invested}
            
            AVAILABLE FINANCIAL PRODUCTS:
            {products_summary}
            
            Based on the user's financial profile and the available products, provide personalized
            recommendations. Explain why each recommended product suits their needs. Consider their age,
            income, credit score, and current investments when making recommendations.
            
            Only recommend products that the user qualifies for based on the requirements.
            
            USER QUERY: {user_query}
            """
            
            return prompt.strip()
            
        except Exception as e:
            logger.error(f"Error generating product recommendation prompt: {str(e)}")
            # Fallback to a basic prompt
            return f"""
            You are a financial product recommendation assistant.
            
            USER QUERY: {user_query}
            """
    
    def generate_investment_analysis_prompt(self, user_data: Dict[str, Any], user_query: str) -> str:
        """
        Generate a prompt for detailed investment analysis and recommendations.
        
        Args:
            user_data: Dictionary containing user information
            user_query: The user's query to the chatbot
            
        Returns:
            str: A formatted prompt for investment analysis
        """
        try:
            # Extract demographic info
            demographic = user_data.get('demographic', {})
            name = demographic.get('name', 'User')
            age = demographic.get('age', 'unknown')
            risk_tolerance = demographic.get('risk_tolerance', 'moderate')
            
            # Investment data
            investments = user_data.get('investments', [])
            investment_analysis = user_data.get('investment_analysis', {})
            
            # Convert investments to string format
            investments_str = ""
            for inv in investments:
                investments_str += f"""
                - {inv.get('investment_name', 'Unknown')} ({inv.get('investment_type', 'Unknown')})
                  Amount: ${inv.get('amount', 0)}, Current Value: ${inv.get('current_value', 0)}
                  Return: {inv.get('return_rate', 0)}%, Risk Level: {inv.get('risk_level', 'Unknown')}
                """
            
            if not investments_str:
                investments_str = "No current investments"
            
            prompt = f"""
            You are an investment analysis advisor specialized in portfolio optimization.
            
            USER PROFILE:
            - Name: {name}
            - Age: {age}
            - Risk Tolerance: {risk_tolerance}
            
            CURRENT INVESTMENT PORTFOLIO:
            {investments_str}
            
            PORTFOLIO ANALYSIS:
            - Total Invested: ${investment_analysis.get('total_invested', 0)}
            - Current Value: ${investment_analysis.get('current_value', 0)}
            - Overall Return: {investment_analysis.get('overall_return', 0)}%
            - Asset Allocation: {'; '.join([f"{k}: {v}%" for k, v in investment_analysis.get('asset_allocation', {}).items()])}
            
            Based on the user's portfolio, age, and risk tolerance, provide a detailed analysis and
            actionable recommendations. Consider diversification, risk-adjusted returns, fee impact,
            and alignment with their financial goals.
            
            Identify strengths and weaknesses in their current allocation. Suggest specific improvements
            while considering tax implications.
            
            USER QUERY: {user_query}
            """
            
            return prompt.strip()
            
        except Exception as e:
            logger.error(f"Error generating investment analysis prompt: {str(e)}")
            # Fallback to a basic prompt
            return f"""
            You are an investment analysis advisor.
            
            USER QUERY: {user_query}
            """ 