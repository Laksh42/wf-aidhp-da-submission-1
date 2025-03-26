import pandas as pd
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import numpy as np
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Utility for processing user data and extracting insights.
    Used for generating meta-prompts and providing analysis.
    """
    
    def __init__(self):
        """Initialize the processor."""
        pass
    
    def extract_transaction_insights(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract insights from transaction data.
        
        Args:
            transactions: List of transaction dictionaries
            
        Returns:
            Dictionary of insights
        """
        if not transactions:
            return {}
        
        try:
            # Convert to DataFrame for easier analysis
            df = pd.DataFrame(transactions)
            
            # Calculate monthly spending (debits only)
            debits = df[df['transaction_type'] == 'debit']
            monthly_spending = round(debits['amount'].sum(), 2)
            
            # Get top spending categories
            if 'category' in debits.columns and not debits.empty:
                category_counts = debits['category'].value_counts()
                top_categories = category_counts.head(3).index.tolist()
            else:
                top_categories = []
            
            # Find large transactions (above 1000)
            large_transactions = []
            for _, row in debits[debits['amount'] > 1000].iterrows():
                large_transactions.append(f"{row['merchant']} (${row['amount']})")
            
            # Detect recurring payments (same merchant, similar amount)
            recurring_payments = []
            merchant_counts = debits['merchant'].value_counts()
            recurring_merchants = merchant_counts[merchant_counts > 1].index
            
            for merchant in recurring_merchants:
                merchant_data = debits[debits['merchant'] == merchant]
                if len(merchant_data) >= 2:
                    avg_amount = merchant_data['amount'].mean()
                    recurring_payments.append(f"{merchant} (${round(avg_amount, 2)} monthly)")
            
            return {
                'monthly_spending': monthly_spending,
                'top_categories': top_categories,
                'large_transactions': ", ".join(large_transactions) if large_transactions else None,
                'recurring_payments': ", ".join(recurring_payments[:3]) if recurring_payments else None
            }
            
        except Exception as e:
            logger.error(f"Error extracting transaction insights: {str(e)}")
            return {}
    
    def extract_sentiment_insights(self, social_media: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract insights from social media sentiment data.
        
        Args:
            social_media: List of social media post dictionaries
            
        Returns:
            Dictionary of insights
        """
        if not social_media:
            return {}
        
        try:
            # Convert to DataFrame for easier analysis
            df = pd.DataFrame(social_media)
            
            # Calculate overall sentiment
            if 'sentiment' in df.columns:
                sentiment_counts = df['sentiment'].value_counts()
                most_common = sentiment_counts.idxmax()
                
                total_posts = len(df)
                positive_pct = round(sentiment_counts.get('positive', 0) / total_posts * 100)
                negative_pct = round(sentiment_counts.get('negative', 0) / total_posts * 100)
                
                if positive_pct > 60:
                    overall_sentiment = f"Very positive ({positive_pct}% positive posts)"
                elif positive_pct > 40:
                    overall_sentiment = f"Generally positive ({positive_pct}% positive posts)"
                elif negative_pct > 60:
                    overall_sentiment = f"Very negative ({negative_pct}% negative posts)"
                elif negative_pct > 40:
                    overall_sentiment = f"Generally negative ({negative_pct}% negative posts)"
                else:
                    overall_sentiment = "Neutral/mixed"
            else:
                overall_sentiment = "Unknown"
            
            # Extract financial interests from topics
            financial_interests = []
            financial_topics = ['investment', 'finance', 'retirement', 'saving', 'budget', 
                               'mortgage', 'stock', 'bond', 'etf', 'mutual fund', 'real estate']
            
            if 'topics' in df.columns:
                for topics_str in df['topics']:
                    topics = str(topics_str).lower().split()
                    for topic in topics:
                        if any(fin_topic in topic for fin_topic in financial_topics):
                            financial_interests.append(topic)
            
            # Get unique interests and sort by frequency
            if financial_interests:
                counter = Counter(financial_interests)
                top_interests = [item for item, _ in counter.most_common(5)]
            else:
                top_interests = []
            
            # Extract financial concerns from negative posts
            financial_concerns = []
            if 'sentiment' in df.columns and 'post_text' in df.columns:
                negative_posts = df[df['sentiment'] == 'negative']['post_text'].tolist()
                if negative_posts:
                    financial_concerns = negative_posts[:2]  # Just include recent negative posts
            
            return {
                'overall_sentiment': overall_sentiment,
                'financial_interests': top_interests,
                'financial_concerns': "; ".join(financial_concerns) if financial_concerns else None
            }
            
        except Exception as e:
            logger.error(f"Error extracting sentiment insights: {str(e)}")
            return {}
    
    def analyze_investment_portfolio(self, investments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze investment portfolio distribution and performance.
        
        Args:
            investments: List of investment dictionaries
            
        Returns:
            Dictionary of analysis
        """
        if not investments:
            return {}
        
        try:
            df = pd.DataFrame(investments)
            
            # Calculate total investment value
            total_invested = df['amount'].sum()
            current_value = df['current_value'].sum()
            
            # Calculate performance
            overall_return = round((current_value - total_invested) / total_invested * 100, 2)
            
            # Calculate asset allocation
            if 'investment_type' in df.columns:
                asset_allocation = df.groupby('investment_type')['current_value'].sum()
                allocation_pct = (asset_allocation / current_value * 100).round(1).to_dict()
            else:
                allocation_pct = {}
            
            return {
                'total_invested': total_invested,
                'current_value': current_value,
                'overall_return': overall_return,
                'asset_allocation': allocation_pct
            }
            
        except Exception as e:
            logger.error(f"Error analyzing investment portfolio: {str(e)}")
            return {} 