from motor.motor_asyncio import AsyncIOMotorDatabase
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import logging
import os
from pathlib import Path
import openai
from datetime import datetime

from app.config import settings
from app.database.models import ProductRecommendation, MetaPrompt
from app.utils.vector_store import VectorStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RecommendationEngine:
    """
    Engine for generating personalized financial product recommendations.
    Uses a RAG approach with user's meta-prompt and a vector store of product descriptions.
    """
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.vector_store = VectorStore()
        openai.api_key = settings.OPENAI_API_KEY
        self._load_products()
    
    def _load_products(self):
        """Load the financial products from CSV."""
        try:
            products_path = Path(settings.PRODUCTS_FILE)
            
            if os.path.exists(products_path):
                self.products_df = pd.read_csv(products_path)
                logger.info(f"Loaded {len(self.products_df)} financial products")
                
                # Initialize vector store with products
                self.vector_store.add_texts(
                    texts=self.products_df['description'].tolist(),
                    metadatas=[{"name": name} for name in self.products_df['name'].tolist()]
                )
            else:
                logger.warning(f"Products file not found at {products_path}. Creating a sample file.")
                self._create_sample_products()
        
        except Exception as e:
            logger.error(f"Error loading products: {str(e)}")
            self._create_sample_products()
    
    def _create_sample_products(self):
        """Create sample products if no products file exists."""
        # Create sample products
        products = [
            {
                "name": "High-Yield Savings Account",
                "description": "A savings account with a competitive interest rate, ideal for building an emergency fund or saving for short-term goals. No minimum balance required and FDIC insured up to $250,000."
            },
            {
                "name": "Premium Checking Account",
                "description": "A full-featured checking account with no monthly fees, unlimited transactions, and access to a nationwide ATM network. Includes mobile banking, bill pay, and real-time alerts."
            },
            {
                "name": "Retirement Planning IRA",
                "description": "An Individual Retirement Account (IRA) designed to help you save for retirement with tax advantages. Choose between Traditional or Roth IRA options based on your tax situation."
            },
            {
                "name": "Home Mortgage Loan",
                "description": "Fixed and adjustable-rate mortgage options with competitive rates for purchasing or refinancing your home. Various term lengths available with personalized payment options."
            },
            {
                "name": "Education Savings Plan",
                "description": "A tax-advantaged 529 college savings plan to help save for future education expenses. Contributions grow tax-free when used for qualified education expenses."
            },
            {
                "name": "Credit Builder Card",
                "description": "A secured credit card designed to help establish or rebuild credit history. Reports to all three major credit bureaus with a low annual fee and gradual credit limit increases."
            },
            {
                "name": "Personal Investment Account",
                "description": "A flexible investment account with access to stocks, bonds, mutual funds, and ETFs. Includes personalized investment guidance and portfolio management tools."
            },
            {
                "name": "Auto Loan",
                "description": "Competitive auto loan rates for new and used vehicles with flexible term options. No application fees and discounts available with automatic payments."
            },
            {
                "name": "Small Business Line of Credit",
                "description": "Flexible financing for small business owners with competitive rates. Access funds when needed for inventory, equipment, or managing cash flow fluctuations."
            },
            {
                "name": "Premium Travel Rewards Card",
                "description": "A credit card that earns premium travel rewards on everyday purchases. Includes travel insurance, no foreign transaction fees, and concierge services."
            }
        ]
        
        # Create DataFrame
        self.products_df = pd.DataFrame(products)
        
        # Save to CSV
        os.makedirs(os.path.dirname(settings.PRODUCTS_FILE), exist_ok=True)
        self.products_df.to_csv(settings.PRODUCTS_FILE, index=False)
        logger.info(f"Created sample products file with {len(self.products_df)} products")
        
        # Initialize vector store with products
        self.vector_store.add_texts(
            texts=self.products_df['description'].tolist(),
            metadatas=[{"name": name} for name in self.products_df['name'].tolist()]
        )
    
    async def generate_recommendations(self, user_id: str) -> List[ProductRecommendation]:
        """
        Generate personalized product recommendations for the user.
        Returns a list of the top 3 recommended products with explanations.
        """
        try:
            # Get user meta-prompt
            meta_prompt_doc = await self.db.meta_prompts.find_one({"user_id": user_id})
            
            if not meta_prompt_doc:
                logger.warning(f"No meta-prompt found for user {user_id}. Generating a generic one.")
                # Return generic recommendations if no meta-prompt is available
                return self._generate_generic_recommendations()
            
            meta_prompt = meta_prompt_doc["prompt_text"]
            
            # Use the vector store to retrieve relevant products
            relevant_products = self.vector_store.similarity_search(meta_prompt, k=5)
            
            # Generate personalized recommendations with explanations
            recommendations = await self._generate_personalized_recommendations(meta_prompt, relevant_products)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            # Return generic recommendations in case of error
            return self._generate_generic_recommendations()
    
    async def _generate_personalized_recommendations(
        self, 
        meta_prompt: str, 
        relevant_products: List[Dict[str, Any]]
    ) -> List[ProductRecommendation]:
        """
        Generate personalized recommendations with explanations using the LLM.
        """
        # Format product information for the LLM prompt
        products_text = ""
        for i, product in enumerate(relevant_products, 1):
            products_text += f"{i}. {product['metadata']['name']}: {product['page_content']}\n\n"
        
        # Create LLM prompt
        prompt = f"""Based on the user profile below, recommend the top 3 most suitable financial products from the list provided. 
For each recommendation, provide a clear explanation why it fits this specific user's needs and financial situation.

USER PROFILE:
{meta_prompt}

AVAILABLE PRODUCTS:
{products_text}

INSTRUCTIONS:
- Rank products from most to least suitable for this user
- For each product, write a personalized explanation (2-3 sentences) explaining why it meets their specific needs
- Focus on how the product features align with the user's financial situation, goals, and behavior
- Be specific and reference details from their profile
- Provide a confidence score (0-100) for each recommendation

FORMAT YOUR RESPONSE AS:
1. [Product Name]
   Reason: [Personalized explanation]
   Confidence: [Score]

2. [Product Name]
   Reason: [Personalized explanation]
   Confidence: [Score]

3. [Product Name]
   Reason: [Personalized explanation]
   Confidence: [Score]
"""
        
        # Call the LLM
        try:
            response = await openai.ChatCompletion.acreate(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a financial advisor assistant that provides personalized product recommendations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse the response to extract the recommended products
            recommended_products = self._parse_recommendations(response_text, relevant_products)
            
            # Ensure we have exactly 3 recommendations
            while len(recommended_products) < 3:
                # Add missing recommendations from the relevant products
                for product in relevant_products:
                    product_name = product['metadata']['name']
                    if not any(rec.name == product_name for rec in recommended_products):
                        recommended_products.append(
                            ProductRecommendation(
                                name=product_name,
                                description=product['page_content'],
                                reason=f"This product may also be suitable based on your financial profile.",
                                score=65.0
                            )
                        )
                        if len(recommended_products) >= 3:
                            break
            
            # Only return the top 3
            return recommended_products[:3]
            
        except Exception as e:
            logger.error(f"Error from OpenAI API: {str(e)}")
            return self._generate_generic_recommendations()
    
    def _parse_recommendations(
        self, 
        response_text: str, 
        relevant_products: List[Dict[str, Any]]
    ) -> List[ProductRecommendation]:
        """Parse the LLM response to extract recommended products."""
        recommendations = []
        
        try:
            # Split the response by numbered lines
            parts = response_text.split("\n\n")
            current_product = None
            current_reason = None
            current_confidence = None
            
            for part in parts:
                lines = part.strip().split("\n")
                
                # Skip empty parts
                if not lines or not lines[0].strip():
                    continue
                
                # Check if this is a product line (starts with a number)
                first_line = lines[0].strip()
                if first_line and first_line[0].isdigit():
                    # If we have a complete product from previous lines, add it
                    if current_product and current_reason:
                        # Find the product description
                        description = next(
                            (p['page_content'] for p in relevant_products if p['metadata']['name'] == current_product),
                            "No detailed description available."
                        )
                        
                        recommendations.append(
                            ProductRecommendation(
                                name=current_product,
                                description=description,
                                reason=current_reason,
                                score=current_confidence or 80.0
                            )
                        )
                    
                    # Extract the new product name
                    product_part = first_line.split(".", 1)[1].strip() if "." in first_line else first_line
                    if "[" in product_part and "]" in product_part:
                        current_product = product_part.split("[")[1].split("]")[0].strip()
                    else:
                        current_product = product_part.strip()
                    
                    # Reset other fields
                    current_reason = None
                    current_confidence = None
                
                # Check for reason line
                for line in lines:
                    if line.strip().lower().startswith("reason:"):
                        current_reason = line.split(":", 1)[1].strip()
                    elif "confidence:" in line.lower():
                        confidence_part = line.lower().split("confidence:")[1].strip()
                        try:
                            # Extract numeric value
                            confidence_value = ''.join(c for c in confidence_part if c.isdigit() or c == '.')
                            if confidence_value:
                                current_confidence = float(confidence_value)
                        except:
                            current_confidence = 80.0
            
            # Don't forget the last product
            if current_product and current_reason:
                description = next(
                    (p['page_content'] for p in relevant_products if p['metadata']['name'] == current_product),
                    "No detailed description available."
                )
                
                recommendations.append(
                    ProductRecommendation(
                        name=current_product,
                        description=description,
                        reason=current_reason,
                        score=current_confidence or 80.0
                    )
                )
            
        except Exception as e:
            logger.error(f"Error parsing recommendations: {str(e)}")
        
        return recommendations
    
    def _generate_generic_recommendations(self) -> List[ProductRecommendation]:
        """Generate generic recommendations when personalized ones can't be created."""
        if self.products_df.empty or len(self.products_df) < 3:
            # Create some default recommendations if we don't have products
            return [
                ProductRecommendation(
                    name="High-Yield Savings Account",
                    description="A savings account with competitive interest rates and no minimum balance requirements.",
                    reason="A good foundation for your financial goals. Helps build emergency savings with better returns than standard accounts.",
                    score=90.0
                ),
                ProductRecommendation(
                    name="Retirement Planning IRA",
                    description="Individual Retirement Account with tax advantages for long-term retirement savings.",
                    reason="Essential for long-term financial security. Tax advantages help your retirement savings grow more efficiently.",
                    score=85.0
                ),
                ProductRecommendation(
                    name="Premium Checking Account",
                    description="Feature-rich checking account with no monthly fees and extensive ATM network.",
                    reason="Eliminates common banking fees and offers convenient features for everyday banking needs.",
                    score=80.0
                )
            ]
        
        # Get the first 3 products from our list
        recommendations = []
        for _, row in self.products_df.head(3).iterrows():
            recommendations.append(
                ProductRecommendation(
                    name=row['name'],
                    description=row['description'],
                    reason=f"This is a popular financial product that may meet your needs.",
                    score=75.0
                )
            )
        
        return recommendations 