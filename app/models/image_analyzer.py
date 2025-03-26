import logging
import base64
import os
from pathlib import Path
from io import BytesIO
from typing import Dict, Any, List, Optional, Tuple
import openai
from PIL import Image
import numpy as np

from app.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageAnalyzer:
    """
    Service for analyzing images and extracting financial information.
    Uses OpenAI's Vision model to process documents, receipts, statements, etc.
    """
    
    def __init__(self):
        """Initialize the image analyzer service."""
        openai.api_key = settings.OPENAI_API_KEY
        self.upload_folder = Path(settings.UPLOAD_DIR)
        os.makedirs(self.upload_folder, exist_ok=True)
    
    async def analyze_image(self, image_data: bytes, analysis_type: str = "general") -> Dict[str, Any]:
        """
        Analyze an image and extract relevant financial information.
        
        Args:
            image_data: The binary image data
            analysis_type: Type of analysis to perform (general, receipt, statement, document)
            
        Returns:
            Dictionary containing the extracted information
        """
        try:
            # Encode image to base64
            base64_image = self._encode_image(image_data)
            
            # Get prompts based on analysis type
            system_prompt, user_prompt = self._get_prompts_for_analysis_type(analysis_type)
            
            # Call OpenAI Vision API
            response = await openai.ChatCompletion.acreate(
                model=settings.OPENAI_VISION_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000
            )
            
            # Parse the response
            result = self._parse_response(response.choices[0].message.content, analysis_type)
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            return {"error": str(e), "analysis_type": analysis_type, "success": False}
    
    def _encode_image(self, image_data: bytes) -> str:
        """
        Encode image data to base64 string.
        
        Args:
            image_data: Binary image data
            
        Returns:
            Base64-encoded string
        """
        try:
            # Open the image with PIL to process it
            with Image.open(BytesIO(image_data)) as img:
                # Resize large images to reduce API costs
                max_size = (1024, 1024)
                if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                    img.thumbnail(max_size, Image.LANCZOS)
                
                # Convert to RGB if needed
                if img.mode != "RGB":
                    img = img.convert("RGB")
                
                # Save to buffer
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=85)
                buffer.seek(0)
                
                # Encode
                return base64.b64encode(buffer.getvalue()).decode('utf-8')
                
        except Exception as e:
            logger.error(f"Error encoding image: {str(e)}")
            # Fallback to direct encoding
            return base64.b64encode(image_data).decode('utf-8')
    
    def _get_prompts_for_analysis_type(self, analysis_type: str) -> Tuple[str, str]:
        """
        Get appropriate prompts based on the analysis type.
        
        Args:
            analysis_type: Type of analysis to perform
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        # Default system prompt for financial document analysis
        system_prompt = """You are a financial document analysis assistant. Your task is to carefully extract and organize 
        relevant financial information from images of documents. Provide accurate, structured data and ignore any irrelevant information.
        Always format your response in a way that is easy to parse programmatically (JSON-like format when possible).
        If you cannot clearly see or read certain information, indicate that with '[unclear]' rather than guessing."""
        
        # User prompts based on analysis type
        if analysis_type == "receipt":
            user_prompt = """This is a receipt or invoice. Please extract the following information and format it as a structured response:
            - Merchant/Business name
            - Date of transaction
            - Total amount
            - Tax amount (if visible)
            - Tip amount (if applicable)
            - List of items purchased with their individual prices
            - Payment method (if visible)
            
            Format your response in a clear structure that can be easily processed."""
            
        elif analysis_type == "statement":
            user_prompt = """This appears to be a financial statement. Please extract the following information and format it as a structured response:
            - Account type and number (last 4 digits only for security)
            - Institution name
            - Statement period
            - Opening balance
            - Closing balance
            - Total deposits/credits
            - Total withdrawals/debits
            - Significant transactions (anything unusual or large)
            - Any fees charged
            
            Format your response in a clear structure that can be easily processed."""
            
        elif analysis_type == "document":
            user_prompt = """This appears to be a financial document. Please:
            1. Identify what type of document this is (contract, policy, tax form, etc.)
            2. Extract key financial information such as:
               - Amounts and what they represent
               - Dates and deadlines
               - Account references
               - Party names
               - Key terms or obligations
            3. Summarize the most important financial implications in 2-3 sentences
            
            Format your response in a clear structure that can be easily processed."""
            
        else:  # general analysis
            user_prompt = """Please analyze this image that contains financial information. Extract any relevant details such as:
            - Document type
            - Key financial figures and what they represent
            - Dates
            - Account information (obscuring full numbers for security)
            - Any actionable information
            
            Provide a brief summary of what this document is and its financial significance. Format your response in a clear 
            structure that can be easily processed."""
        
        return system_prompt, user_prompt
    
    def _parse_response(self, response_text: str, analysis_type: str) -> Dict[str, Any]:
        """
        Parse the OpenAI response into a structured format.
        
        Args:
            response_text: Text response from OpenAI
            analysis_type: Type of analysis performed
            
        Returns:
            Structured dictionary with extracted information
        """
        # Basic structure for the response
        result = {
            "success": True,
            "analysis_type": analysis_type,
            "raw_text": response_text,
            "structured_data": {}
        }
        
        # Try to extract structured information based on the format
        try:
            # Simple line-by-line parsing for key-value pairs
            lines = response_text.split('\n')
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower().replace(' ', '_')
                    value = value.strip()
                    if key and value:
                        result["structured_data"][key] = value
            
            # Add specific extraction logic based on analysis type
            if analysis_type == "receipt":
                self._extract_receipt_data(response_text, result)
            elif analysis_type == "statement":
                self._extract_statement_data(response_text, result)
            
            # If we couldn't extract structured data, provide the full text
            if not result["structured_data"]:
                result["structured_data"] = {"full_text": response_text}
            
        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
            result["structured_data"] = {"full_text": response_text}
            result["parse_error"] = str(e)
        
        return result
    
    def _extract_receipt_data(self, text: str, result: Dict[str, Any]) -> None:
        """
        Extract structured data from receipt analysis.
        
        Args:
            text: Raw text response
            result: Result dictionary to update
        """
        # Look for common receipt patterns
        try:
            # Try to extract items and prices
            items = []
            current_section = ""
            
            for line in text.split('\n'):
                if "item" in line.lower() and "price" in line.lower():
                    current_section = "items"
                    continue
                
                if current_section == "items" and ":" not in line and line.strip():
                    # Try to split by last space to separate item from price
                    parts = line.rsplit(' ', 1)
                    if len(parts) == 2:
                        item_name = parts[0].strip()
                        price_str = parts[1].strip().replace('$', '')
                        
                        try:
                            price = float(price_str)
                            items.append({"item": item_name, "price": price})
                        except ValueError:
                            # Not a valid price format, just add as text
                            items.append({"item": line.strip(), "price": None})
            
            if items:
                result["structured_data"]["items"] = items
        
        except Exception as e:
            logger.error(f"Error extracting receipt items: {str(e)}")
    
    def _extract_statement_data(self, text: str, result: Dict[str, Any]) -> None:
        """
        Extract structured data from statement analysis.
        
        Args:
            text: Raw text response
            result: Result dictionary to update
        """
        # Look for common statement patterns
        try:
            # Try to extract transactions
            transactions = []
            current_section = ""
            
            for line in text.split('\n'):
                if "transaction" in line.lower() or "activity" in line.lower():
                    current_section = "transactions"
                    continue
                
                if current_section == "transactions" and line.strip():
                    # Try to extract date, description, amount
                    # This is a simplified approach - real parsing would need more robust logic
                    parts = line.split(' ', 1)
                    if len(parts) == 2:
                        date_str = parts[0].strip()
                        rest = parts[1].strip()
                        
                        # Try to find the amount (usually last item with $ sign)
                        if '$' in rest:
                            desc_parts = rest.rsplit('$', 1)
                            description = desc_parts[0].strip()
                            amount_str = desc_parts[1].strip()
                            
                            try:
                                amount = float(amount_str.replace(',', ''))
                                transactions.append({
                                    "date": date_str,
                                    "description": description,
                                    "amount": amount
                                })
                            except ValueError:
                                # Not a valid amount format, just add as text
                                transactions.append({
                                    "text": line.strip()
                                })
            
            if transactions:
                result["structured_data"]["transactions"] = transactions
        
        except Exception as e:
            logger.error(f"Error extracting statement transactions: {str(e)}")
    
    async def save_uploaded_image(self, image_data: bytes, filename: str) -> str:
        """
        Save an uploaded image to disk.
        
        Args:
            image_data: Binary image data
            filename: Original filename
            
        Returns:
            Path to the saved file
        """
        try:
            # Create a safe filename
            safe_filename = ''.join(c for c in filename if c.isalnum() or c in '._-')
            timestamp = int(os.path.getctime(self.upload_folder))
            unique_filename = f"{timestamp}_{safe_filename}"
            
            # Save the file
            file_path = self.upload_folder / unique_filename
            with open(file_path, 'wb') as f:
                f.write(image_data)
            
            logger.info(f"Saved uploaded image to {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error saving uploaded image: {str(e)}")
            return "" 