from typing import Optional, Any, Union, List
import os
import re
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from functools import lru_cache

# Load environment variables
load_dotenv()

# Clean environment variables with comments
def clean_env_var(var_name, default=None):
    val = os.getenv(var_name, default)
    if val and isinstance(val, str):
        # Remove any trailing comments (assuming comments start with #)
        val = val.split('#')[0].strip()
    return val

def get_int_env(var_name, default=0):
    """Get an integer environment variable, handling comment issues"""
    val = clean_env_var(var_name, str(default))
    try:
        # Extract just the first number part and convert to int
        match = re.search(r'^\d+', val)
        if match:
            return int(match.group(0))
        return int(val)
    except (ValueError, TypeError):
        return default

def get_bool_env(var_name, default=False):
    """Get a boolean environment variable, handling various formats"""
    val = clean_env_var(var_name, str(default)).lower()
    return val in ("true", "1", "t", "yes", "y")

def get_list_env(var_name, default=None, separator=","):
    """Get a list environment variable by splitting a string"""
    if default is None:
        default = []
    val = clean_env_var(var_name)
    if not val:
        return default
    return [item.strip() for item in val.split(separator)]

class Settings(BaseSettings):
    # Base configuration
    APP_NAME: str = "Wells Fargo Financial Assistant"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # MongoDB settings
    MONGODB_URL: str = "mongodb://localhost:27017"
    MONGODB_DB: str = "financial_advisor"
    MONGODB_USER: Optional[str] = None
    MONGODB_PASSWORD: Optional[str] = None
    
    # Redis settings (optional)
    REDIS_URL: Optional[str] = "redis://localhost:6379"
    REDIS_DB: Optional[str] = "0"
    REDIS_PASSWORD: Optional[str] = None
    
    # API Keys
    MISTRAL_API_KEY: str
    HUGGINGFACE_TOKEN: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    PIXIU_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    
    # Model settings
    DEFAULT_MODEL: str = "mistral-tiny"
    FINANCE_MODEL: str = "pixiu-financial"
    CHAT_MODEL: str = "mistral-tiny"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    OPENAI_VISION_MODEL: str = "gpt-4-vision-preview"
    MISTRAL_MODEL: str = "mistral-large-latest"
    OPENAI_MODEL: str = "gpt-4"
    GOOGLE_MODEL: str = "gemini-1.0-pro"
    
    # Security settings
    SECRET_KEY: str
    JWT_SECRET: str = "your-jwt-secret-here"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS settings
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://localhost:8000"
    CORS_ORIGINS: str = "http://localhost:3000"
    
    # File paths and limits
    UPLOAD_DIR: str = "uploads"
    TEMP_DIR: str = "temp"
    DATA_DIR: str = "./data"
    PRODUCTS_FILE: str = "data/products.csv"
    MAX_UPLOAD_SIZE: int = 10485760
    
    # Cache settings
    CACHE_TTL: int = 3600
    CONVERSATION_HISTORY_TTL: int = 86400
    
    # Rate limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_PERIOD: int = 3600
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "app.log"
    
    # Feature flags
    ENABLE_RLHF: bool = False
    ENABLE_SENTIMENT_ANALYSIS: bool = True
    ENABLE_IMAGE_ANALYSIS: bool = True
    ENABLE_ADAPTIVE_RECOMMENDATIONS: bool = True
    ENABLE_RECOMMENDATIONS: bool = True
    ENABLE_MOCK_DATA: bool = False
    
    # Vector store settings
    VECTOR_STORE_PATH: str = "data/vector_store"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

settings = get_settings()

# Application settings
APP_VERSION: str = clean_env_var("APP_VERSION", "0.1.0")

# MongoDB connection string builder
@property
def MONGODB_CONNECTION_STRING(self) -> str:
    """Build MongoDB connection string with or without credentials based on env vars."""
    if self.MONGODB_USER and self.MONGODB_PASSWORD and self.MONGODB_USER != "your_mongodb_user":
        # If credentials are provided and not defaults, use them
        auth_part = f"{self.MONGODB_USER}:{self.MONGODB_PASSWORD}@"
        url_parts = self.MONGODB_URL.split("://")
        if len(url_parts) > 1:
            return f"{url_parts[0]}://{auth_part}{url_parts[1]}"
        return f"mongodb://{auth_part}localhost:27017"
    # Otherwise return the URL as is
    return self.MONGODB_URL

# Redis configuration
REDIS_DB: int = get_int_env("REDIS_DB", 0)

# API Keys
PIXIU_API_KEY: Optional[str] = clean_env_var("PIXIU_API_KEY")

# Model configurations
DEFAULT_MODEL: str = clean_env_var("DEFAULT_MODEL", "mistral-tiny")
CHAT_MODEL: str = clean_env_var("CHAT_MODEL", "mistral-tiny")
EMBEDDING_MODEL: str = clean_env_var("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OPENAI_VISION_MODEL: str = clean_env_var("OPENAI_VISION_MODEL", "gpt-4-vision-preview")

# LLM Service configurations
LLM_MODEL: str = clean_env_var("LLM_MODEL", "gpt-3.5-turbo") 
LLM_API_URL: str = clean_env_var("LLM_API_URL", "https://api.openai.com/v1/chat/completions")
LLM_MAX_TOKENS: int = get_int_env("LLM_MAX_TOKENS", 1000)
LLM_TEMPERATURE: float = float(clean_env_var("LLM_TEMPERATURE", "0.7"))

# File storage paths
UPLOAD_DIR: str = clean_env_var("UPLOAD_DIR", "uploads")
TEMP_DIR: str = clean_env_var("TEMP_DIR", "temp")
DATA_DIR: str = clean_env_var("DATA_DIR", "./data")
MAX_UPLOAD_SIZE: int = get_int_env("MAX_UPLOAD_SIZE", 10485760)

# Cache Settings
# 1 hour in seconds
CACHE_TTL: int = get_int_env("CACHE_TTL", 3600)
# 24 hours in seconds
CONVERSATION_HISTORY_TTL: int = get_int_env("CONVERSATION_HISTORY_TTL", 86400)

# Rate Limiting
RATE_LIMIT_REQUESTS: int = get_int_env("RATE_LIMIT_REQUESTS", 100)
# 1 hour in seconds
RATE_LIMIT_PERIOD: int = get_int_env("RATE_LIMIT_PERIOD", 3600)

# Logging
LOG_LEVEL: str = clean_env_var("LOG_LEVEL", "INFO")
LOG_FILE: str = clean_env_var("LOG_FILE", "app.log")

# Feature flags
ENABLE_IMAGE_ANALYSIS: bool = get_bool_env("ENABLE_IMAGE_ANALYSIS", True)
ENABLE_RECOMMENDATIONS: bool = get_bool_env("ENABLE_RECOMMENDATIONS", True)
ENABLE_RLHF: bool = get_bool_env("ENABLE_RLHF", False)
ENABLE_SENTIMENT_ANALYSIS: bool = get_bool_env("ENABLE_SENTIMENT_ANALYSIS", True)
ENABLE_ADAPTIVE_RECOMMENDATIONS: bool = get_bool_env("ENABLE_ADAPTIVE_RECOMMENDATIONS", True)
ENABLE_MOCK_DATA: bool = get_bool_env("ENABLE_MOCK_DATA", False) 