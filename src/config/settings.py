import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

class Settings(BaseSettings):
    """System settings and configuration."""
    
    # Project Info
    PROJECT_NAME: str = "Research LangChain AlgoTrade Development System"
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    
    # LLM Configuration
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_API_BASE: str = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    DEFAULT_MODEL: str = "gpt-4.1-mini"
    
    # Vector Store Configuration
    CHROMA_DB_PATH: str = str(PROJECT_ROOT / "data" / "chroma_db")
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    
    # Trading Configuration
    DEFAULT_CURRENCY: str = "USD"
    INITIAL_CASH: float = 100000.0
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

# Global settings instance
settings = Settings()
