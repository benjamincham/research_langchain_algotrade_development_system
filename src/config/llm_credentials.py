"""
LLM Credentials Management

This module manages API credentials for multiple LLM providers using environment variables.
Supports OpenAI, Anthropic, Google (Gemini), Groq, and Azure OpenAI.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class LLMCredentials(BaseSettings):
    """
    Manage LLM provider credentials from environment variables.
    
    Supports:
    - OpenAI (GPT models)
    - Anthropic (Claude models)
    - Google (Gemini models)
    - Groq (Fast inference)
    - Azure OpenAI
    
    Example:
        >>> creds = LLMCredentials()
        >>> available = creds.get_available_providers()
        >>> print(f"Available providers: {available}")
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # OpenAI
    openai_api_key: Optional[str] = None
    
    # Anthropic (Claude)
    anthropic_api_key: Optional[str] = None
    
    # Google (Gemini)
    google_api_key: Optional[str] = None
    
    # Groq (Fast inference)
    groq_api_key: Optional[str] = None
    
    # Azure OpenAI
    azure_openai_api_key: Optional[str] = None
    azure_openai_endpoint: Optional[str] = None
    azure_openai_deployment: Optional[str] = None
    
    def get_available_providers(self) -> List[str]:
        """
        Get list of providers with valid credentials.
        
        Returns:
            List of provider names (e.g., ["openai", "anthropic", "google"])
        
        Example:
            >>> creds = LLMCredentials()
            >>> providers = creds.get_available_providers()
            >>> if "openai" in providers:
            ...     print("OpenAI is configured")
        """
        available = []
        
        if self.openai_api_key and len(self.openai_api_key.strip()) > 0:
            available.append("openai")
            logger.debug("OpenAI credentials found")
        
        if self.anthropic_api_key and len(self.anthropic_api_key.strip()) > 0:
            available.append("anthropic")
            logger.debug("Anthropic credentials found")
        
        if self.google_api_key and len(self.google_api_key.strip()) > 0:
            available.append("google")
            logger.debug("Google credentials found")
        
        if self.groq_api_key and len(self.groq_api_key.strip()) > 0:
            available.append("groq")
            logger.debug("Groq credentials found")
        
        if (self.azure_openai_api_key and 
            self.azure_openai_endpoint and 
            self.azure_openai_deployment):
            available.append("azure_openai")
            logger.debug("Azure OpenAI credentials found")
        
        logger.info(f"Available LLM providers: {available}")
        return available
    
    def validate_credentials(self) -> None:
        """
        Validate that at least one provider is configured.
        
        Raises:
            ValueError: If no providers are configured
        
        Example:
            >>> creds = LLMCredentials()
            >>> creds.validate_credentials()  # Raises if no providers
        """
        available = self.get_available_providers()
        
        if not available:
            raise ValueError(
                "No LLM providers configured. Please set at least one API key in .env file.\n"
                "Example: OPENAI_API_KEY=sk-... or ANTHROPIC_API_KEY=sk-ant-...\n"
                "See .env.example for all supported providers."
            )
        
        logger.info(f"Credential validation passed. {len(available)} provider(s) configured.")


# Singleton instance
_credentials: Optional[LLMCredentials] = None


def get_credentials() -> LLMCredentials:
    """
    Get or create the singleton LLMCredentials instance.
    
    Returns:
        LLMCredentials instance
    
    Example:
        >>> creds = get_credentials()
        >>> providers = creds.get_available_providers()
    """
    global _credentials
    if _credentials is None:
        _credentials = LLMCredentials()
        _credentials.validate_credentials()
    return _credentials


def reset_credentials() -> None:
    """
    Reset the singleton instance (useful for testing).
    
    Example:
        >>> reset_credentials()
        >>> # Credentials will be reloaded on next get_credentials() call
    """
    global _credentials
    _credentials = None
    logger.debug("Credentials singleton reset")
