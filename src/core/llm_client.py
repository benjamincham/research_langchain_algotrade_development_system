"""
LLM Client Factory Functions

This module provides factory functions to create LLM instances with automatic
failover using LangChain's built-in with_fallbacks() method.

Supports multiple providers: OpenAI, Anthropic, Google (Gemini), Groq.
"""

from langchain_core.language_models import BaseChatModel
from typing import Optional
import logging
import os

logger = logging.getLogger(__name__)


def create_llm_with_fallbacks(
    preferred_provider: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 4096
) -> BaseChatModel:
    """
    Create an LLM with automatic failover using LangChain's built-in fallbacks.
    
    Args:
        preferred_provider: Preferred provider to try first (e.g., "openai", "anthropic")
        temperature: Temperature for generation (0.0 to 1.0)
        max_tokens: Maximum tokens to generate
    
    Returns:
        BaseChatModel with fallbacks configured
    
    Raises:
        ValueError: If no providers are configured
        ImportError: If required provider package is not installed
    
    Example:
        >>> llm = create_llm_with_fallbacks()
        >>> response = llm.invoke("Analyze AAPL stock")
        >>> print(response.content)
    """
    from ..config.llm_credentials import get_credentials
    
    credentials = get_credentials()
    available = credentials.get_available_providers()
    
    if not available:
        raise ValueError(
            "No LLM providers configured. Please set API keys in .env file.\n"
            "Example: OPENAI_API_KEY=sk-... or ANTHROPIC_API_KEY=sk-ant-..."
        )
    
    # Reorder providers if preference specified
    if preferred_provider and preferred_provider in available:
        available.remove(preferred_provider)
        available.insert(0, preferred_provider)
    
    logger.info(f"Configuring LLM with providers: {available}")
    
    # Create model instances for each provider
    models = []
    
    for provider in available:
        try:
            model = _create_model_for_provider(
                provider, 
                credentials, 
                temperature, 
                max_tokens
            )
            if model:
                models.append(model)
                logger.info(f"✓ Configured {provider}")
        except Exception as e:
            logger.warning(f"✗ Failed to configure {provider}: {e}")
    
    if not models:
        raise ValueError("Failed to configure any LLM providers")
    
    # Use LangChain's built-in with_fallbacks() method
    primary = models[0]
    fallbacks = models[1:]
    
    if fallbacks:
        logger.info(f"Primary: {available[0]}, Fallbacks: {available[1:]}")
        return primary.with_fallbacks(fallbacks)
    else:
        logger.info(f"Single provider: {available[0]} (no fallbacks)")
        return primary


def _create_model_for_provider(
    provider: str,
    credentials,
    temperature: float,
    max_tokens: int
) -> Optional[BaseChatModel]:
    """
    Create a model instance for a specific provider.
    
    Args:
        provider: Provider name ("openai", "anthropic", "google", "groq")
        credentials: LLMCredentials instance
        temperature: Temperature for generation
        max_tokens: Maximum tokens to generate
    
    Returns:
        BaseChatModel instance or None if failed
    """
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=credentials.openai_api_key
        )
    
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model="claude-3-5-haiku-20241022",
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=credentials.anthropic_api_key
        )
    
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=temperature,
            max_output_tokens=max_tokens,
            google_api_key=credentials.google_api_key
        )
    
    elif provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=credentials.groq_api_key
        )
    
    else:
        logger.warning(f"Unknown provider: {provider}")
        return None


def create_cheap_llm(
    temperature: float = 0.7,
    max_tokens: int = 4096
) -> BaseChatModel:
    """
    Create LLM optimized for cost (tries cheapest providers first).
    
    Provider order (cheapest first):
    1. Google Gemini (free tier)
    2. Groq ($0.59-0.79 per 1M tokens)
    3. OpenAI GPT-4o-mini ($0.15-0.60 per 1M tokens)
    4. Anthropic Claude Haiku ($0.80-4.00 per 1M tokens)
    
    Args:
        temperature: Temperature for generation
        max_tokens: Maximum tokens to generate
    
    Returns:
        BaseChatModel with cost-optimized provider order
    
    Example:
        >>> llm = create_cheap_llm()
        >>> response = llm.invoke("Simple task")
    """
    from ..config.llm_credentials import get_credentials
    
    credentials = get_credentials()
    
    # Order by cost (cheapest first)
    cost_order = ["google", "groq", "openai", "anthropic"]
    available = [p for p in cost_order if p in credentials.get_available_providers()]
    
    if not available:
        raise ValueError("No LLM providers configured")
    
    logger.info(f"Creating cost-optimized LLM with order: {available}")
    return create_llm_with_fallbacks(
        preferred_provider=available[0],
        temperature=temperature,
        max_tokens=max_tokens
    )


def create_powerful_llm(
    temperature: float = 0.7,
    max_tokens: int = 4096
) -> BaseChatModel:
    """
    Create LLM optimized for capability (tries most capable providers first).
    
    Provider order (most capable first):
    1. Anthropic Claude
    2. OpenAI GPT-4
    3. Google Gemini
    4. Groq Llama
    
    Args:
        temperature: Temperature for generation
        max_tokens: Maximum tokens to generate
    
    Returns:
        BaseChatModel with capability-optimized provider order
    
    Example:
        >>> llm = create_powerful_llm()
        >>> response = llm.invoke("Complex reasoning task")
    """
    from ..config.llm_credentials import get_credentials
    
    credentials = get_credentials()
    
    # Order by capability
    capability_order = ["anthropic", "openai", "google", "groq"]
    available = [p for p in capability_order if p in credentials.get_available_providers()]
    
    if not available:
        raise ValueError("No LLM providers configured")
    
    logger.info(f"Creating capability-optimized LLM with order: {available}")
    return create_llm_with_fallbacks(
        preferred_provider=available[0],
        temperature=temperature,
        max_tokens=max_tokens
    )


def get_default_llm(temperature: float = 0.7) -> BaseChatModel:
    """
    Get the default LLM instance for the system.
    
    Tries to create a powerful LLM first, then falls back to a cheap one.
    
    Args:
        temperature: Temperature for generation
        
    Returns:
        BaseChatModel instance
    """
    try:
        return create_powerful_llm(temperature=temperature)
    except Exception as e:
        logger.warning(f"Failed to create powerful LLM, falling back to cheap: {e}")
        return create_cheap_llm(temperature=temperature)
