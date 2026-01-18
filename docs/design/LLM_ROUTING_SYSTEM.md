# LLM Routing System Design: Using LangChain's Built-in Fallbacks

## Problem Statement

In a production agentic system, relying on a single LLM provider creates critical vulnerabilities:

1. **Provider Outages**: OpenAI, Anthropic, or any provider can experience downtime
2. **Rate Limiting**: Hitting rate limits halts the entire system
3. **Cost Optimization**: Different providers have different pricing
4. **Model Availability**: Specific models may be unavailable or deprecated
5. **Hard-Coded Credentials**: Hard-coding API keys makes the system inflexible and insecure

**Goal**: Configure multi-provider LLM support with automatic failover using **LangChain's built-in features** (no custom implementation needed).

## LangChain's Built-in Solution

LangChain provides **`with_fallbacks()`** method on all Runnable objects, including chat models. This is the standard, recommended way to implement failover.

### Basic Usage

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Create primary and fallback models
primary = ChatOpenAI(model="gpt-4o-mini")
fallback = ChatAnthropic(model="claude-3-5-haiku-20241022")

# Add fallback using built-in method
model = primary.with_fallbacks([fallback])

# Use normally - automatically fails over on errors
response = model.invoke("Analyze AAPL stock")
```

**That's it!** No custom classes, no reinventing the wheel.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     SIMPLIFIED LLM ROUTING SYSTEM                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │              CREDENTIAL MANAGER (Simple)                         │   │
│  │  • Load credentials from environment variables                   │   │
│  │  • Validate that at least one provider is configured             │   │
│  │  • Return list of available providers                            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │              LLM FACTORY (Simple)                                │   │
│  │  • Create ChatModel instances for each available provider        │   │
│  │  • Apply with_fallbacks() using LangChain's built-in method      │   │
│  │  • Return configured model ready to use                          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │              AGENTS USE CONFIGURED MODEL                         │   │
│  │  • Completely transparent failover                               │   │
│  │  • No custom code needed                                         │   │
│  │  • Standard LangChain patterns                                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Implementation

### 1. Credential Management (Keep Simple)

```python
# config/llm_credentials.py

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
import os

class LLMCredentials(BaseSettings):
    """Simple credential management using environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # OpenAI
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    
    # Anthropic
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    
    # Google (Gemini)
    google_api_key: Optional[str] = Field(None, env="GOOGLE_API_KEY")
    
    # Groq
    groq_api_key: Optional[str] = Field(None, env="GROQ_API_KEY")
    
    # Azure OpenAI
    azure_openai_api_key: Optional[str] = Field(None, env="AZURE_OPENAI_API_KEY")
    azure_openai_endpoint: Optional[str] = Field(None, env="AZURE_OPENAI_ENDPOINT")
    
    def get_available_providers(self) -> list[str]:
        """Return list of providers with valid credentials."""
        available = []
        
        if self.openai_api_key:
            available.append("openai")
        if self.anthropic_api_key:
            available.append("anthropic")
        if self.google_api_key:
            available.append("google")
        if self.groq_api_key:
            available.append("groq")
        if self.azure_openai_api_key and self.azure_openai_endpoint:
            available.append("azure_openai")
        
        return available


# Singleton
_credentials: Optional[LLMCredentials] = None

def get_credentials() -> LLMCredentials:
    """Get or create credential manager singleton."""
    global _credentials
    if _credentials is None:
        _credentials = LLMCredentials()
    return _credentials
```

### 2. LLM Factory (Simple Wrapper)

```python
# core/llm_client.py

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.language_models import BaseChatModel
from typing import Optional
import logging

from ..config.llm_credentials import get_credentials

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
        temperature: Temperature for generation
        max_tokens: Maximum tokens to generate
    
    Returns:
        BaseChatModel with fallbacks configured
    
    Raises:
        ValueError: If no providers are configured
    """
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
            if provider == "openai":
                models.append(
                    ChatOpenAI(
                        model="gpt-4o-mini",
                        temperature=temperature,
                        max_tokens=max_tokens,
                        api_key=credentials.openai_api_key
                    )
                )
            
            elif provider == "anthropic":
                models.append(
                    ChatAnthropic(
                        model="claude-3-5-haiku-20241022",
                        temperature=temperature,
                        max_tokens=max_tokens,
                        api_key=credentials.anthropic_api_key
                    )
                )
            
            elif provider == "google":
                models.append(
                    ChatGoogleGenerativeAI(
                        model="gemini-2.0-flash-exp",
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                        google_api_key=credentials.google_api_key
                    )
                )
            
            elif provider == "groq":
                models.append(
                    ChatGroq(
                        model="llama-3.3-70b-versatile",
                        temperature=temperature,
                        max_tokens=max_tokens,
                        api_key=credentials.groq_api_key
                    )
                )
            
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


def create_cheap_llm() -> BaseChatModel:
    """Create LLM optimized for cost (tries cheapest providers first)."""
    credentials = get_credentials()
    
    # Order by cost (cheapest first)
    cost_order = ["google", "groq", "openai", "anthropic"]
    available = [p for p in cost_order if p in credentials.get_available_providers()]
    
    if not available:
        raise ValueError("No LLM providers configured")
    
    return create_llm_with_fallbacks(preferred_provider=available[0])


def create_powerful_llm() -> BaseChatModel:
    """Create LLM optimized for capability (tries most capable providers first)."""
    credentials = get_credentials()
    
    # Order by capability
    capability_order = ["anthropic", "openai", "google", "groq"]
    available = [p for p in capability_order if p in credentials.get_available_providers()]
    
    if not available:
        raise ValueError("No LLM providers configured")
    
    return create_llm_with_fallbacks(preferred_provider=available[0])
```

### 3. Usage in Agents

```python
# Example: Using in BaseAgent

from src.core.llm_client import create_llm_with_fallbacks

class BaseAgent:
    def __init__(self, name: str, role: str, llm: Optional[BaseChatModel] = None):
        self.name = name
        self.role = role
        
        # If no LLM provided, create one with fallbacks
        self.llm = llm or create_llm_with_fallbacks()
    
    def invoke(self, prompt: str) -> str:
        # Failover happens automatically via LangChain's built-in mechanism
        response = self.llm.invoke(prompt)
        return response.content
```

### 4. Environment Configuration

```bash
# .env file

# Configure as many providers as you want for redundancy
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
GROQ_API_KEY=gsk_...
```

## How It Works

### Automatic Failover

When you call `model.invoke()`:

1. **LangChain tries the primary model** (first in the list)
2. **If it fails** (error, timeout, rate limit), LangChain automatically tries the first fallback
3. **If that fails**, tries the next fallback
4. **Continues until success** or all models exhausted

### What Triggers Failover?

- API errors (rate limits, authentication, service unavailable)
- Timeouts
- Network errors
- Model-specific errors

### Logging

LangChain logs all failover events automatically:

```
INFO: Configuring LLM with providers: ['openai', 'anthropic', 'google']
INFO: Primary: openai, Fallbacks: ['anthropic', 'google']
WARNING: Error in ChatOpenAI: Rate limit exceeded
INFO: Falling back to ChatAnthropic
```

## Configuration Patterns

### Development (Free/Cheap)

```python
# Use cheapest models
llm = create_cheap_llm()
```

```bash
# .env
GOOGLE_API_KEY=AIza...  # Free
GROQ_API_KEY=gsk_...    # Very cheap
```

### Production (Reliability)

```python
# Use all available providers for maximum uptime
llm = create_llm_with_fallbacks()
```

```bash
# .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
GROQ_API_KEY=gsk_...
```

### Production (Performance)

```python
# Use most capable models first
llm = create_powerful_llm()
```

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...  # Claude (most capable)
OPENAI_API_KEY=sk-...         # GPT-4 (fallback)
```

### Custom Order

```python
# Specify exact provider order
llm = create_llm_with_fallbacks(preferred_provider="anthropic")
```

## Benefits

### 1. No Reinventing the Wheel ✅
- Uses LangChain's battle-tested `with_fallbacks()` method
- Well-documented and maintained by LangChain team
- Standard pattern across the ecosystem

### 2. Simple Implementation ✅
- ~100 lines of code vs ~500 lines for custom solution
- Easy to understand and maintain
- No custom Runnable classes needed

### 3. Transparent to Agents ✅
- Agents just use `llm.invoke()` as normal
- Failover happens automatically
- No special handling needed

### 4. Flexible Configuration ✅
- Add/remove providers by changing .env file
- No code changes needed
- Multiple configuration strategies

### 5. Production-Ready ✅
- Built-in logging
- Error handling
- Streaming support
- Async support

## Comparison: Custom vs Built-in

| Feature | Custom `RoutedChatModel` | LangChain `with_fallbacks()` |
|---------|--------------------------|------------------------------|
| Lines of code | ~500 | ~100 |
| Maintenance | Custom code to maintain | Maintained by LangChain |
| Documentation | Need to write our own | Official LangChain docs |
| Features | Need to implement | Already implemented |
| Testing | Need to write tests | Already tested |
| Community support | None | Large community |
| Updates | Manual | Automatic with LangChain |

**Winner**: LangChain's built-in solution

## Implementation Checklist

- [ ] Implement `LLMCredentials` with pydantic-settings
- [ ] Implement `create_llm_with_fallbacks()` factory function
- [ ] Add `create_cheap_llm()` and `create_powerful_llm()` helpers
- [ ] Update `BaseAgent` to use factory functions
- [ ] Create `.env.example` file
- [ ] Add logging for provider configuration
- [ ] Add unit tests
- [ ] Update documentation

## Migration from Custom Solution

If you already implemented the custom `RoutedChatModel`:

```python
# OLD (custom)
from src.core.llm_router import RoutedChatModel
llm = RoutedChatModel(strategy="fallback")

# NEW (LangChain built-in)
from src.core.llm_client import create_llm_with_fallbacks
llm = create_llm_with_fallbacks()
```

Same functionality, simpler implementation!

## Conclusion

**Use LangChain's built-in `with_fallbacks()` method instead of creating a custom solution.** It's simpler, better maintained, and follows LangChain best practices.

Our role is to:
1. Manage credentials via environment variables
2. Create model instances for available providers
3. Apply `with_fallbacks()` using LangChain's method
4. Provide convenient factory functions for common patterns

That's it. No need to reinvent the wheel.
