# LLM Routing System Design: Provider-Agnostic with Seamless Failover

## Problem Statement

In a production agentic system, relying on a single LLM provider creates critical vulnerabilities:

1. **Provider Outages**: OpenAI, Anthropic, or any provider can experience downtime (e.g., OpenAI's multi-day outage in Nov 2023)
2. **Rate Limiting**: Hitting rate limits halts the entire system
3. **Cost Optimization**: Different providers have different pricing; routing to cheaper models for simple tasks saves money
4. **Model Availability**: Specific models may be unavailable or deprecated
5. **Hard-Coded Credentials**: Hard-coding API keys and provider logic makes the system inflexible and insecure

**Goal**: Design a provider-agnostic LLM routing system that:
- Supports multiple LLM providers seamlessly
- Manages credentials via configuration (no hard-coding)
- Implements automatic failover on errors
- Enables dynamic routing based on cost, latency, and availability
- Remains transparent to the agent system

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        LLM ROUTING SYSTEM                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                  CREDENTIAL MANAGER                              │   │
│  │  • Load credentials from environment variables                   │   │
│  │  • Support .env files and secrets management                     │   │
│  │  • Validate credentials on startup                               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                  PROVIDER REGISTRY                               │   │
│  │  • Register available LLM providers                              │   │
│  │  • Map provider configs to LangChain chat models                 │   │
│  │  • Track provider health and availability                        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                  ROUTING STRATEGY                                │   │
│  │  • Fallback: Sequential failover on errors                       │   │
│  │  • Load Balancing: Distribute across providers                   │   │
│  │  • Cost-Aware: Route to cheapest provider first                  │   │
│  │  • Dynamic: Runtime selection via config                         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                  UNIFIED LLM INTERFACE                           │   │
│  │  • Single interface for all agents                               │   │
│  │  • Transparent routing and failover                              │   │
│  │  • Logging and observability                                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Component 1: Credential Manager

### Environment-Based Credential Management

**Principle**: All credentials are loaded from environment variables or `.env` files. No hard-coding.

```python
# config/llm_credentials.py

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, Dict
import os

class LLMProviderCredentials(BaseSettings):
    """Credentials for all LLM providers."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # OpenAI
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    openai_org_id: Optional[str] = Field(None, env="OPENAI_ORG_ID")
    openai_base_url: Optional[str] = Field(None, env="OPENAI_BASE_URL")
    
    # Anthropic
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    
    # Google (Gemini)
    google_api_key: Optional[str] = Field(None, env="GOOGLE_API_KEY")
    
    # Azure OpenAI
    azure_openai_api_key: Optional[str] = Field(None, env="AZURE_OPENAI_API_KEY")
    azure_openai_endpoint: Optional[str] = Field(None, env="AZURE_OPENAI_ENDPOINT")
    azure_openai_api_version: Optional[str] = Field(None, env="AZURE_OPENAI_API_VERSION")
    
    # AWS Bedrock
    aws_access_key_id: Optional[str] = Field(None, env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(None, env="AWS_SECRET_ACCESS_KEY")
    aws_region: Optional[str] = Field(None, env="AWS_REGION")
    
    # Groq
    groq_api_key: Optional[str] = Field(None, env="GROQ_API_KEY")
    
    # Together AI
    together_api_key: Optional[str] = Field(None, env="TOGETHER_API_KEY")
    
    def get_available_providers(self) -> list[str]:
        """Return list of providers with valid credentials."""
        available = []
        
        if self.openai_api_key:
            available.append("openai")
        if self.anthropic_api_key:
            available.append("anthropic")
        if self.google_api_key:
            available.append("google")
        if self.azure_openai_api_key and self.azure_openai_endpoint:
            available.append("azure_openai")
        if self.aws_access_key_id and self.aws_secret_access_key:
            available.append("bedrock")
        if self.groq_api_key:
            available.append("groq")
        if self.together_api_key:
            available.append("together")
        
        return available
    
    def validate_provider(self, provider: str) -> bool:
        """Validate that credentials exist for a provider."""
        return provider in self.get_available_providers()


# Singleton instance
_credentials: Optional[LLMProviderCredentials] = None

def get_credentials() -> LLMProviderCredentials:
    """Get or create credential manager singleton."""
    global _credentials
    if _credentials is None:
        _credentials = LLMProviderCredentials()
    return _credentials
```

### Example .env File

```bash
# .env file (NOT committed to git)

# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_ORG_ID=org-...

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Google Gemini
GOOGLE_API_KEY=AIza...

# Azure OpenAI
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview

# AWS Bedrock
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1

# Groq
GROQ_API_KEY=gsk_...

# Together AI
TOGETHER_API_KEY=...
```

## Component 2: Provider Registry

### Provider Configuration Schema

```python
# config/llm_providers.py

from pydantic import BaseModel
from typing import Literal, Optional
from enum import Enum

class ProviderType(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE_OPENAI = "azure_openai"
    BEDROCK = "bedrock"
    GROQ = "groq"
    TOGETHER = "together"

class ModelConfig(BaseModel):
    """Configuration for a specific model."""
    
    provider: ProviderType
    model_name: str
    
    # Performance characteristics
    max_tokens: int = 4096
    temperature: float = 0.7
    
    # Cost (per 1M tokens)
    input_cost: float  # USD per 1M input tokens
    output_cost: float  # USD per 1M output tokens
    
    # Capabilities
    supports_function_calling: bool = True
    supports_streaming: bool = True
    context_window: int = 8192
    
    # Routing preferences
    priority: int = 1  # Lower = higher priority
    enabled: bool = True

class ProviderRegistry:
    """Registry of available LLM providers and models."""
    
    # Default model configurations
    DEFAULT_MODELS = {
        # OpenAI
        "gpt-4o": ModelConfig(
            provider=ProviderType.OPENAI,
            model_name="gpt-4o",
            input_cost=2.50,
            output_cost=10.00,
            context_window=128000,
            priority=2
        ),
        "gpt-4o-mini": ModelConfig(
            provider=ProviderType.OPENAI,
            model_name="gpt-4o-mini",
            input_cost=0.15,
            output_cost=0.60,
            context_window=128000,
            priority=1
        ),
        
        # Anthropic
        "claude-3-5-sonnet": ModelConfig(
            provider=ProviderType.ANTHROPIC,
            model_name="claude-3-5-sonnet-20241022",
            input_cost=3.00,
            output_cost=15.00,
            context_window=200000,
            priority=2
        ),
        "claude-3-5-haiku": ModelConfig(
            provider=ProviderType.ANTHROPIC,
            model_name="claude-3-5-haiku-20241022",
            input_cost=0.80,
            output_cost=4.00,
            context_window=200000,
            priority=1
        ),
        
        # Google
        "gemini-2.0-flash": ModelConfig(
            provider=ProviderType.GOOGLE,
            model_name="gemini-2.0-flash-exp",
            input_cost=0.00,  # Free tier
            output_cost=0.00,
            context_window=1000000,
            priority=1
        ),
        
        # Groq (very fast, cheap)
        "llama-3.3-70b": ModelConfig(
            provider=ProviderType.GROQ,
            model_name="llama-3.3-70b-versatile",
            input_cost=0.59,
            output_cost=0.79,
            context_window=32768,
            priority=1
        )
    }
    
    def __init__(self, credentials: LLMProviderCredentials):
        self.credentials = credentials
        self.models = self._filter_available_models()
    
    def _filter_available_models(self) -> dict[str, ModelConfig]:
        """Filter models to only those with available credentials."""
        available_providers = self.credentials.get_available_providers()
        
        return {
            name: config 
            for name, config in self.DEFAULT_MODELS.items()
            if config.provider.value in available_providers and config.enabled
        }
    
    def get_models_by_priority(self) -> list[tuple[str, ModelConfig]]:
        """Get models sorted by priority (lower = higher priority)."""
        return sorted(
            self.models.items(),
            key=lambda x: (x[1].priority, x[1].input_cost)
        )
    
    def get_cheapest_model(self) -> Optional[tuple[str, ModelConfig]]:
        """Get the cheapest available model."""
        if not self.models:
            return None
        
        return min(
            self.models.items(),
            key=lambda x: x[1].input_cost + x[1].output_cost
        )
```

## Component 3: Unified LLM Interface

### Provider-Agnostic Chat Model

```python
# core/llm_router.py

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Optional, List, Any
import logging

logger = logging.getLogger(__name__)

class RoutedChatModel(BaseChatModel):
    """
    Provider-agnostic chat model with automatic failover.
    
    This class provides a single interface for all agents while
    transparently handling provider routing and failover.
    """
    
    def __init__(
        self,
        strategy: Literal["fallback", "cost_optimized", "dynamic"] = "fallback",
        custom_model_order: Optional[list[str]] = None
    ):
        super().__init__()
        
        self.credentials = get_credentials()
        self.registry = ProviderRegistry(self.credentials)
        self.strategy = strategy
        self.custom_model_order = custom_model_order
        
        # Initialize model instances
        self.model_instances = self._initialize_models()
        
        if not self.model_instances:
            raise ValueError(
                "No LLM providers available. Please configure credentials in .env file."
            )
        
        logger.info(f"Initialized RoutedChatModel with {len(self.model_instances)} providers")
    
    def _initialize_models(self) -> dict[str, BaseChatModel]:
        """Initialize LangChain chat model instances for each provider."""
        instances = {}
        
        for model_name, config in self.registry.models.items():
            try:
                instance = self._create_model_instance(model_name, config)
                instances[model_name] = instance
                logger.info(f"Initialized model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize {model_name}: {e}")
        
        return instances
    
    def _create_model_instance(
        self, 
        model_name: str, 
        config: ModelConfig
    ) -> BaseChatModel:
        """Create a LangChain chat model instance based on provider."""
        
        if config.provider == ProviderType.OPENAI:
            return ChatOpenAI(
                model=config.model_name,
                api_key=self.credentials.openai_api_key,
                base_url=self.credentials.openai_base_url,
                organization=self.credentials.openai_org_id,
                temperature=config.temperature,
                max_tokens=config.max_tokens
            )
        
        elif config.provider == ProviderType.ANTHROPIC:
            return ChatAnthropic(
                model=config.model_name,
                api_key=self.credentials.anthropic_api_key,
                temperature=config.temperature,
                max_tokens=config.max_tokens
            )
        
        elif config.provider == ProviderType.GOOGLE:
            return ChatGoogleGenerativeAI(
                model=config.model_name,
                google_api_key=self.credentials.google_api_key,
                temperature=config.temperature,
                max_output_tokens=config.max_tokens
            )
        
        # Add more providers as needed
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")
    
    def _get_model_order(self) -> list[str]:
        """Determine the order of models to try based on strategy."""
        
        if self.custom_model_order:
            # Use custom order if provided
            return [m for m in self.custom_model_order if m in self.model_instances]
        
        if self.strategy == "cost_optimized":
            # Try cheapest models first
            return [
                name for name, _ in 
                sorted(
                    self.registry.models.items(),
                    key=lambda x: x[1].input_cost + x[1].output_cost
                )
                if name in self.model_instances
            ]
        
        elif self.strategy == "fallback":
            # Try by priority
            return [
                name for name, _ in self.registry.get_models_by_priority()
                if name in self.model_instances
            ]
        
        else:  # dynamic
            # Can be changed at runtime via environment variable
            dynamic_model = os.getenv("DYNAMIC_LLM_MODEL")
            if dynamic_model and dynamic_model in self.model_instances:
                return [dynamic_model] + [
                    m for m in self.model_instances.keys() 
                    if m != dynamic_model
                ]
            else:
                return list(self.model_instances.keys())
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Generate response with automatic failover.
        
        Tries models in order until one succeeds.
        """
        model_order = self._get_model_order()
        last_exception = None
        
        for i, model_name in enumerate(model_order):
            try:
                logger.debug(f"Attempting request with model: {model_name}")
                
                model = self.model_instances[model_name]
                result = model._generate(
                    messages=messages,
                    stop=stop,
                    run_manager=run_manager,
                    **kwargs
                )
                
                if i > 0:
                    logger.info(f"Successfully failed over to model: {model_name}")
                
                return result
            
            except Exception as e:
                logger.warning(f"Model {model_name} failed: {str(e)}")
                last_exception = e
                continue
        
        # All models failed
        raise Exception(
            f"All {len(model_order)} models failed. Last error: {str(last_exception)}"
        )
    
    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "routed-chat"
```

## Component 4: Usage in Agent System

### Transparent Integration

```python
# Agents use the RoutedChatModel exactly like any LangChain chat model

from src.core.llm_router import RoutedChatModel

# Option 1: Fallback strategy (default)
llm = RoutedChatModel(strategy="fallback")

# Option 2: Cost-optimized (cheapest first)
llm = RoutedChatModel(strategy="cost_optimized")

# Option 3: Custom order
llm = RoutedChatModel(
    strategy="fallback",
    custom_model_order=["gpt-4o-mini", "claude-3-5-haiku", "gemini-2.0-flash"]
)

# Option 4: Dynamic (runtime selection via env var)
llm = RoutedChatModel(strategy="dynamic")

# Use in agents - completely transparent
from src.core.base_agent import BaseAgent

class MyAgent(BaseAgent):
    def __init__(self):
        llm = RoutedChatModel(strategy="fallback")
        super().__init__(
            name="MyAgent",
            role="Research Agent",
            llm=llm
        )
```

### Dynamic Runtime Switching

```python
# Change model at runtime without code changes
import os

# Set environment variable
os.environ["DYNAMIC_LLM_MODEL"] = "claude-3-5-sonnet"

# The next invocation will use Claude
llm = RoutedChatModel(strategy="dynamic")
response = llm.invoke("What is the capital of France?")
```

## Benefits

### 1. No Hard-Coding
- All credentials in environment variables or `.env` files
- Provider logic abstracted behind unified interface
- Easy to add new providers without changing agent code

### 2. Automatic Failover
- If OpenAI fails, automatically tries Anthropic, then Google, etc.
- Transparent to agents - they just see a successful response
- Logs all failover events for monitoring

### 3. Cost Optimization
- Route simple queries to cheap models (Groq, Gemini)
- Route complex queries to powerful models (GPT-4, Claude)
- Track costs per provider

### 4. Flexibility
- Change providers at runtime via environment variables
- Custom routing strategies per agent
- Easy to test different providers

### 5. Production-Ready
- Comprehensive logging
- Error handling with retries
- Health monitoring
- Observability hooks

## Configuration Examples

### Development Environment

```bash
# .env.development

# Use free/cheap models for development
GOOGLE_API_KEY=AIza...
GROQ_API_KEY=gsk_...
```

### Production Environment

```bash
# .env.production

# Use reliable, high-quality models
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://...
```

### Cost-Conscious Setup

```bash
# .env.cost_optimized

# Prioritize cheapest models
GROQ_API_KEY=gsk_...
GOOGLE_API_KEY=AIza...
OPENAI_API_KEY=sk-...  # Fallback only
```

## Implementation Checklist

- [ ] Implement `LLMProviderCredentials` with pydantic-settings
- [ ] Create `ProviderRegistry` with model configurations
- [ ] Implement `RoutedChatModel` with failover logic
- [ ] Add support for all major providers (OpenAI, Anthropic, Google, Azure, Bedrock, Groq)
- [ ] Add logging and observability
- [ ] Create `.env.example` file
- [ ] Update `BaseAgent` to use `RoutedChatModel` by default
- [ ] Add unit tests for failover scenarios
- [ ] Add integration tests with mock providers
- [ ] Document configuration in README

## Next Steps

1. Implement credential manager and provider registry
2. Implement `RoutedChatModel` with fallback strategy
3. Test with multiple providers
4. Add cost tracking and monitoring
5. Add health checks for providers
6. Implement load balancing strategy
