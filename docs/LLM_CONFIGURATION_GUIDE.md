# LLM Configuration Guide

This guide explains how to configure multiple LLM providers with automatic failover using **LangChain's built-in `with_fallbacks()` method**.

## Quick Start

### 1. Copy the Example Environment File

```bash
cp .env.example .env
```

### 2. Add Your API Keys

Edit `.env` and add at least one provider's credentials:

```bash
# Minimum configuration (use free Gemini)
GOOGLE_API_KEY=AIza...

# Recommended configuration (with failover)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
```

### 3. Use in Your Code

```python
from src.core.llm_client import create_llm_with_fallbacks

# That's it! Automatic failover is built-in
llm = create_llm_with_fallbacks()

# Use like any LangChain chat model
response = llm.invoke("Analyze AAPL stock")
```

## How It Works

### LangChain's Built-in Fallbacks

LangChain provides the `with_fallbacks()` method on all Runnable objects. This is the **standard, recommended way** to implement failover.

**Example**:
```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Create models
primary = ChatOpenAI(model="gpt-4o-mini")
fallback = ChatAnthropic(model="claude-3-5-haiku-20241022")

# Add fallback using LangChain's built-in method
model = primary.with_fallbacks([fallback])

# Automatic failover on errors
response = model.invoke("Hello")
```

**Behavior**:
1. Tries `primary` model first
2. If it fails (rate limit, timeout, error), automatically tries `fallback`
3. Returns result from whichever succeeds
4. Raises exception only if all models fail

### Our Factory Functions

We provide convenient factory functions that:
1. Load credentials from environment variables
2. Create model instances for available providers
3. Apply `with_fallbacks()` automatically

```python
from src.core.llm_client import (
    create_llm_with_fallbacks,  # General purpose
    create_cheap_llm,            # Cost-optimized
    create_powerful_llm          # Performance-optimized
)
```

## Configuration Strategies

### Strategy 1: General Purpose (Recommended)

**Use Case**: Balanced reliability and cost.

```python
llm = create_llm_with_fallbacks()
```

**Behavior**:
- Tries providers in order they're configured
- Automatic failover on errors
- Works with any available providers

**Best For**:
- Production systems
- General use cases
- When you want automatic failover

### Strategy 2: Cost-Optimized

**Use Case**: Minimize costs by trying cheapest providers first.

```python
llm = create_cheap_llm()
```

**Provider Order** (cheapest first):
1. Google Gemini (free)
2. Groq ($0.59-0.79 per 1M tokens)
3. OpenAI GPT-4o-mini ($0.15-0.60 per 1M tokens)
4. Anthropic Claude Haiku ($0.80-4.00 per 1M tokens)

**Best For**:
- Development/testing
- High-volume, simple tasks
- Budget-conscious deployments

### Strategy 3: Performance-Optimized

**Use Case**: Maximum capability, tries most powerful models first.

```python
llm = create_powerful_llm()
```

**Provider Order** (most capable first):
1. Anthropic Claude
2. OpenAI GPT-4
3. Google Gemini
4. Groq Llama

**Best For**:
- Complex reasoning tasks
- Critical operations
- When quality matters more than cost

### Strategy 4: Custom Preferred Provider

**Use Case**: Prefer a specific provider but have fallbacks.

```python
llm = create_llm_with_fallbacks(preferred_provider="anthropic")
```

**Behavior**:
- Tries your preferred provider first
- Falls back to others if it fails

**Best For**:
- Provider preferences
- Testing specific providers
- Compliance requirements

## Provider-Specific Configuration

### OpenAI

```bash
OPENAI_API_KEY=sk-...
```

**Models Used**:
- `gpt-4o-mini`: Fast, cheap ($0.15/$0.60 per 1M tokens)

**Get API Key**: https://platform.openai.com/api-keys

### Anthropic (Claude)

```bash
ANTHROPIC_API_KEY=sk-ant-...
```

**Models Used**:
- `claude-3-5-haiku`: Fast, affordable ($0.80/$4.00 per 1M tokens)

**Get API Key**: https://console.anthropic.com/

### Google (Gemini)

```bash
GOOGLE_API_KEY=AIza...
```

**Models Used**:
- `gemini-2.0-flash-exp`: Free tier, 1M token context

**Get API Key**: https://aistudio.google.com/app/apikey

### Groq (Fast Inference)

```bash
GROQ_API_KEY=gsk_...
```

**Models Used**:
- `llama-3.3-70b-versatile`: Very fast, cheap ($0.59/$0.79 per 1M tokens)

**Get API Key**: https://console.groq.com/

**Note**: Groq is optimized for speed. Great for simple, high-volume tasks.

## Recommended Configurations

### For Development

**Goal**: Free or very cheap, fast iteration.

```bash
# .env
GOOGLE_API_KEY=AIza...        # Free
GROQ_API_KEY=gsk_...          # Very cheap
```

```python
llm = create_cheap_llm()
```

**Cost**: ~$0.00 - $0.50 per day

### For Production (Reliability)

**Goal**: Maximum uptime, automatic failover.

```bash
# .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
GROQ_API_KEY=gsk_...
```

```python
llm = create_llm_with_fallbacks()
```

**Cost**: ~$5 - $20 per day (depending on volume)

### For Production (Cost-Optimized)

**Goal**: Minimize costs while maintaining reliability.

```bash
# .env
GROQ_API_KEY=gsk_...          # Primary (cheapest)
GOOGLE_API_KEY=AIza...        # Backup (free)
OPENAI_API_KEY=sk-...         # Emergency fallback
```

```python
llm = create_cheap_llm()
```

**Cost**: ~$1 - $5 per day

### For Production (Performance)

**Goal**: Best quality, cost is secondary.

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...  # Primary (most capable)
OPENAI_API_KEY=sk-...         # Fallback
```

```python
llm = create_powerful_llm()
```

**Cost**: ~$10 - $30 per day

## Failover Behavior

### What Triggers Failover?

The system automatically fails over when:

1. **API Errors**:
   - Rate limit exceeded (429)
   - Authentication failure (401)
   - Service unavailable (503)
   - Timeout

2. **Model Errors**:
   - Model not found
   - Invalid request format
   - Context length exceeded

3. **Network Errors**:
   - Connection timeout
   - DNS resolution failure
   - SSL errors

### Failover Logging

LangChain logs all failover events:

```
INFO: Configuring LLM with providers: ['openai', 'anthropic', 'google']
INFO: Primary: openai, Fallbacks: ['anthropic', 'google']
WARNING: Error in ChatOpenAI: Rate limit exceeded
INFO: Falling back to ChatAnthropic
```

## Testing Your Configuration

### Test Script

```python
# test_llm_config.py

from src.core.llm_client import create_llm_with_fallbacks
from src.config.llm_credentials import get_credentials

# Check available providers
credentials = get_credentials()
available = credentials.get_available_providers()
print(f"Available providers: {available}")

if not available:
    print("❌ No providers configured! Add API keys to .env file.")
    exit(1)

# Test LLM
llm = create_llm_with_fallbacks()
try:
    response = llm.invoke("Say 'Hello, World!' in one word.")
    print(f"✅ LLM working! Response: {response.content}")
except Exception as e:
    print(f"❌ LLM failed: {e}")
```

Run it:
```bash
python test_llm_config.py
```

### Expected Output

```
Available providers: ['openai', 'anthropic', 'google']
INFO: Configuring LLM with providers: ['openai', 'anthropic', 'google']
INFO: Primary: openai, Fallbacks: ['anthropic', 'google']
✅ LLM working! Response: Hello!
```

## Advanced Usage

### Custom Temperature and Max Tokens

```python
llm = create_llm_with_fallbacks(
    temperature=0.3,      # Lower = more deterministic
    max_tokens=8192       # Longer responses
)
```

### Using in Agents

```python
from src.core.base_agent import BaseAgent
from src.core.llm_client import create_llm_with_fallbacks

class ResearchAgent(BaseAgent):
    def __init__(self):
        llm = create_llm_with_fallbacks()
        super().__init__(
            name="ResearchAgent",
            role="Market Research",
            llm=llm
        )
```

### Manual Fallback Configuration

If you need more control:

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Create models manually
primary = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
fallback1 = ChatAnthropic(model="claude-3-5-haiku-20241022")
fallback2 = ChatOpenAI(model="gpt-3.5-turbo")

# Apply fallbacks manually
llm = primary.with_fallbacks([fallback1, fallback2])
```

## Troubleshooting

### Problem: "No LLM providers configured"

**Cause**: No valid API keys in `.env` file.

**Solution**:
1. Check that `.env` file exists
2. Verify API keys are correct (no extra spaces, quotes)
3. Ensure at least one provider's credentials are set

### Problem: "All models failed"

**Cause**: All configured providers are unavailable or rate-limited.

**Solution**:
1. Check provider status pages (status.openai.com, etc.)
2. Verify API keys are still valid
3. Check rate limits on provider dashboards
4. Add more providers for better redundancy

### Problem: High costs

**Cause**: Using expensive models or high volume.

**Solution**:
1. Use `create_cheap_llm()` instead
2. Add free providers (Google Gemini)
3. Monitor costs via provider dashboards
4. Implement rate limiting in your application

## Security Best Practices

### 1. Never Commit .env Files

```bash
# .gitignore already includes:
.env
.env.local
.env.*.local
```

### 2. Use Different Keys for Dev/Prod

```bash
# .env.development
OPENAI_API_KEY=sk-dev-...

# .env.production
OPENAI_API_KEY=sk-prod-...
```

### 3. Rotate Keys Regularly

- Set calendar reminders to rotate API keys every 90 days
- Use provider dashboards to revoke old keys

### 4. Use Secrets Management in Production

Instead of `.env` files, use:
- **AWS Secrets Manager**
- **Azure Key Vault**
- **HashiCorp Vault**
- **Kubernetes Secrets**

## Cost Estimation

### Per-Agent Costs (Approximate)

| Agent Type | Tokens/Call | Calls/Day | Cost/Day (GPT-4o-mini) | Cost/Day (Gemini) |
|------------|-------------|-----------|------------------------|-------------------|
| Research Leader | 2000 | 10 | $0.03 | $0.00 |
| Technical Subagent | 1000 | 50 | $0.08 | $0.00 |
| Strategy Generator | 3000 | 5 | $0.02 | $0.00 |
| Quality Gate | 500 | 20 | $0.02 | $0.00 |

**Total System Cost/Day**:
- With GPT-4o-mini: ~$5 - $10
- With Gemini: ~$0 (free tier)
- With Groq: ~$1 - $3

## Why This Approach?

### Advantages of Using LangChain's Built-in Fallbacks

1. **No Reinventing the Wheel**: Uses battle-tested LangChain features
2. **Simple**: ~100 lines of code vs ~500 for custom solution
3. **Maintainable**: LangChain team maintains the core logic
4. **Well-Documented**: Official LangChain documentation
5. **Community Support**: Large ecosystem and community
6. **Future-Proof**: Automatic updates with LangChain releases

### What We Add

Our implementation adds:
- Credential management via environment variables
- Convenient factory functions for common patterns
- Provider ordering strategies (cost, performance)
- Configuration validation
- Logging and monitoring

## Next Steps

1. Copy `.env.example` to `.env`
2. Add your API keys
3. Run `python test_llm_config.py` to verify
4. Use `create_llm_with_fallbacks()` in your agents
5. Monitor costs and adjust as needed
