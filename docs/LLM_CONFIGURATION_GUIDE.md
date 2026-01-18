# LLM Configuration Guide

This guide explains how to configure multiple LLM providers for the trading research system with seamless failover and no hard-coding.

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
from src.core.llm_router import RoutedChatModel

# That's it! The system automatically uses available providers
llm = RoutedChatModel()

# Use like any LangChain chat model
response = llm.invoke("Analyze AAPL stock")
```

## Configuration Strategies

### Strategy 1: Fallback (Recommended for Production)

**Use Case**: Maximum reliability. If one provider fails, automatically try the next.

**Configuration**:
```python
llm = RoutedChatModel(strategy="fallback")
```

**Behavior**:
1. Tries highest priority model first (based on `priority` in `ProviderRegistry`)
2. If it fails (error, timeout, rate limit), tries next model
3. Continues until success or all models exhausted

**Example**:
```
Try: gpt-4o-mini (priority 1)
  ❌ Failed: Rate limit exceeded
Try: claude-3-5-haiku (priority 1)
  ✅ Success!
```

**Best For**:
- Production systems
- Critical operations
- When uptime is more important than cost

### Strategy 2: Cost-Optimized

**Use Case**: Minimize costs by always trying the cheapest model first.

**Configuration**:
```python
llm = RoutedChatModel(strategy="cost_optimized")
```

**Behavior**:
1. Tries cheapest model first (based on `input_cost + output_cost`)
2. Falls back to more expensive models if cheaper ones fail

**Example**:
```
Try: gemini-2.0-flash ($0.00)
  ❌ Failed: Service unavailable
Try: llama-3.3-70b ($1.38/1M tokens)
  ✅ Success!
```

**Best For**:
- Development/testing
- High-volume, low-complexity tasks
- Budget-conscious deployments

### Strategy 3: Dynamic (Runtime Selection)

**Use Case**: Change the model at runtime without code changes.

**Configuration**:
```python
llm = RoutedChatModel(strategy="dynamic")
```

**Behavior**:
1. Reads `DYNAMIC_LLM_MODEL` environment variable
2. Uses that model if available
3. Falls back to other models if specified model fails

**Example**:
```bash
# In terminal or deployment config
export DYNAMIC_LLM_MODEL=claude-3-5-sonnet

# Or in .env
DYNAMIC_LLM_MODEL=claude-3-5-sonnet
```

**Best For**:
- A/B testing different models
- Switching providers without redeployment
- Debugging specific provider issues

### Strategy 4: Custom Order

**Use Case**: Explicitly control the order of providers.

**Configuration**:
```python
llm = RoutedChatModel(
    strategy="fallback",
    custom_model_order=[
        "gpt-4o-mini",           # Try OpenAI first
        "claude-3-5-haiku",      # Then Anthropic
        "gemini-2.0-flash"       # Then Google
    ]
)
```

**Best For**:
- Specific provider preferences
- Testing provider performance
- Compliance requirements (e.g., must use specific providers)

## Provider-Specific Configuration

### OpenAI

```bash
# Standard configuration
OPENAI_API_KEY=sk-...

# Optional: Organization ID
OPENAI_ORG_ID=org-...

# Optional: Custom base URL (for proxies or Azure)
OPENAI_BASE_URL=https://your-proxy.com/v1
```

**Available Models**:
- `gpt-4o`: Most capable, expensive ($2.50/$10.00 per 1M tokens)
- `gpt-4o-mini`: Fast, cheap ($0.15/$0.60 per 1M tokens) ⭐ **Recommended**

### Anthropic (Claude)

```bash
ANTHROPIC_API_KEY=sk-ant-...
```

**Available Models**:
- `claude-3-5-sonnet`: Most capable ($3.00/$15.00 per 1M tokens)
- `claude-3-5-haiku`: Fast, cheap ($0.80/$4.00 per 1M tokens) ⭐ **Recommended**

### Google (Gemini)

```bash
GOOGLE_API_KEY=AIza...
```

**Available Models**:
- `gemini-2.0-flash`: Free tier, 1M token context ⭐ **Best for development**

### Groq (Fast Inference)

```bash
GROQ_API_KEY=gsk_...
```

**Available Models**:
- `llama-3.3-70b`: Very fast, cheap ($0.59/$0.79 per 1M tokens) ⭐ **Best for high-volume**

**Note**: Groq is optimized for speed. Great for simple tasks requiring low latency.

### Azure OpenAI

```bash
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

**Use Case**: Enterprise deployments with Azure compliance requirements.

### AWS Bedrock

```bash
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
```

**Use Case**: AWS-native deployments, Claude models via Bedrock.

## Recommended Configurations

### For Development

**Goal**: Free or very cheap, fast iteration.

```bash
# .env
GOOGLE_API_KEY=AIza...        # Free
GROQ_API_KEY=gsk_...          # Very cheap
```

```python
llm = RoutedChatModel(strategy="cost_optimized")
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
llm = RoutedChatModel(
    strategy="fallback",
    custom_model_order=[
        "gpt-4o-mini",
        "claude-3-5-haiku",
        "gemini-2.0-flash",
        "llama-3.3-70b"
    ]
)
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
llm = RoutedChatModel(strategy="cost_optimized")
```

**Cost**: ~$1 - $5 per day

### For Enterprise (Compliance)

**Goal**: Use only approved providers (e.g., Azure for data residency).

```bash
# .env
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

```python
llm = RoutedChatModel(
    strategy="fallback",
    custom_model_order=["azure-gpt-4o"]
)
```

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

### What Does NOT Trigger Failover?

- **Invalid prompts**: If your prompt is malformed, all models will fail
- **Content policy violations**: If content is blocked, it's likely blocked everywhere
- **Insufficient credits**: If you're out of credits, failover won't help

### Failover Logging

All failover events are logged:

```
INFO: Attempting request with model: gpt-4o-mini
WARNING: Model gpt-4o-mini failed: Rate limit exceeded
INFO: Attempting request with model: claude-3-5-haiku
INFO: Successfully failed over to model: claude-3-5-haiku
```

## Testing Your Configuration

### Test Script

```python
# test_llm_config.py

from src.core.llm_router import RoutedChatModel
from src.config.llm_credentials import get_credentials

# Check available providers
credentials = get_credentials()
available = credentials.get_available_providers()
print(f"Available providers: {available}")

if not available:
    print("❌ No providers configured! Add API keys to .env file.")
    exit(1)

# Test LLM
llm = RoutedChatModel(strategy="fallback")
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
✅ LLM working! Response: Hello!
```

## Troubleshooting

### Problem: "No LLM providers available"

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

### Problem: "Model X not found"

**Cause**: Model name is incorrect or not available in your region.

**Solution**:
1. Check `ProviderRegistry.DEFAULT_MODELS` for correct names
2. Verify model is available in your account
3. Update model name in configuration

### Problem: High costs

**Cause**: Using expensive models or high volume.

**Solution**:
1. Switch to `cost_optimized` strategy
2. Use cheaper models (Groq, Gemini)
3. Add rate limiting to your application
4. Monitor costs via provider dashboards

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

Example with AWS Secrets Manager:
```python
import boto3
import json

def get_secrets():
    client = boto3.client('secretsmanager', region_name='us-east-1')
    secret = client.get_secret_value(SecretId='trading-system/llm-keys')
    return json.loads(secret['SecretString'])

# Set environment variables from secrets
secrets = get_secrets()
os.environ['OPENAI_API_KEY'] = secrets['openai_api_key']
```

## Cost Estimation

### Per-Agent Costs (Approximate)

| Agent Type | Tokens/Call | Calls/Day | Cost/Day (GPT-4o-mini) | Cost/Day (Groq) |
|------------|-------------|-----------|------------------------|-----------------|
| Research Leader | 2000 | 10 | $0.03 | $0.01 |
| Technical Subagent | 1000 | 50 | $0.08 | $0.03 |
| Strategy Generator | 3000 | 5 | $0.02 | $0.01 |
| Quality Gate | 500 | 20 | $0.02 | $0.01 |

**Total System Cost/Day**:
- With GPT-4o-mini: ~$5 - $10
- With Groq: ~$1 - $3
- With Gemini: ~$0 (free tier)

## Next Steps

1. Copy `.env.example` to `.env`
2. Add your API keys
3. Run `python test_llm_config.py` to verify
4. Choose a routing strategy based on your needs
5. Monitor costs and adjust as needed
