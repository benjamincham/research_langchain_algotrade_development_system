# Phase 1: Core Infrastructure - Passing Criteria Checklist

**Phase**: Core Infrastructure  
**Status**: ⏳ In Progress (50% Complete)  
**Estimated Remaining Effort**: 2 days  
**Target Completion**: 2026-01-20

---

## Overview

Phase 1 establishes the foundational infrastructure for the entire system, including multi-provider LLM support with automatic failover, credential management, error handling, and configuration validation.

---

## Functional Objectives

### 1.1 Multi-Provider LLM Support

**Objective**: Support OpenAI, Anthropic, Google, and Groq providers

**Tasks**:
- [ ] Create `config/llm_credentials.py` with `LLMCredentials` class
- [ ] Use pydantic-settings for environment variable loading
- [ ] Support all 4 providers: OpenAI, Anthropic, Google, Groq
- [ ] Implement `get_available_providers()` method
- [ ] Validate credentials on initialization

**Acceptance Criteria**:
- [ ] Can initialize OpenAI with valid `OPENAI_API_KEY`
- [ ] Can initialize Anthropic with valid `ANTHROPIC_API_KEY`
- [ ] Can initialize Google with valid `GOOGLE_API_KEY`
- [ ] Can initialize Groq with valid `GROQ_API_KEY`
- [ ] Invalid credentials raise `ValueError` with clear message
- [ ] `get_available_providers()` returns only providers with valid credentials

**Test Cases**:
```python
# Test 1: Valid credentials
os.environ["OPENAI_API_KEY"] = "sk-test..."
creds = LLMCredentials()
assert "openai" in creds.get_available_providers()

# Test 2: Invalid credentials
os.environ["OPENAI_API_KEY"] = ""
creds = LLMCredentials()
assert "openai" not in creds.get_available_providers()

# Test 3: Multiple providers
os.environ["OPENAI_API_KEY"] = "sk-test..."
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test..."
creds = LLMCredentials()
assert len(creds.get_available_providers()) == 2
```

---

### 1.2 Automatic Failover

**Objective**: Seamless failover on provider errors using LangChain's `with_fallbacks()`

**Tasks**:
- [ ] Implement `create_llm_with_fallbacks()` in `core/llm_client.py`
- [ ] Use LangChain's native `with_fallbacks()` method
- [ ] Create model instances for all available providers
- [ ] Apply fallbacks in order of availability
- [ ] Add logging for primary and fallback providers

**Acceptance Criteria**:
- [ ] If primary provider fails, automatically tries fallback
- [ ] Failover happens transparently (no user intervention)
- [ ] Logs indicate which provider is being used
- [ ] Logs indicate when failover occurs
- [ ] Raises exception only if all providers fail

**Test Cases**:
```python
# Test 1: Primary succeeds
llm = create_llm_with_fallbacks()
response = llm.invoke("Hello")
assert response.content is not None

# Test 2: Primary fails, fallback succeeds
# (Mock primary to raise error, verify fallback is used)

# Test 3: All providers fail
# (Mock all to raise errors, verify exception is raised)
```

---

### 1.3 Credential Management

**Objective**: Load credentials from .env file with no hard-coding

**Tasks**:
- [ ] Create `.env.example` with all provider variables
- [ ] Document each environment variable
- [ ] Add `.env` to `.gitignore` (already done)
- [ ] Validate that no API keys are hard-coded in source

**Acceptance Criteria**:
- [ ] `.env.example` exists with all provider variables
- [ ] Each variable has a comment explaining its purpose
- [ ] `.env` is in `.gitignore`
- [ ] No API keys found in source code (grep check)
- [ ] Credentials load successfully from `.env` file

**Test Cases**:
```bash
# Test 1: .env.example exists
test -f .env.example

# Test 2: No hard-coded keys
! grep -r "sk-[a-zA-Z0-9]" src/

# Test 3: Credentials load from .env
cp .env.example .env
# Edit .env with test keys
python -c "from src.config.llm_credentials import get_credentials; assert len(get_credentials().get_available_providers()) > 0"
```

---

### 1.4 Factory Functions

**Objective**: Convenient LLM creation patterns

**Tasks**:
- [ ] Implement `create_llm_with_fallbacks()` (general purpose)
- [ ] Implement `create_cheap_llm()` (cost-optimized)
- [ ] Implement `create_powerful_llm()` (performance-optimized)
- [ ] Add `preferred_provider` parameter for custom ordering
- [ ] Add `temperature` and `max_tokens` parameters

**Acceptance Criteria**:
- [ ] `create_llm_with_fallbacks()` returns working LLM
- [ ] `create_cheap_llm()` prioritizes: Google → Groq → OpenAI → Anthropic
- [ ] `create_powerful_llm()` prioritizes: Anthropic → OpenAI → Google → Groq
- [ ] Can specify `preferred_provider` to override default order
- [ ] Can specify `temperature` and `max_tokens`
- [ ] All functions return `BaseChatModel` instance

**Test Cases**:
```python
# Test 1: General purpose
llm = create_llm_with_fallbacks()
assert isinstance(llm, BaseChatModel)
response = llm.invoke("Test")
assert response.content is not None

# Test 2: Cost-optimized
llm = create_cheap_llm()
# Verify Google or Groq is primary (check logs)

# Test 3: Performance-optimized
llm = create_powerful_llm()
# Verify Anthropic or OpenAI is primary (check logs)

# Test 4: Custom order
llm = create_llm_with_fallbacks(preferred_provider="anthropic")
# Verify Anthropic is primary (check logs)

# Test 5: Custom parameters
llm = create_llm_with_fallbacks(temperature=0.3, max_tokens=8192)
# Verify parameters are set (check model attributes)
```

---

### 1.5 Error Handling

**Objective**: Graceful error handling and logging

**Tasks**:
- [ ] Create `core/error_handler.py` with custom exceptions
- [ ] Define `LLMProviderError`, `ConfigurationError`, `ValidationError`
- [ ] Implement error logging utilities
- [ ] Add user-friendly error messages
- [ ] Catch and log all LLM provider errors

**Acceptance Criteria**:
- [ ] Custom exception classes defined
- [ ] All errors are logged with appropriate levels (ERROR, WARNING, INFO)
- [ ] Error messages are user-friendly (no stack traces in user output)
- [ ] Errors include suggestions for resolution
- [ ] Critical errors stop execution, non-critical are logged and continue

**Test Cases**:
```python
# Test 1: No providers configured
os.environ.clear()
with pytest.raises(ConfigurationError, match="No LLM providers configured"):
    create_llm_with_fallbacks()

# Test 2: Invalid API key
os.environ["OPENAI_API_KEY"] = "invalid"
# Should log warning and try next provider

# Test 3: All providers fail
# Mock all providers to fail
with pytest.raises(LLMProviderError, match="All .* models failed"):
    llm = create_llm_with_fallbacks()
    llm.invoke("Test")
```

---

### 1.6 Configuration Validation

**Objective**: Validate all settings on startup

**Tasks**:
- [ ] Add validation to `config/settings.py`
- [ ] Validate required fields are present
- [ ] Validate field types and formats
- [ ] Validate at least one LLM provider is configured
- [ ] Raise clear errors for invalid configuration

**Acceptance Criteria**:
- [ ] Invalid configuration raises `ConfigurationError`
- [ ] Error message specifies which field is invalid
- [ ] Error message suggests how to fix
- [ ] Validation runs on application startup
- [ ] Valid configuration passes without errors

**Test Cases**:
```python
# Test 1: Missing required field
# Remove required field from settings
with pytest.raises(ConfigurationError):
    Settings()

# Test 2: Invalid field type
# Set string field to int
with pytest.raises(ConfigurationError):
    Settings(log_level=123)

# Test 3: No LLM providers
os.environ.clear()
with pytest.raises(ConfigurationError, match="At least one LLM provider"):
    get_credentials()

# Test 4: Valid configuration
os.environ["OPENAI_API_KEY"] = "sk-test..."
creds = get_credentials()
assert len(creds.get_available_providers()) > 0
```

---

## Integration Tests

### Test 1: End-to-End LLM Invocation

**Objective**: Create agent, invoke LLM, get response

```python
def test_e2e_llm_invocation():
    # Setup
    os.environ["OPENAI_API_KEY"] = "sk-test..."
    
    # Create LLM
    llm = create_llm_with_fallbacks()
    
    # Invoke
    response = llm.invoke("What is 2+2?")
    
    # Verify
    assert response.content is not None
    assert len(response.content) > 0
```

**Acceptance Criteria**:
- [ ] Test passes with valid credentials
- [ ] Response is not empty
- [ ] Response is relevant to prompt

---

### Test 2: Failover Behavior

**Objective**: Verify automatic failover when primary fails

```python
def test_failover():
    # Setup: Configure multiple providers
    os.environ["OPENAI_API_KEY"] = "sk-test..."
    os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test..."
    
    # Create LLM
    llm = create_llm_with_fallbacks()
    
    # Mock primary to fail
    # (Implementation depends on mocking strategy)
    
    # Invoke
    response = llm.invoke("Test")
    
    # Verify fallback was used
    # (Check logs for "Falling back to" message)
```

**Acceptance Criteria**:
- [ ] Test passes when primary fails
- [ ] Fallback provider is used
- [ ] Logs indicate failover occurred

---

### Test 3: Multiple Factory Functions

**Objective**: Verify all factory functions work

```python
def test_factory_functions():
    # Setup
    os.environ["OPENAI_API_KEY"] = "sk-test..."
    os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test..."
    os.environ["GOOGLE_API_KEY"] = "AIza-test..."
    
    # Test each factory
    llm1 = create_llm_with_fallbacks()
    llm2 = create_cheap_llm()
    llm3 = create_powerful_llm()
    
    # Verify all work
    assert llm1.invoke("Test").content is not None
    assert llm2.invoke("Test").content is not None
    assert llm3.invoke("Test").content is not None
```

**Acceptance Criteria**:
- [ ] All 3 factory functions work
- [ ] Each returns a working LLM
- [ ] Each can successfully invoke and get response

---

## Documentation Requirements

### Code Documentation

- [ ] All functions have docstrings
- [ ] Docstrings include parameters, return types, and examples
- [ ] Complex logic has inline comments
- [ ] Type hints on all function signatures

### User Documentation

- [ ] README updated with setup instructions
- [ ] `.env.example` has clear comments
- [ ] LLM_CONFIGURATION_GUIDE.md is up-to-date
- [ ] Examples provided for each factory function

---

## Performance Requirements

### Latency

- [ ] LLM initialization < 1 second
- [ ] Credential loading < 100ms
- [ ] Failover decision < 500ms

### Reliability

- [ ] Failover success rate > 95%
- [ ] No memory leaks in long-running processes
- [ ] Graceful handling of all provider errors

---

## Security Requirements

- [ ] No API keys in source code
- [ ] No API keys in logs
- [ ] `.env` file in `.gitignore`
- [ ] Credentials loaded securely from environment
- [ ] No credentials exposed in error messages

---

## Phase 1 Completion Checklist

### Must Pass (Critical)

- [ ] All 4 LLM providers can be initialized
- [ ] Failover works when primary fails
- [ ] Credentials loaded from `.env` file
- [ ] `create_llm_with_fallbacks()` works
- [ ] `create_cheap_llm()` works
- [ ] `create_powerful_llm()` works
- [ ] Invalid credentials raise clear errors
- [ ] All errors are logged
- [ ] Configuration validation works
- [ ] Integration test: Create agent → Invoke LLM → Get response

### Should Pass (High Priority)

- [ ] Cost tracking for LLM calls
- [ ] Latency tracking for LLM calls
- [ ] Provider health monitoring
- [ ] Comprehensive documentation
- [ ] All unit tests pass
- [ ] All integration tests pass

### Nice to Have (Medium Priority)

- [ ] LLM call caching
- [ ] Provider performance analytics
- [ ] Automatic provider selection based on task
- [ ] Rate limiting per provider

---

## Sign-Off Criteria

Phase 1 is considered complete when:

1. ✅ All "Must Pass" items are checked
2. ✅ All integration tests pass
3. ✅ Documentation is complete and reviewed
4. ✅ Code review completed (if applicable)
5. ✅ No critical bugs or security issues

**Sign-Off Date**: _____________  
**Approved By**: _____________

---

## Next Phase

Once Phase 1 is complete, proceed to **Phase 2: Memory System**.
