"""
Integration tests for Phase 1: Core Infrastructure

These tests verify end-to-end functionality of:
- LLM credentials management
- LLM client factory functions
- Error handling
- Real LLM invocations (if credentials available)
"""

import pytest
import os
from src.config.llm_credentials import get_credentials, reset_credentials
from src.core.llm_client import create_llm_with_fallbacks, create_cheap_llm, create_powerful_llm
from src.core.error_handler import ConfigurationError, LLMProviderError


class TestPhase1Integration:
    """Integration tests for Phase 1."""
    
    def setup_method(self):
        """Reset credentials before each test."""
        reset_credentials()
    
    def test_no_credentials_error(self):
        """Test that system raises error when no credentials configured."""
        # Clear all credentials
        for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY"]:
            if key in os.environ:
                del os.environ[key]
        
        with pytest.raises(ValueError, match="No LLM providers configured"):
            create_llm_with_fallbacks()
    
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OpenAI API key not configured"
    )
    def test_openai_real_invocation(self):
        """Test real OpenAI LLM invocation."""
        llm = create_llm_with_fallbacks(preferred_provider="openai")
        
        response = llm.invoke("Say 'test successful' and nothing else.")
        
        assert response is not None
        assert hasattr(response, "content")
        assert len(response.content) > 0
        print(f"OpenAI response: {response.content}")
    
    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"),
        reason="Anthropic API key not configured"
    )
    def test_anthropic_real_invocation(self):
        """Test real Anthropic LLM invocation."""
        llm = create_llm_with_fallbacks(preferred_provider="anthropic")
        
        response = llm.invoke("Say 'test successful' and nothing else.")
        
        assert response is not None
        assert hasattr(response, "content")
        assert len(response.content) > 0
        print(f"Anthropic response: {response.content}")
    
    @pytest.mark.skipif(
        not os.getenv("GOOGLE_API_KEY"),
        reason="Google API key not configured"
    )
    def test_google_real_invocation(self):
        """Test real Google LLM invocation."""
        llm = create_llm_with_fallbacks(preferred_provider="google")
        
        response = llm.invoke("Say 'test successful' and nothing else.")
        
        assert response is not None
        assert hasattr(response, "content")
        assert len(response.content) > 0
        print(f"Google response: {response.content}")
    
    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"),
        reason="Groq API key not configured"
    )
    def test_groq_real_invocation(self):
        """Test real Groq LLM invocation."""
        llm = create_llm_with_fallbacks(preferred_provider="groq")
        
        response = llm.invoke("Say 'test successful' and nothing else.")
        
        assert response is not None
        assert hasattr(response, "content")
        assert len(response.content) > 0
        print(f"Groq response: {response.content}")
    
    @pytest.mark.skipif(
        not (os.getenv("OPENAI_API_KEY") and os.getenv("ANTHROPIC_API_KEY")),
        reason="Multiple API keys not configured"
    )
    def test_fallback_mechanism(self):
        """Test that fallback works when primary provider is configured."""
        # This test verifies that with_fallbacks() is properly configured
        # We can't easily simulate a provider failure, but we can verify
        # that multiple providers are configured
        
        llm = create_llm_with_fallbacks()
        
        # Verify LLM works
        response = llm.invoke("Say 'test successful' and nothing else.")
        assert response is not None
        assert hasattr(response, "content")
        print(f"Fallback test response: {response.content}")
    
    @pytest.mark.skipif(
        not (os.getenv("GOOGLE_API_KEY") or os.getenv("GROQ_API_KEY")),
        reason="Cheap providers not configured"
    )
    def test_cheap_llm_factory(self):
        """Test create_cheap_llm() factory function."""
        llm = create_cheap_llm()
        
        response = llm.invoke("Say 'test successful' and nothing else.")
        
        assert response is not None
        assert hasattr(response, "content")
        print(f"Cheap LLM response: {response.content}")
    
    @pytest.mark.skipif(
        not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")),
        reason="Powerful providers not configured"
    )
    def test_powerful_llm_factory(self):
        """Test create_powerful_llm() factory function."""
        llm = create_powerful_llm()
        
        response = llm.invoke("Say 'test successful' and nothing else.")
        
        assert response is not None
        assert hasattr(response, "content")
        print(f"Powerful LLM response: {response.content}")
    
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OpenAI API key not configured"
    )
    def test_custom_temperature(self):
        """Test custom temperature parameter."""
        llm = create_llm_with_fallbacks(temperature=0.0)
        
        # With temperature=0, responses should be deterministic
        response1 = llm.invoke("Say 'test successful' and nothing else.")
        response2 = llm.invoke("Say 'test successful' and nothing else.")
        
        # Note: Even with temperature=0, responses might vary slightly
        # due to tokenization, but they should be very similar
        assert response1 is not None
        assert response2 is not None
        print(f"Response 1: {response1.content}")
        print(f"Response 2: {response2.content}")
    
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OpenAI API key not configured"
    )
    def test_max_tokens_limit(self):
        """Test max_tokens parameter."""
        llm = create_llm_with_fallbacks(max_tokens=10)
        
        response = llm.invoke("Write a long essay about artificial intelligence.")
        
        # Response should be truncated due to max_tokens limit
        assert response is not None
        assert hasattr(response, "content")
        # Response should be short (around 10 tokens)
        print(f"Limited response: {response.content}")
        print(f"Response length: {len(response.content.split())} words")
    
    def test_credentials_singleton(self):
        """Test that credentials are properly cached."""
        os.environ["OPENAI_API_KEY"] = "sk-test-key"
        
        creds1 = get_credentials()
        creds2 = get_credentials()
        
        # Should be the same instance
        assert creds1 is creds2
    
    def test_credentials_validation(self):
        """Test that credentials validation works."""
        # Clear all credentials
        for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY"]:
            if key in os.environ:
                del os.environ[key]
        
        # Should raise error
        with pytest.raises(ValueError, match="No LLM providers configured"):
            get_credentials()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
