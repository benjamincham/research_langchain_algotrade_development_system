"""
Unit tests for LLM credentials management.
"""

import pytest
import os
from src.config.llm_credentials import LLMCredentials, get_credentials, reset_credentials


class TestLLMCredentials:
    """Test suite for LLMCredentials class."""
    
    def setup_method(self):
        """Clear environment variables before each test."""
        reset_credentials()
        for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY"]:
            if key in os.environ:
                del os.environ[key]
    
    def test_no_credentials(self):
        """Test that no credentials returns empty list."""
        creds = LLMCredentials()
        available = creds.get_available_providers()
        assert available == []
    
    def test_openai_credentials(self):
        """Test OpenAI credentials detection."""
        os.environ["OPENAI_API_KEY"] = "sk-test-key"
        creds = LLMCredentials()
        available = creds.get_available_providers()
        assert "openai" in available
        assert len(available) == 1
    
    def test_anthropic_credentials(self):
        """Test Anthropic credentials detection."""
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test-key"
        creds = LLMCredentials()
        available = creds.get_available_providers()
        assert "anthropic" in available
        assert len(available) == 1
    
    def test_google_credentials(self):
        """Test Google credentials detection."""
        os.environ["GOOGLE_API_KEY"] = "AIza-test-key"
        creds = LLMCredentials()
        available = creds.get_available_providers()
        assert "google" in available
        assert len(available) == 1
    
    def test_groq_credentials(self):
        """Test Groq credentials detection."""
        os.environ["GROQ_API_KEY"] = "gsk_test-key"
        creds = LLMCredentials()
        available = creds.get_available_providers()
        assert "groq" in available
        assert len(available) == 1
    
    def test_multiple_credentials(self):
        """Test multiple providers configured."""
        os.environ["OPENAI_API_KEY"] = "sk-test-key"
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test-key"
        os.environ["GOOGLE_API_KEY"] = "AIza-test-key"
        
        creds = LLMCredentials()
        available = creds.get_available_providers()
        
        assert len(available) == 3
        assert "openai" in available
        assert "anthropic" in available
        assert "google" in available
    
    def test_empty_string_credentials(self):
        """Test that empty string credentials are not counted."""
        os.environ["OPENAI_API_KEY"] = ""
        os.environ["ANTHROPIC_API_KEY"] = "   "  # Whitespace only
        
        creds = LLMCredentials()
        available = creds.get_available_providers()
        
        assert available == []
    
    def test_validate_credentials_no_providers(self):
        """Test validation fails when no providers configured."""
        creds = LLMCredentials()
        
        with pytest.raises(ValueError, match="No LLM providers configured"):
            creds.validate_credentials()
    
    def test_validate_credentials_with_providers(self):
        """Test validation passes when providers configured."""
        os.environ["OPENAI_API_KEY"] = "sk-test-key"
        
        creds = LLMCredentials()
        # Should not raise
        creds.validate_credentials()
    
    def test_get_credentials_singleton(self):
        """Test get_credentials returns singleton."""
        os.environ["OPENAI_API_KEY"] = "sk-test-key"
        
        creds1 = get_credentials()
        creds2 = get_credentials()
        
        assert creds1 is creds2
    
    def test_reset_credentials(self):
        """Test reset_credentials clears singleton."""
        os.environ["OPENAI_API_KEY"] = "sk-test-key"
        
        creds1 = get_credentials()
        reset_credentials()
        creds2 = get_credentials()
        
        assert creds1 is not creds2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
