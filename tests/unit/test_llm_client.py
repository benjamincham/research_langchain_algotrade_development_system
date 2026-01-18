"""
Unit tests for LLM client factory functions.
"""

import pytest
import os
from unittest.mock import Mock, patch
from src.core.llm_client import (
    create_llm_with_fallbacks,
    create_cheap_llm,
    create_powerful_llm,
    _create_model_for_provider
)
from src.config.llm_credentials import reset_credentials


class TestLLMClient:
    """Test suite for LLM client factory functions."""
    
    def setup_method(self):
        """Clear environment variables and reset credentials before each test."""
        reset_credentials()
        for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY"]:
            if key in os.environ:
                del os.environ[key]
    
    def test_no_providers_configured(self):
        """Test that error is raised when no providers configured."""
        with pytest.raises(ValueError, match="No LLM providers configured"):
            create_llm_with_fallbacks()
    
    @patch('src.core.llm_client._create_model_for_provider')
    def test_single_provider(self, mock_create):
        """Test LLM creation with single provider."""
        os.environ["OPENAI_API_KEY"] = "sk-test-key"
        
        # Mock model creation
        mock_model = Mock()
        mock_create.return_value = mock_model
        
        llm = create_llm_with_fallbacks()
        
        # Should return the mock model directly (no fallbacks)
        assert llm == mock_model
        assert mock_create.call_count == 1
    
    @patch('src.core.llm_client._create_model_for_provider')
    def test_multiple_providers(self, mock_create):
        """Test LLM creation with multiple providers (fallbacks)."""
        os.environ["OPENAI_API_KEY"] = "sk-test-key"
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test-key"
        
        # Mock model creation
        mock_primary = Mock()
        mock_fallback = Mock()
        mock_primary.with_fallbacks = Mock(return_value=mock_primary)
        mock_create.side_effect = [mock_primary, mock_fallback]
        
        llm = create_llm_with_fallbacks()
        
        # Should call with_fallbacks
        mock_primary.with_fallbacks.assert_called_once()
        assert mock_create.call_count == 2
    
    @patch('src.core.llm_client._create_model_for_provider')
    def test_preferred_provider(self, mock_create):
        """Test that preferred provider is used first."""
        os.environ["OPENAI_API_KEY"] = "sk-test-key"
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test-key"
        
        # Mock model creation
        mock_model = Mock()
        mock_model.with_fallbacks = Mock(return_value=mock_model)
        mock_create.return_value = mock_model
        
        llm = create_llm_with_fallbacks(preferred_provider="anthropic")
        
        # First call should be for anthropic
        first_call_provider = mock_create.call_args_list[0][0][0]
        assert first_call_provider == "anthropic"
    
    @patch('src.core.llm_client._create_model_for_provider')
    def test_create_cheap_llm(self, mock_create):
        """Test cost-optimized LLM creation."""
        os.environ["OPENAI_API_KEY"] = "sk-test-key"
        os.environ["GOOGLE_API_KEY"] = "AIza-test-key"
        
        # Mock model creation
        mock_model = Mock()
        mock_model.with_fallbacks = Mock(return_value=mock_model)
        mock_create.return_value = mock_model
        
        llm = create_cheap_llm()
        
        # First call should be for google (cheapest)
        first_call_provider = mock_create.call_args_list[0][0][0]
        assert first_call_provider == "google"
    
    @patch('src.core.llm_client._create_model_for_provider')
    def test_create_powerful_llm(self, mock_create):
        """Test capability-optimized LLM creation."""
        os.environ["OPENAI_API_KEY"] = "sk-test-key"
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test-key"
        
        # Mock model creation
        mock_model = Mock()
        mock_model.with_fallbacks = Mock(return_value=mock_model)
        mock_create.return_value = mock_model
        
        llm = create_powerful_llm()
        
        # First call should be for anthropic (most capable)
        first_call_provider = mock_create.call_args_list[0][0][0]
        assert first_call_provider == "anthropic"
    
    @patch('src.core.llm_client._create_model_for_provider')
    def test_provider_failure_continues(self, mock_create):
        """Test that provider failure doesn't stop configuration."""
        os.environ["OPENAI_API_KEY"] = "sk-test-key"
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test-key"
        
        # Mock: first provider fails, second succeeds
        mock_model = Mock()
        mock_create.side_effect = [Exception("Provider failed"), mock_model]
        
        llm = create_llm_with_fallbacks()
        
        # Should return the second model
        assert llm == mock_model
    
    @patch('src.core.llm_client._create_model_for_provider')
    def test_all_providers_fail(self, mock_create):
        """Test error when all providers fail."""
        os.environ["OPENAI_API_KEY"] = "sk-test-key"
        
        # Mock: provider creation fails
        mock_create.side_effect = Exception("Provider failed")
        
        with pytest.raises(ValueError, match="Failed to configure any LLM providers"):
            create_llm_with_fallbacks()
    
    def test_custom_temperature(self):
        """Test custom temperature parameter."""
        os.environ["OPENAI_API_KEY"] = "sk-test-key"
        
        with patch('src.core.llm_client._create_model_for_provider') as mock_create:
            mock_model = Mock()
            mock_create.return_value = mock_model
            
            llm = create_llm_with_fallbacks(temperature=0.3)
            
            # Check that temperature was passed
            call_args = mock_create.call_args_list[0]
            assert call_args[0][2] == 0.3  # temperature is 3rd arg
    
    def test_custom_max_tokens(self):
        """Test custom max_tokens parameter."""
        os.environ["OPENAI_API_KEY"] = "sk-test-key"
        
        with patch('src.core.llm_client._create_model_for_provider') as mock_create:
            mock_model = Mock()
            mock_create.return_value = mock_model
            
            llm = create_llm_with_fallbacks(max_tokens=8192)
            
            # Check that max_tokens was passed
            call_args = mock_create.call_args_list[0]
            assert call_args[0][3] == 8192  # max_tokens is 4th arg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
