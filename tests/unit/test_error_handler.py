"""
Unit tests for error handling system.
"""

import pytest
from src.core.error_handler import (
    AlgoTradeError,
    ConfigurationError,
    LLMProviderError,
    ValidationError,
    MemoryError,
    AgentError,
    ToolError,
    handle_error,
    format_error_message,
    log_error_with_context,
    ErrorContext
)


class TestCustomExceptions:
    """Test suite for custom exception classes."""
    
    def test_algotrade_error_basic(self):
        """Test basic AlgoTradeError."""
        error = AlgoTradeError("Test error")
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.details == {}
    
    def test_algotrade_error_with_details(self):
        """Test AlgoTradeError with details."""
        error = AlgoTradeError("Test error", details={"key": "value", "count": 5})
        assert "Test error" in str(error)
        assert "key=value" in str(error)
        assert "count=5" in str(error)
    
    def test_configuration_error(self):
        """Test ConfigurationError."""
        error = ConfigurationError("Missing API key")
        assert isinstance(error, AlgoTradeError)
        assert str(error) == "Missing API key"
    
    def test_llm_provider_error(self):
        """Test LLMProviderError."""
        error = LLMProviderError("Provider failed", details={"provider": "openai"})
        assert isinstance(error, AlgoTradeError)
        assert "Provider failed" in str(error)
        assert "provider=openai" in str(error)
    
    def test_validation_error(self):
        """Test ValidationError."""
        error = ValidationError("Invalid input")
        assert isinstance(error, AlgoTradeError)
        assert str(error) == "Invalid input"
    
    def test_memory_error(self):
        """Test MemoryError."""
        error = MemoryError("Database connection failed")
        assert isinstance(error, AlgoTradeError)
        assert str(error) == "Database connection failed"
    
    def test_agent_error(self):
        """Test AgentError."""
        error = AgentError("Agent timeout")
        assert isinstance(error, AlgoTradeError)
        assert str(error) == "Agent timeout"
    
    def test_tool_error(self):
        """Test ToolError."""
        error = ToolError("Tool not found")
        assert isinstance(error, AlgoTradeError)
        assert str(error) == "Tool not found"


class TestHandleError:
    """Test suite for handle_error function."""
    
    def test_handle_error_with_raise(self):
        """Test handle_error re-raises exception."""
        error = ValueError("Test error")
        
        with pytest.raises(ValueError, match="Test error"):
            handle_error(error, "Test context", raise_exception=True)
    
    def test_handle_error_without_raise(self):
        """Test handle_error doesn't re-raise exception."""
        error = ValueError("Test error")
        
        result = handle_error(error, "Test context", raise_exception=False)
        assert result == error
    
    def test_handle_error_with_algotrade_error(self):
        """Test handle_error with AlgoTradeError."""
        error = ConfigurationError("Config error", details={"key": "value"})
        
        result = handle_error(error, "Test context", raise_exception=False)
        assert result == error


class TestFormatErrorMessage:
    """Test suite for format_error_message function."""
    
    def test_format_configuration_error_user_friendly(self):
        """Test formatting ConfigurationError (user-friendly)."""
        error = ConfigurationError("Missing API key")
        message = format_error_message(error, user_friendly=True)
        
        assert "Configuration Error" in message
        assert "Missing API key" in message
        assert ".env" in message
    
    def test_format_configuration_error_technical(self):
        """Test formatting ConfigurationError (technical)."""
        error = ConfigurationError("Missing API key")
        message = format_error_message(error, user_friendly=False)
        
        assert message == "Missing API key"
    
    def test_format_llm_provider_error_user_friendly(self):
        """Test formatting LLMProviderError (user-friendly)."""
        error = LLMProviderError("Provider failed")
        message = format_error_message(error, user_friendly=True)
        
        assert "LLM Provider Error" in message
        assert "Provider failed" in message
        assert "API keys" in message
    
    def test_format_validation_error_user_friendly(self):
        """Test formatting ValidationError (user-friendly)."""
        error = ValidationError("Invalid input")
        message = format_error_message(error, user_friendly=True)
        
        assert "Validation Error" in message
        assert "Invalid input" in message
    
    def test_format_memory_error_user_friendly(self):
        """Test formatting MemoryError (user-friendly)."""
        error = MemoryError("Database failed")
        message = format_error_message(error, user_friendly=True)
        
        assert "Memory System Error" in message
        assert "Database failed" in message
    
    def test_format_agent_error_user_friendly(self):
        """Test formatting AgentError (user-friendly)."""
        error = AgentError("Agent timeout")
        message = format_error_message(error, user_friendly=True)
        
        assert "Agent Error" in message
        assert "Agent timeout" in message
    
    def test_format_tool_error_user_friendly(self):
        """Test formatting ToolError (user-friendly)."""
        error = ToolError("Tool not found")
        message = format_error_message(error, user_friendly=True)
        
        assert "Tool Error" in message
        assert "Tool not found" in message
    
    def test_format_generic_error_user_friendly(self):
        """Test formatting generic error (user-friendly)."""
        error = ValueError("Generic error")
        message = format_error_message(error, user_friendly=True)
        
        assert "unexpected error" in message
        assert "Generic error" in message


class TestLogErrorWithContext:
    """Test suite for log_error_with_context function."""
    
    def test_log_error_with_context(self, caplog):
        """Test logging error with context."""
        error = ValueError("Test error")
        context = {"step": "processing", "count": 5}
        
        log_error_with_context(error, context)
        
        assert "Test error" in caplog.text
        assert "step=processing" in caplog.text
        assert "count=5" in caplog.text


class TestErrorContext:
    """Test suite for ErrorContext context manager."""
    
    def test_error_context_no_error(self):
        """Test ErrorContext when no error occurs."""
        with ErrorContext("Test operation"):
            pass  # No error
    
    def test_error_context_with_raise(self):
        """Test ErrorContext re-raises exception."""
        with pytest.raises(ValueError, match="Test error"):
            with ErrorContext("Test operation", raise_on_error=True):
                raise ValueError("Test error")
    
    def test_error_context_without_raise(self):
        """Test ErrorContext suppresses exception."""
        with ErrorContext("Test operation", raise_on_error=False):
            raise ValueError("Test error")
        # Should not raise
    
    def test_error_context_logs_error(self, caplog):
        """Test ErrorContext logs error."""
        with ErrorContext("Test operation", raise_on_error=False):
            raise ValueError("Test error")
        
        assert "Test error" in caplog.text
        assert "Test operation" in caplog.text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
