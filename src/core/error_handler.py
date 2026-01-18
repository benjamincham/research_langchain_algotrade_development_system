"""
Error Handling System

This module provides custom exception classes and error handling utilities
for the Research LangChain AlgoTrade Development System.
"""

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class AlgoTradeError(Exception):
    """Base exception for all AlgoTrade system errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self) -> str:
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


class ConfigurationError(AlgoTradeError):
    """
    Raised when there is a configuration error.
    
    Examples:
    - Missing required configuration
    - Invalid configuration values
    - No LLM providers configured
    """
    pass


class LLMProviderError(AlgoTradeError):
    """
    Raised when there is an error with LLM providers.
    
    Examples:
    - All providers failed
    - Invalid API key
    - Rate limit exceeded
    - Provider service unavailable
    """
    pass


class ValidationError(AlgoTradeError):
    """
    Raised when validation fails.
    
    Examples:
    - Invalid tool code
    - Invalid strategy parameters
    - Invalid data format
    """
    pass


class MemoryError(AlgoTradeError):
    """
    Raised when there is an error with the memory system.
    
    Examples:
    - Database connection failed
    - Collection not found
    - Query failed
    """
    pass


class AgentError(AlgoTradeError):
    """
    Raised when there is an error with agent execution.
    
    Examples:
    - Agent failed to complete task
    - Agent timeout
    - Agent returned invalid output
    """
    pass


class ToolError(AlgoTradeError):
    """
    Raised when there is an error with tool execution.
    
    Examples:
    - Tool not found
    - Tool execution failed
    - Tool validation failed
    """
    pass


def handle_error(
    error: Exception,
    context: str,
    raise_exception: bool = True,
    log_level: str = "error"
) -> Optional[Exception]:
    """
    Handle an error with logging and optional re-raising.
    
    Args:
        error: The exception that occurred
        context: Context description (e.g., "LLM client initialization")
        raise_exception: Whether to re-raise the exception
        log_level: Logging level ("debug", "info", "warning", "error", "critical")
    
    Returns:
        The exception if not re-raised, None otherwise
    
    Example:
        >>> try:
        ...     risky_operation()
        ... except Exception as e:
        ...     handle_error(e, "Risky operation", raise_exception=False)
    """
    # Format error message
    error_msg = f"{context}: {str(error)}"
    
    # Add details if available
    if isinstance(error, AlgoTradeError) and error.details:
        error_msg += f" | Details: {error.details}"
    
    # Log at appropriate level
    log_func = getattr(logger, log_level.lower(), logger.error)
    log_func(error_msg, exc_info=True)
    
    # Re-raise if requested
    if raise_exception:
        raise error
    
    return error


def format_error_message(
    error: Exception,
    user_friendly: bool = True
) -> str:
    """
    Format an error message for display.
    
    Args:
        error: The exception to format
        user_friendly: Whether to make the message user-friendly
    
    Returns:
        Formatted error message
    
    Example:
        >>> try:
        ...     operation()
        ... except Exception as e:
        ...     print(format_error_message(e, user_friendly=True))
    """
    if isinstance(error, ConfigurationError):
        if user_friendly:
            return (
                f"Configuration Error: {error.message}\n\n"
                "Please check your .env file and ensure all required settings are present.\n"
                "See .env.example for reference."
            )
        return str(error)
    
    elif isinstance(error, LLMProviderError):
        if user_friendly:
            return (
                f"LLM Provider Error: {error.message}\n\n"
                "Suggestions:\n"
                "1. Check your API keys in .env file\n"
                "2. Verify your account has sufficient credits\n"
                "3. Check provider status pages for outages\n"
                "4. Try adding additional providers for failover"
            )
        return str(error)
    
    elif isinstance(error, ValidationError):
        if user_friendly:
            return (
                f"Validation Error: {error.message}\n\n"
                "Please review the validation errors and correct the issues."
            )
        return str(error)
    
    elif isinstance(error, MemoryError):
        if user_friendly:
            return (
                f"Memory System Error: {error.message}\n\n"
                "Suggestions:\n"
                "1. Check database connection\n"
                "2. Verify database files are not corrupted\n"
                "3. Ensure sufficient disk space"
            )
        return str(error)
    
    elif isinstance(error, AgentError):
        if user_friendly:
            return (
                f"Agent Error: {error.message}\n\n"
                "The agent encountered an error during execution.\n"
                "Please review the logs for more details."
            )
        return str(error)
    
    elif isinstance(error, ToolError):
        if user_friendly:
            return (
                f"Tool Error: {error.message}\n\n"
                "Suggestions:\n"
                "1. Verify tool is registered\n"
                "2. Check tool validation status\n"
                "3. Review tool code for errors"
            )
        return str(error)
    
    else:
        # Generic error
        if user_friendly:
            return (
                f"An unexpected error occurred: {str(error)}\n\n"
                "Please check the logs for more details."
            )
        return str(error)


def log_error_with_context(
    error: Exception,
    context: Dict[str, Any],
    log_level: str = "error"
) -> None:
    """
    Log an error with additional context information.
    
    Args:
        error: The exception that occurred
        context: Dictionary of context information
        log_level: Logging level
    
    Example:
        >>> try:
        ...     process_data(data)
        ... except Exception as e:
        ...     log_error_with_context(e, {"data_size": len(data), "step": "processing"})
    """
    log_func = getattr(logger, log_level.lower(), logger.error)
    
    context_str = ", ".join(f"{k}={v}" for k, v in context.items())
    log_func(f"Error: {str(error)} | Context: {context_str}", exc_info=True)


class ErrorContext:
    """
    Context manager for error handling with automatic logging.
    
    Example:
        >>> with ErrorContext("Database operation", raise_on_error=False):
        ...     perform_database_operation()
    """
    
    def __init__(
        self,
        context: str,
        raise_on_error: bool = True,
        log_level: str = "error"
    ):
        self.context = context
        self.raise_on_error = raise_on_error
        self.log_level = log_level
    
    def __enter__(self):
        logger.debug(f"Entering context: {self.context}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            handle_error(
                exc_val,
                self.context,
                raise_exception=self.raise_on_error,
                log_level=self.log_level
            )
            # Return True to suppress exception if not re-raising
            return not self.raise_on_error
        
        logger.debug(f"Exiting context: {self.context}")
        return False
