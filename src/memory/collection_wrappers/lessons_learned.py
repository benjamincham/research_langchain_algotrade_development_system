"""
LessonsLearnedCollection - Collection wrapper for lessons learned.

This module provides a specialized collection for storing and retrieving
lessons learned from failures, successes, and optimizations.
"""

from pydantic import BaseModel, Field
from typing import Literal
from loguru import logger

from .base_collection import BaseCollection


class LessonMetadata(BaseModel):
    """Metadata schema for lessons learned."""
    
    type: Literal["failure", "success", "optimization", "insight"] = Field(
        ..., description="Type of lesson"
    )
    severity: Literal["critical", "high", "medium", "low"] = Field(
        ..., description="Severity/importance level"
    )
    context: str = Field(..., description="Context where lesson was learned")
    timestamp: str = Field(..., description="ISO timestamp")
    tags: list[str] = Field(default_factory=list, description="Tags for categorization")


class LessonsLearnedCollection(BaseCollection):
    """
    Collection for lessons learned.
    
    Stores lessons learned from various contexts (backtesting, live trading, etc.)
    with severity levels and types for filtering.
    
    Example:
        >>> collection = LessonsLearnedCollection(client, "lessons_learned")
        >>> metadata = LessonMetadata(
        ...     type="failure",
        ...     severity="critical",
        ...     context="backtesting",
        ...     timestamp="2024-01-19T10:00:00",
        ...     tags=["overfitting", "data_leakage"]
        ... )
        >>> collection.add(
        ...     id="lesson_001",
        ...     document="Strategy overfit to training data due to look-ahead bias",
        ...     metadata=metadata.model_dump()
        ... )
    """
    
    def get_schema(self) -> type[BaseModel]:
        """Get the Pydantic schema for this collection."""
        return LessonMetadata
    
    def get_failures(self, limit: int = 10) -> list[dict]:
        """
        Get failure lessons.
        
        Args:
            limit: Maximum number of results to return
            
        Returns:
            List of failure lessons
            
        Example:
            >>> failures = collection.get_failures()
            >>> for failure in failures:
            ...     print(failure['document'])
        """
        logger.debug("Getting failure lessons")
        return self.get_by_metadata({"type": "failure"}, limit=limit)
    
    def get_successes(self, limit: int = 10) -> list[dict]:
        """
        Get success lessons.
        
        Args:
            limit: Maximum number of results to return
            
        Returns:
            List of success lessons
            
        Example:
            >>> successes = collection.get_successes()
        """
        logger.debug("Getting success lessons")
        return self.get_by_metadata({"type": "success"}, limit=limit)
    
    def get_critical_lessons(self, limit: int = 10) -> list[dict]:
        """
        Get critical severity lessons.
        
        Args:
            limit: Maximum number of results to return
            
        Returns:
            List of critical lessons
            
        Example:
            >>> critical = collection.get_critical_lessons()
        """
        logger.debug("Getting critical severity lessons")
        return self.get_by_metadata({"severity": "critical"}, limit=limit)
    
    def get_by_context(self, context: str, limit: int = 10) -> list[dict]:
        """
        Get lessons by context.
        
        Args:
            context: Context where lesson was learned (e.g., 'backtesting', 'live_trading')
            limit: Maximum number of results to return
            
        Returns:
            List of lessons from the specified context
            
        Example:
            >>> backtest_lessons = collection.get_by_context("backtesting")
        """
        logger.debug(f"Getting lessons from context: {context}")
        return self.get_by_metadata({"context": context}, limit=limit)
    
    def get_by_severity(self, severity: str, limit: int = 10) -> list[dict]:
        """
        Get lessons by severity level.
        
        Args:
            severity: Severity level (critical, high, medium, low)
            limit: Maximum number of results to return
            
        Returns:
            List of lessons with the specified severity
            
        Example:
            >>> high_severity = collection.get_by_severity("high")
        """
        logger.debug(f"Getting lessons with severity: {severity}")
        return self.get_by_metadata({"severity": severity}, limit=limit)
    
    def get_by_type(self, lesson_type: str, limit: int = 10) -> list[dict]:
        """
        Get lessons by type.
        
        Args:
            lesson_type: Type of lesson (failure, success, optimization, insight)
            limit: Maximum number of results to return
            
        Returns:
            List of lessons of the specified type
            
        Example:
            >>> insights = collection.get_by_type("insight")
        """
        logger.debug(f"Getting lessons of type: {lesson_type}")
        return self.get_by_metadata({"type": lesson_type}, limit=limit)
