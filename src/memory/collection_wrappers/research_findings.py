"""
ResearchFindingsCollection - Collection wrapper for research findings.

This module provides a specialized collection for storing and retrieving
research findings with domain-specific query methods.
"""

from pydantic import BaseModel, Field
from typing import Literal, Optional
from loguru import logger

from .base_collection import BaseCollection


class ResearchFindingMetadata(BaseModel):
    """Metadata schema for research findings."""
    
    ticker: str = Field(..., description="Stock ticker symbol")
    type: Literal["technical", "fundamental", "sentiment", "pattern"] = Field(
        ..., description="Type of research finding"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    agent_id: str = Field(..., description="ID of agent that generated finding")
    timestamp: str = Field(..., description="ISO timestamp")
    source: str = Field(..., description="Data source")
    timeframe: str = Field(..., description="Timeframe (e.g., '1D', '1W')")
    tags: list[str] = Field(default_factory=list, description="Tags for categorization")


class ResearchFindingsCollection(BaseCollection):
    """
    Collection for research findings.
    
    Stores research findings from various agents with metadata for filtering
    and semantic search capabilities.
    
    Example:
        >>> collection = ResearchFindingsCollection(client, "research_findings")
        >>> metadata = ResearchFindingMetadata(
        ...     ticker="AAPL",
        ...     type="technical",
        ...     confidence=0.85,
        ...     agent_id="agent_001",
        ...     timestamp="2024-01-19T10:00:00",
        ...     source="yahoo_finance",
        ...     timeframe="1D",
        ...     tags=["momentum", "bullish"]
        ... )
        >>> collection.add(
        ...     id="finding_001",
        ...     document="AAPL shows strong bullish momentum",
        ...     metadata=metadata.model_dump()
        ... )
    """
    
    def get_schema(self) -> type[BaseModel]:
        """Get the Pydantic schema for this collection."""
        return ResearchFindingMetadata
    
    def get_by_ticker(self, ticker: str, limit: int = 10) -> list[dict]:
        """
        Get findings for a specific ticker.
        
        Args:
            ticker: Stock ticker symbol
            limit: Maximum number of results to return
            
        Returns:
            List of findings for the ticker
            
        Example:
            >>> findings = collection.get_by_ticker("AAPL", limit=5)
            >>> for finding in findings:
            ...     print(finding['metadata']['ticker'])
        """
        logger.debug(f"Getting findings for ticker: {ticker}")
        return self.get_by_metadata({"ticker": ticker}, limit=limit)
    
    def get_by_type(self, finding_type: str, limit: int = 10) -> list[dict]:
        """
        Get findings by type.
        
        Args:
            finding_type: Type of finding (technical, fundamental, sentiment, pattern)
            limit: Maximum number of results to return
            
        Returns:
            List of findings of the specified type
            
        Example:
            >>> technical_findings = collection.get_by_type("technical")
        """
        logger.debug(f"Getting findings of type: {finding_type}")
        return self.get_by_metadata({"type": finding_type}, limit=limit)
    
    def get_high_confidence(
        self, 
        min_confidence: float = 0.8, 
        limit: int = 10
    ) -> list[dict]:
        """
        Get high-confidence findings.
        
        Args:
            min_confidence: Minimum confidence threshold (0.0 to 1.0)
            limit: Maximum number of results to return
            
        Returns:
            List of high-confidence findings
            
        Example:
            >>> high_conf = collection.get_high_confidence(min_confidence=0.9)
        """
        logger.debug(f"Getting high-confidence findings (>= {min_confidence})")
        
        # ChromaDB doesn't support >= operator in metadata filtering
        # So we get all and filter in Python
        all_findings = self.get_all()
        
        high_conf_findings = []
        for f in all_findings:
            confidence = f['metadata'].get('confidence', 0)
            if isinstance(confidence, str):
                try:
                    confidence = float(confidence)
                except ValueError:
                    confidence = 0
            if confidence >= min_confidence:
                high_conf_findings.append(f)
        
        return high_conf_findings[:limit]
    
    def get_by_agent(self, agent_id: str, limit: int = 10) -> list[dict]:
        """
        Get findings by agent ID.
        
        Args:
            agent_id: ID of the agent that generated the findings
            limit: Maximum number of results to return
            
        Returns:
            List of findings from the specified agent
            
        Example:
            >>> agent_findings = collection.get_by_agent("agent_001")
        """
        logger.debug(f"Getting findings from agent: {agent_id}")
        return self.get_by_metadata({"agent_id": agent_id}, limit=limit)
    
    def get_by_timeframe(self, timeframe: str, limit: int = 10) -> list[dict]:
        """
        Get findings by timeframe.
        
        Args:
            timeframe: Timeframe (e.g., '1D', '1W', '1M')
            limit: Maximum number of results to return
            
        Returns:
            List of findings for the specified timeframe
            
        Example:
            >>> daily_findings = collection.get_by_timeframe("1D")
        """
        logger.debug(f"Getting findings for timeframe: {timeframe}")
        return self.get_by_metadata({"timeframe": timeframe}, limit=limit)
