"""
StrategyLibraryCollection - Collection wrapper for trading strategies.

This module provides a specialized collection for storing and retrieving
trading strategies with performance metrics and domain-specific query methods.
"""

from pydantic import BaseModel, Field
from typing import Literal, Dict, List
from loguru import logger

from .base_collection import BaseCollection


class PerformanceMetrics(BaseModel):
    """Performance metrics sub-schema."""
    
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    max_drawdown: float = Field(..., description="Maximum drawdown")
    total_return: float = Field(..., description="Total return")
    win_rate: float = Field(..., ge=0.0, le=1.0, description="Win rate")


class StrategyMetadata(BaseModel):
    """Metadata schema for strategies."""
    
    name: str = Field(..., description="Strategy name")
    type: Literal["momentum", "mean_reversion", "breakout", "arbitrage"] = Field(
        ..., description="Strategy type"
    )
    tickers: List[str] = Field(..., description="Tickers this strategy applies to")
    timeframe: str = Field(..., description="Timeframe")
    parameters: Dict[str, float] = Field(..., description="Strategy parameters")
    performance_metrics: PerformanceMetrics = Field(..., description="Performance metrics")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")


class StrategyLibraryCollection(BaseCollection):
    """
    Collection for trading strategies.
    
    Stores trading strategies with performance metrics and provides methods
    for filtering by type, performance, and ticker.
    
    Example:
        >>> collection = StrategyLibraryCollection(client, "strategy_library")
        >>> metrics = PerformanceMetrics(
        ...     sharpe_ratio=1.8,
        ...     max_drawdown=-0.15,
        ...     total_return=0.25,
        ...     win_rate=0.65
        ... )
        >>> metadata = StrategyMetadata(
        ...     name="Momentum Strategy v1",
        ...     type="momentum",
        ...     tickers=["AAPL", "GOOGL"],
        ...     timeframe="1D",
        ...     parameters={"lookback": 20, "threshold": 0.02},
        ...     performance_metrics=metrics,
        ...     created_at="2024-01-19T10:00:00",
        ...     updated_at="2024-01-19T10:00:00"
        ... )
        >>> collection.add(
        ...     id="strategy_001",
        ...     document="Momentum strategy using RSI and MACD",
        ...     metadata=metadata.model_dump()
        ... )
    """
    
    def get_schema(self) -> type[BaseModel]:
        """Get the Pydantic schema for this collection."""
        return StrategyMetadata
    
    def get_by_type(self, strategy_type: str, limit: int = 10) -> list[dict]:
        """
        Get strategies by type.
        
        Args:
            strategy_type: Type of strategy (momentum, mean_reversion, breakout, arbitrage)
            limit: Maximum number of results to return
            
        Returns:
            List of strategies of the specified type
            
        Example:
            >>> momentum_strategies = collection.get_by_type("momentum")
        """
        logger.debug(f"Getting strategies of type: {strategy_type}")
        return self.get_by_metadata({"type": strategy_type}, limit=limit)
    
    def get_top_performers(
        self, 
        min_sharpe: float = 1.5, 
        limit: int = 10
    ) -> list[dict]:
        """
        Get top-performing strategies by Sharpe ratio.
        
        Args:
            min_sharpe: Minimum Sharpe ratio threshold
            limit: Maximum number of results to return
            
        Returns:
            List of top-performing strategies
            
        Example:
            >>> top_strategies = collection.get_top_performers(min_sharpe=2.0)
        """
        logger.debug(f"Getting top performers (Sharpe >= {min_sharpe})")
        
        # ChromaDB doesn't support nested field filtering well
        # So we get all and filter in Python
        all_strategies = self.get_all()
        
        top_performers = [
            s for s in all_strategies 
            if s['metadata'].get('performance_metrics', {}).get('sharpe_ratio', 0) >= min_sharpe
        ]
        
        # Sort by Sharpe ratio descending
        top_performers.sort(
            key=lambda x: x['metadata'].get('performance_metrics', {}).get('sharpe_ratio', 0),
            reverse=True
        )
        
        return top_performers[:limit]
    
    def get_for_ticker(self, ticker: str, limit: int = 10) -> list[dict]:
        """
        Get strategies for a specific ticker.
        
        Args:
            ticker: Stock ticker symbol
            limit: Maximum number of results to return
            
        Returns:
            List of strategies that apply to the ticker
            
        Example:
            >>> aapl_strategies = collection.get_for_ticker("AAPL")
        """
        logger.debug(f"Getting strategies for ticker: {ticker}")
        
        # ChromaDB doesn't support array contains, so we get all and filter
        all_strategies = self.get_all()
        
        ticker_strategies = [
            s for s in all_strategies 
            if ticker in s['metadata'].get('tickers', [])
        ]
        
        return ticker_strategies[:limit]
    
    def update_performance(
        self, 
        strategy_id: str, 
        metrics: PerformanceMetrics
    ) -> dict:
        """
        Update performance metrics for a strategy.
        
        Args:
            strategy_id: ID of the strategy to update
            metrics: New performance metrics
            
        Returns:
            Result of the update operation
            
        Example:
            >>> new_metrics = PerformanceMetrics(
            ...     sharpe_ratio=2.0,
            ...     max_drawdown=-0.10,
            ...     total_return=0.30,
            ...     win_rate=0.70
            ... )
            >>> collection.update_performance("strategy_001", new_metrics)
        """
        logger.info(f"Updating performance metrics for strategy: {strategy_id}")
        
        # Get current metadata
        strategy = self.get(strategy_id)
        if not strategy:
            raise ValueError(f"Strategy not found: {strategy_id}")
        
        # Update performance_metrics and updated_at
        from datetime import datetime
        updated_metadata = {
            "performance_metrics": metrics.model_dump(),
            "updated_at": datetime.now().isoformat()
        }
        
        return self.update_metadata(strategy_id, updated_metadata)
    
    def get_by_timeframe(self, timeframe: str, limit: int = 10) -> list[dict]:
        """
        Get strategies by timeframe.
        
        Args:
            timeframe: Timeframe (e.g., '1D', '1W', '1M')
            limit: Maximum number of results to return
            
        Returns:
            List of strategies for the specified timeframe
            
        Example:
            >>> daily_strategies = collection.get_by_timeframe("1D")
        """
        logger.debug(f"Getting strategies for timeframe: {timeframe}")
        return self.get_by_metadata({"timeframe": timeframe}, limit=limit)
