"""
MarketRegimesCollection - Collection wrapper for market regimes.

This module provides a specialized collection for storing and retrieving
market regime information with volatility and indicator data.
"""

from pydantic import BaseModel, Field
from typing import Literal, Dict, Optional
from loguru import logger

from .base_collection import BaseCollection


class RegimeMetadata(BaseModel):
    """Metadata schema for market regimes."""
    
    regime_type: Literal[
        "bull_high_vol", "bull_low_vol",
        "bear_high_vol", "bear_low_vol",
        "sideways_high_vol", "sideways_low_vol",
        "crisis"
    ] = Field(..., description="Market regime type")
    start_date: str = Field(..., description="Regime start date")
    end_date: str = Field(..., description="Regime end date (empty if current)")
    volatility: float = Field(..., ge=0.0, description="Average volatility")
    indicators: Dict[str, float] = Field(..., description="Technical indicators")


class MarketRegimesCollection(BaseCollection):
    """
    Collection for market regimes.
    
    Stores market regime information with volatility metrics and technical
    indicators for regime detection and analysis.
    
    Example:
        >>> collection = MarketRegimesCollection(client, "market_regimes")
        >>> metadata = RegimeMetadata(
        ...     regime_type="bull_low_vol",
        ...     start_date="2024-01-01",
        ...     end_date="",
        ...     volatility=0.15,
        ...     indicators={"vix": 12.5, "spy_return": 0.08}
        ... )
        >>> collection.add(
        ...     id="regime_001",
        ...     document="Bull market with low volatility, VIX below 15",
        ...     metadata=metadata.model_dump()
        ... )
    """
    
    def get_schema(self) -> type[BaseModel]:
        """Get the Pydantic schema for this collection."""
        return RegimeMetadata
    
    def get_current_regime(self) -> Optional[dict]:
        """
        Get the current market regime (end_date is empty).
        
        Returns:
            Current regime or None if not found
            
        Example:
            >>> current = collection.get_current_regime()
            >>> if current:
            ...     print(current['metadata']['regime_type'])
        """
        logger.debug("Getting current market regime")
        regimes = self.get_by_metadata({"end_date": ""}, limit=1)
        return regimes[0] if regimes else None
    
    def get_by_regime_type(self, regime_type: str, limit: int = 10) -> list[dict]:
        """
        Get regimes by type.
        
        Args:
            regime_type: Type of regime (e.g., 'bull_low_vol', 'bear_high_vol')
            limit: Maximum number of results to return
            
        Returns:
            List of regimes of the specified type
            
        Example:
            >>> bull_regimes = collection.get_by_regime_type("bull_low_vol")
        """
        logger.debug(f"Getting regimes of type: {regime_type}")
        return self.get_by_metadata({"regime_type": regime_type}, limit=limit)
    
    def get_historical_regimes(
        self, 
        start_date: str, 
        end_date: str
    ) -> list[dict]:
        """
        Get regimes within a date range.
        
        Args:
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            
        Returns:
            List of regimes within the date range
            
        Example:
            >>> regimes_2023 = collection.get_historical_regimes(
            ...     "2023-01-01", "2023-12-31"
            ... )
        """
        logger.debug(f"Getting historical regimes from {start_date} to {end_date}")
        
        # ChromaDB metadata filtering is limited for date ranges
        # So we get all and filter in Python
        all_regimes = self.get_all()
        
        historical_regimes = [
            r for r in all_regimes
            if (r['metadata'].get('start_date', '') >= start_date and 
                r['metadata'].get('start_date', '') <= end_date)
        ]
        
        # Sort by start_date
        historical_regimes.sort(key=lambda x: x['metadata'].get('start_date', ''))
        
        return historical_regimes
    
    def get_high_volatility_regimes(
        self, 
        min_volatility: float = 0.25, 
        limit: int = 10
    ) -> list[dict]:
        """
        Get high volatility regimes.
        
        Args:
            min_volatility: Minimum volatility threshold
            limit: Maximum number of results to return
            
        Returns:
            List of high volatility regimes
            
        Example:
            >>> high_vol = collection.get_high_volatility_regimes(min_volatility=0.30)
        """
        logger.debug(f"Getting high volatility regimes (>= {min_volatility})")
        
        # ChromaDB doesn't support >= operator in metadata filtering
        # So we get all and filter in Python
        all_regimes = self.get_all()
        
        high_vol_regimes = []
        for r in all_regimes:
            vol = r['metadata'].get('volatility', 0)
            if isinstance(vol, str):
                try:
                    vol = float(vol)
                except ValueError:
                    vol = 0
            if vol >= min_volatility:
                high_vol_regimes.append(r)
        
        # Sort by volatility descending
        high_vol_regimes.sort(
            key=lambda x: x['metadata'].get('volatility', 0),
            reverse=True
        )
        
        return high_vol_regimes[:limit]
    
    def get_crisis_regimes(self, limit: int = 10) -> list[dict]:
        """
        Get crisis regimes.
        
        Args:
            limit: Maximum number of results to return
            
        Returns:
            List of crisis regimes
            
        Example:
            >>> crises = collection.get_crisis_regimes()
        """
        logger.debug("Getting crisis regimes")
        return self.get_by_metadata({"regime_type": "crisis"}, limit=limit)
    
    def close_regime(self, regime_id: str, end_date: str) -> dict:
        """
        Close a regime by setting its end date.
        
        Args:
            regime_id: ID of the regime to close
            end_date: End date (ISO format)
            
        Returns:
            Result of the update operation
            
        Example:
            >>> collection.close_regime("regime_001", "2024-06-30")
        """
        logger.info(f"Closing regime {regime_id} with end_date={end_date}")
        return self.update_metadata(regime_id, {"end_date": end_date})
