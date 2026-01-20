from typing import Dict, Any, List, Type
from pydantic import Field
from src.tools.tool_registry import BaseTool, ToolInputSchema
from loguru import logger
from datetime import datetime, timedelta

# --- Tool Input Schema ---

class FinancialDataToolInput(ToolInputSchema):
    """Input schema for the FinancialDataTool."""
    ticker: str = Field(..., description="The stock ticker symbol (e.g., AAPL, MSFT).")
    data_type: str = Field(..., description="The type of financial data to retrieve (e.g., 'price_history', 'fundamentals', 'key_metrics').")
    timeframe: str = Field("1d", description="The time interval for price history (e.g., '1d', '1h', '1wk'). Required for 'price_history'.")
    start_date: str = Field(None, description="Start date for data retrieval (YYYY-MM-DD).")
    end_date: str = Field(None, description="End date for data retrieval (YYYY-MM-DD).")

# --- Tool Implementation ---

class FinancialDataTool(BaseTool):
    """
    A tool for retrieving various types of financial and market data.
    
    In a real system, this would interface with a financial data API (e.g., Alpha Vantage, Polygon.io).
    For this development phase, it will return mocked, structured data.
    """
    
    name: str = "financial_data_tool"
    description: str = "Retrieves historical price data, fundamental metrics, and key financial ratios for a given stock ticker."
    input_schema: Type[ToolInputSchema] = FinancialDataToolInput

    async def run(self, ticker: str, data_type: str, timeframe: str = "1d", start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """Mocks the retrieval of financial data."""
        logger.info(f"FinancialDataTool: Retrieving {data_type} for {ticker} (Timeframe: {timeframe})")
        
        if data_type == "price_history":
            return self._mock_price_history(ticker, timeframe, start_date, end_date)
        elif data_type == "fundamentals":
            return self._mock_fundamentals(ticker)
        elif data_type == "key_metrics":
            return self._mock_key_metrics(ticker)
        else:
            return {"error": f"Unsupported data_type: {data_type}"}

    def _mock_price_history(self, ticker: str, timeframe: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """Generates mock historical price data."""
        
        # Simple mock data generation
        data = []
        end = datetime.now()
        start = end - timedelta(days=30)
        
        current_date = start
        while current_date <= end:
            # Only generate data for weekdays
            if current_date.weekday() < 5:
                data.append({
                    "date": current_date.strftime("%Y-%m-%d"),
                    "open": round(150 + (current_date.day * 0.5) + (current_date.day % 3) * 2, 2),
                    "high": round(155 + (current_date.day * 0.5), 2),
                    "low": round(148 + (current_date.day * 0.5), 2),
                    "close": round(150 + (current_date.day * 0.5) + (current_date.day % 3) * 1.5, 2),
                    "volume": 1000000 + current_date.day * 50000
                })
            current_date += timedelta(days=1)
            
        return {
            "ticker": ticker,
            "data_type": "price_history",
            "timeframe": timeframe,
            "data": data
        }

    def _mock_fundamentals(self, ticker: str) -> Dict[str, Any]:
        """Generates mock fundamental data."""
        return {
            "ticker": ticker,
            "data_type": "fundamentals",
            "market_cap": 2500000000000,
            "pe_ratio": 25.5,
            "eps": 6.12,
            "dividend_yield": 0.005,
            "last_updated": datetime.now().strftime("%Y-%m-%d")
        }

    def _mock_key_metrics(self, ticker: str) -> Dict[str, Any]:
        """Generates mock key metrics data."""
        return {
            "ticker": ticker,
            "data_type": "key_metrics",
            "roe": 0.35,
            "roa": 0.15,
            "debt_to_equity": 1.2,
            "current_ratio": 1.5,
            "last_updated": datetime.now().strftime("%Y-%m-%d")
        }
