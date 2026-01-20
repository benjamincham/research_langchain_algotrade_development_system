import pytest
from src.tools.core_tools.financial_data_tool import FinancialDataTool
from src.tools.core_tools.news_scraper_tool import NewsScraperTool
from datetime import datetime
from typing import Type

@pytest.mark.asyncio
async def test_financial_data_tool_price_history():
    tool = FinancialDataTool()
    result = await tool.run(ticker="AAPL", data_type="price_history", timeframe="1d")
    
    assert result["ticker"] == "AAPL"
    assert result["data_type"] == "price_history"
    assert result["timeframe"] == "1d"
    assert isinstance(result["data"], list)
    assert len(result["data"]) > 0
    assert "open" in result["data"][0]
    assert "close" in result["data"][0]

@pytest.mark.asyncio
async def test_financial_data_tool_fundamentals():
    tool = FinancialDataTool()
    result = await tool.run(ticker="MSFT", data_type="fundamentals")
    
    assert result["ticker"] == "MSFT"
    assert result["data_type"] == "fundamentals"
    assert "pe_ratio" in result
    assert isinstance(result["market_cap"], int)

@pytest.mark.asyncio
async def test_financial_data_tool_key_metrics():
    tool = FinancialDataTool()
    result = await tool.run(ticker="GOOGL", data_type="key_metrics")
    
    assert result["ticker"] == "GOOGL"
    assert result["data_type"] == "key_metrics"
    assert "roe" in result
    assert isinstance(result["debt_to_equity"], float)

@pytest.mark.asyncio
async def test_financial_data_tool_unsupported_type():
    tool = FinancialDataTool()
    result = await tool.run(ticker="TSLA", data_type="unsupported")
    
    assert "error" in result
    assert "Unsupported data_type" in result["error"]

@pytest.mark.asyncio
async def test_news_scraper_tool_basic():
    tool = NewsScraperTool()
    query = "Tesla stock"
    result = await tool.run(query=query, limit=3, time_range="24h")
    
    assert result["query"] == query
    assert result["time_range"] == "24h"
    assert isinstance(result["articles"], list)
    assert len(result["articles"]) == 3
    assert "title" in result["articles"][0]
    assert "summary" in result["articles"][0]
    assert "published_at" in result["articles"][0]
