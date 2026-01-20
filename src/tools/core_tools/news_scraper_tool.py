from typing import Dict, Any, List, Type
from pydantic import Field
from src.tools.tool_registry import BaseTool, ToolInputSchema
from loguru import logger
from datetime import datetime, timedelta

# --- Tool Input Schema ---

class NewsScraperToolInput(ToolInputSchema):
    """Input schema for the NewsScraperTool."""
    query: str = Field(..., description="The search query for news articles (e.g., 'AAPL earnings', 'interest rate hike').")
    limit: int = Field(5, description="The maximum number of news articles to return.")
    time_range: str = Field("24h", description="The time range for news articles (e.g., '24h', '7d', '30d').")

# --- Tool Implementation ---

class NewsScraperTool(BaseTool):
    """
    A tool for scraping and retrieving recent news articles and headlines.
    
    In a real system, this would interface with a news API or a web scraper.
    For this development phase, it will return mocked, structured data.
    """
    
    name: str = "news_scraper_tool"
    description: str = "Scrapes and retrieves recent news articles and headlines based on a search query and time range."
    input_schema: Type[ToolInputSchema] = NewsScraperToolInput

    async def run(self, query: str, limit: int = 5, time_range: str = "24h") -> Dict[str, Any]:
        """Mocks the retrieval of news articles."""
        logger.info(f"NewsScraperTool: Scraping news for query: '{query}' (Limit: {limit}, Range: {time_range})")
        
        articles = []
        now = datetime.now()
        
        for i in range(limit):
            articles.append({
                "title": f"Mock News Headline {i+1} for '{query}'",
                "source": f"MockSource {i % 3}",
                "summary": f"This is a mock summary of a news article related to '{query}'. It discusses the market impact and future outlook.",
                "published_at": (now - timedelta(hours=i*2)).isoformat(),
                "url": f"http://mockurl.com/{query.replace(' ', '_')}/{i}"
            })
            
        return {
            "query": query,
            "time_range": time_range,
            "articles": articles
        }
