from .base_subagent import ResearchSubAgent
from typing import Any, Optional
from src.memory.memory_manager import MemoryManager

class MarketResearchSubAgent(ResearchSubAgent):
    """
    Specialized subagent for general market research and news aggregation.
    """
    
    def __init__(
        self,
        llm: Optional[Any] = None,
        memory_manager: Optional[MemoryManager] = None
    ):
        super().__init__(
            name="MarketResearcher",
            role="General Market Research Specialist",
            llm=llm,
            memory_manager=memory_manager
        )

    def _generate_research_prompt(self, ticker: str, timeframe: str, task_description: str, focus_area: str) -> str:
        """Generates the prompt for the LLM to perform the general market research."""
        return f"""
        You are the {self.role} agent. Your task is to execute the following general market research task:
        "{task_description}"
        
        Focus Area: {focus_area}
        Ticker: {ticker}
        Timeframe: {timeframe}
        
        You have access to general news, economic calendars, and market overviews. Provide a finding.
        
        Your output MUST be a JSON object with the following structure:
        {{
            "finding_text": "A concise summary of the general market finding (e.g., 'The Federal Reserve announced a rate hike, causing market-wide volatility').",
            "confidence": "A float between 0.0 and 1.0 representing your confidence in the finding.",
            "type": "market",
            "source": "Economic News and Market Overviews."
        }}
        
        The final output MUST be only the JSON object.
        """
