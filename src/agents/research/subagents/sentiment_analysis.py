from .base_subagent import ResearchSubAgent
from typing import Any, Optional
from src.memory.memory_manager import MemoryManager

class SentimentAnalysisSubAgent(ResearchSubAgent):
    """
    Specialized subagent for performing sentiment analysis.
    """
    
    def __init__(
        self,
        llm: Optional[Any] = None,
        memory_manager: Optional[MemoryManager] = None
    ):
        super().__init__(
            name="SentimentAnalyst",
            role="Sentiment Analysis Specialist",
            llm=llm,
            memory_manager=memory_manager
        )

    def _generate_research_prompt(self, ticker: str, timeframe: str, task_description: str, focus_area: str) -> str:
        """Generates the prompt for the LLM to perform the sentiment analysis."""
        return f"""
        You are the {self.role} agent. Your task is to execute the following sentiment analysis task:
        "{task_description}"
        
        Focus Area: {focus_area}
        Ticker: {ticker}
        Timeframe: {timeframe}
        
        You have access to social media, news headlines, and forum data. Analyze the data and provide a finding.
        
        Your output MUST be a JSON object with the following structure:
        {{
            "finding_text": "A concise summary of the market sentiment (e.g., 'Social media sentiment is overwhelmingly positive with a score of 0.85').",
            "confidence": "A float between 0.0 and 1.0 representing your confidence in the finding.",
            "type": "sentiment",
            "source": "Social Media (e.g., Twitter, Reddit, News Aggregators)."
        }}
        
        The final output MUST be only the JSON object.
        """
