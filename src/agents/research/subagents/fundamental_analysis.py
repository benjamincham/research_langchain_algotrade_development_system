from .base_subagent import ResearchSubAgent
from typing import Any, Optional
from src.memory.memory_manager import MemoryManager

class FundamentalAnalysisSubAgent(ResearchSubAgent):
    """
    Specialized subagent for performing fundamental analysis.
    """
    
    def __init__(
        self,
        llm: Optional[Any] = None,
        memory_manager: Optional[MemoryManager] = None
    ):
        super().__init__(
            name="FundamentalAnalyst",
            role="Fundamental Analysis Specialist",
            llm=llm,
            memory_manager=memory_manager
        )

    def _generate_research_prompt(self, ticker: str, timeframe: str, task_description: str, focus_area: str) -> str:
        """Generates the prompt for the LLM to perform the fundamental analysis."""
        return f"""
        You are the {self.role} agent. Your task is to execute the following fundamental analysis task:
        "{task_description}"
        
        Focus Area: {focus_area}
        Ticker: {ticker}
        Timeframe: {timeframe}
        
        You have access to financial statements, news, and company reports. Analyze the data and provide a finding.
        
        Your output MUST be a JSON object with the following structure:
        {{
            "finding_text": "A concise summary of the fundamental finding (e.g., 'P/E ratio of 15.0 suggests the stock is undervalued').",
            "confidence": "A float between 0.0 and 1.0 representing your confidence in the finding.",
            "type": "fundamental",
            "source": "Financial Statements (e.g., Q3 Earnings Report, SEC Filings)."
        }}
        
        The final output MUST be only the JSON object.
        """
