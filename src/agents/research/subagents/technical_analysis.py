from .base_subagent import ResearchSubAgent
from typing import Any, Optional
from src.memory.memory_manager import MemoryManager

class TechnicalAnalysisSubAgent(ResearchSubAgent):
    """
    Specialized subagent for performing technical analysis.
    """
    
    def __init__(
        self,
        llm: Optional[Any] = None,
        memory_manager: Optional[MemoryManager] = None
    ):
        super().__init__(
            name="TechnicalAnalyst",
            role="Technical Analysis Specialist",
            llm=llm,
            memory_manager=memory_manager
        )

    def _generate_research_prompt(self, ticker: str, timeframe: str, task_description: str, focus_area: str) -> str:
        """Generates the prompt for the LLM to perform the technical analysis."""
        return f"""
        You are the {self.role} agent. Your task is to execute the following technical analysis task:
        "{task_description}"
        
        Focus Area: {focus_area}
        Ticker: {ticker}
        Timeframe: {timeframe}
        
        You have access to historical price and volume data. Analyze the data and provide a finding.
        
        Your output MUST be a JSON object with the following structure:
        {{
            "finding_text": "A concise summary of the technical finding (e.g., 'MACD crossover suggests bullish momentum').",
            "confidence": "A float between 0.0 and 1.0 representing your confidence in the finding.",
            "type": "technical",
            "source": "Technical Indicators (e.g., MACD, RSI, Bollinger Bands)."
        }}
        
        The final output MUST be only the JSON object.
        """
