from .base_subagent import ResearchSubAgent
from typing import Any, Optional
from src.memory.memory_manager import MemoryManager

class PatternMiningSubAgent(ResearchSubAgent):
    """
    Specialized subagent for identifying market patterns and anomalies.
    """
    
    def __init__(
        self,
        llm: Optional[Any] = None,
        memory_manager: Optional[MemoryManager] = None
    ):
        super().__init__(
            name="PatternMiner",
            role="Pattern Mining Specialist",
            llm=llm,
            memory_manager=memory_manager
        )

    def _generate_research_prompt(self, ticker: str, timeframe: str, task_description: str, focus_area: str) -> str:
        """Generates the prompt for the LLM to perform the pattern mining."""
        return f"""
        You are the {self.role} agent. Your task is to execute the following pattern mining task:
        "{task_description}"
        
        Focus Area: {focus_area}
        Ticker: {ticker}
        Timeframe: {timeframe}
        
        You have access to historical data and pattern recognition algorithms. Identify any significant patterns or anomalies.
        
        Your output MUST be a JSON object with the following structure:
        {{
            "finding_text": "A concise summary of the pattern or anomaly found (e.g., 'A head and shoulders pattern has formed on the 1h chart').",
            "confidence": "A float between 0.0 and 1.0 representing your confidence in the finding.",
            "type": "pattern",
            "source": "Pattern Recognition Algorithms (e.g., Candlestick, Chart Patterns)."
        }}
        
        The final output MUST be only the JSON object.
        """
