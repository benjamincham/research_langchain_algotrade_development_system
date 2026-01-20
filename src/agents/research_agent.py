from typing import Dict, Any, List, Optional
from src.core.base_agent import BaseAgent
from src.core.logging import logger
from src.memory.memory_manager import MemoryManager
import json

class ResearchAgent(BaseAgent):
    """
    Leader agent for the Research Swarm.
    
    Coordinates specialized subagents to perform comprehensive market research.
    """
    
    def __init__(
        self,
        name: str = "ResearchLeader",
        llm: Optional[Any] = None,
        memory_manager: Optional[MemoryManager] = None
    ):
        system_prompt = """
        You are the Research Leader Agent for an algorithmic trading research system.
        
        Your responsibilities:
        1. Analyze the research objective provided by the user
        2. Develop a comprehensive research strategy
        3. Spawn specialized subagents with clear, non-overlapping tasks
        4. Synthesize results from subagents into coherent findings
        5. Resolve conflicts between contradictory findings
        6. Decide if additional research is needed
        """
        super().__init__(name=name, role="Research Leader", llm=llm, system_prompt=system_prompt)
        self.memory_manager = memory_manager
        self.subagents = {}

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the research workflow.
        
        Args:
            input_data: Dictionary containing 'objective' and optional 'config'
            
        Returns:
            Synthesized research findings
        """
        objective = input_data.get("objective")
        if not objective:
            raise ValueError("Research objective is required")
            
        logger.info(f"Starting research for objective: {objective}")
        
        # Step 1: Develop strategy
        strategy_prompt = self._generate_strategy_prompt(objective)
        strategy_response = await self._call_llm(strategy_prompt)
        strategy = self._parse_strategy(strategy_response, objective)
        logger.info(f"Research strategy developed: {strategy}")
        
        # Step 2: Spawn subagents and execute (to be implemented)
        
        # Step 3: Synthesize results (to be implemented)
        
        # For now, return the developed strategy
        return {
            "objective": objective,
            "status": "strategy_developed",
            "strategy": strategy,
            "message": "Research strategy developed and ready for subagent execution"
        }

    def _generate_strategy_prompt(self, objective: str) -> str:
        """Generates the prompt for the LLM to develop a research strategy."""
        return f"""
        You are the Research Leader Agent. Your task is to develop a comprehensive research strategy
        to address the following objective: "{objective}".
        
        The strategy must be a JSON object with the following structure:
        {{
            "ticker": "The primary stock ticker (e.g., AAPL)",
            "timeframe": "The primary timeframe for analysis (e.g., 1d, 1h)",
            "subtasks": [
                {{
                    "agent_type": "The type of subagent to use (e.g., technical_analysis, fundamental_analysis, sentiment_analysis, pattern_mining)",
                    "task_description": "A clear, specific, and non-overlapping task for the subagent.",
                    "focus_area": "The specific focus of the task (e.g., 'MACD crossover', 'Q3 earnings report', 'social media sentiment')"
                }},
                ...
            ]
        }}
        
        Ensure the subtasks cover technical, fundamental, and sentiment aspects where relevant.
        The final output MUST be only the JSON object.
        """

    def _parse_strategy(self, strategy_response: str, objective: str) -> Dict[str, Any]:
        """Parses the LLM's strategy response (JSON string) into a dictionary."""
        try:
            # The LLM is instructed to return only the JSON object
            return json.loads(strategy_response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse strategy JSON: {e}. Response: {strategy_response}")
            # Fallback to a simple, safe strategy if parsing fails
            return {
                "ticker": "UNKNOWN",
                "timeframe": "1d",
                "subtasks": [
                    {
                        "agent_type": "technical_analysis",
                        "task_description": f"Perform basic technical analysis for the objective: {objective}",
                        "focus_area": "basic_indicators"
                    }
                ]
            }
