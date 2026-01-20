from typing import Dict, Any, List, Optional
from src.core.base_agent import BaseAgent
from src.core.logging import logger
from src.memory.memory_manager import MemoryManager
from src.agents.research.subagents import TechnicalAnalysisSubAgent, FundamentalAnalysisSubAgent, SentimentAnalysisSubAgent, PatternMiningSubAgent, MarketResearchSubAgent
from src.agents.research.domain_synthesizers import TechnicalSynthesizer, FundamentalSynthesizer, SentimentSynthesizer
import asyncio
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
        self.subagents = self._initialize_subagents()
        self.synthesizers = self._initialize_synthesizers()

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
        
        # Step 2: Spawn subagents and execute
        raw_findings = await self._execute_subtasks(strategy)
        
        # Step 3: Synthesize results
        final_synthesis = await self._synthesize_results(strategy, raw_findings)
        
        # Step 4: Return final synthesis
        return {
            "objective": objective,
            "status": "research_complete",
            "strategy": strategy,
            "final_synthesis": final_synthesis,
            "message": "Research completed and final synthesis generated"
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

    def _initialize_subagents(self) -> Dict[str, BaseAgent]:
        """Initializes all specialized subagents."""
        return {
            "technical_analysis": TechnicalAnalysisSubAgent(llm=self.llm, memory_manager=self.memory_manager),
            "fundamental_analysis": FundamentalAnalysisSubAgent(llm=self.llm, memory_manager=self.memory_manager),
            "sentiment_analysis": SentimentAnalysisSubAgent(llm=self.llm, memory_manager=self.memory_manager),
            "pattern_mining": PatternMiningSubAgent(llm=self.llm, memory_manager=self.memory_manager),
            "market_research": MarketResearchSubAgent(llm=self.llm, memory_manager=self.memory_manager),
        }

    def _initialize_synthesizers(self) -> Dict[str, BaseAgent]:
        """Initializes all domain synthesizers."""
        return {
            "technical": TechnicalSynthesizer(llm=self.llm, memory_manager=self.memory_manager),
            "fundamental": FundamentalSynthesizer(llm=self.llm, memory_manager=self.memory_manager),
            "sentiment": SentimentSynthesizer(llm=self.llm, memory_manager=self.memory_manager),
        }

    async def _execute_subtasks(self, strategy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Executes subtasks in parallel using the appropriate subagents."""
        tasks = []
        raw_findings = []
        
        for subtask in strategy.get("subtasks", []):
            agent_type = subtask.get("agent_type")
            if agent_type in self.subagents:
                agent = self.subagents[agent_type]
                task_input = {
                    "ticker": strategy["ticker"],
                    "timeframe": strategy["timeframe"],
                    "task_description": subtask["task_description"],
                    "focus_area": subtask["focus_area"]
                }
                tasks.append(agent.run(task_input))
            else:
                logger.warning(f"Unknown agent type in strategy: {agent_type}")

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Subagent task failed: {result}")
                # Handle error finding (e.g., create a low-confidence error finding)
            else:
                raw_findings.append(result)
                
        return raw_findings

    async def _synthesize_results(self, strategy: Dict[str, Any], raw_findings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Synthesizes raw findings using domain synthesizers and the leader agent."""
        
        # Group findings by domain
        grouped_findings = {
            "technical": [],
            "fundamental": [],
            "sentiment": [],
            "market": [],
            "pattern": [],
        }
        for finding in raw_findings:
            finding_type = finding["metadata"]["type"]
            if finding_type in grouped_findings:
                grouped_findings[finding_type].append(finding)
            else:
                logger.warning(f"Finding with unknown type: {finding_type}")

        # Execute domain synthesizers concurrently
        synthesis_tasks = []
        domain_syntheses = []
        
        for domain, synthesizer in self.synthesizers.items():
            if grouped_findings.get(domain):
                synthesis_input = {
                    "objective": strategy["objective"],
                    "ticker": strategy["ticker"],
                    "timeframe": strategy["timeframe"],
                    "raw_findings": grouped_findings[domain]
                }
                synthesis_tasks.append(synthesizer.run(synthesis_input))

        domain_syntheses_results = await asyncio.gather(*synthesis_tasks, return_exceptions=True)
        
        for result in domain_syntheses_results:
            if isinstance(result, Exception):
                logger.error(f"Domain synthesis failed: {result}")
            else:
                domain_syntheses.append(result)

        # Final synthesis by the Research Leader (to be implemented)
        # For now, return a placeholder combining all results
        
        final_synthesis = {
            "leader_synthesis": "Final synthesis by Research Leader (Not yet implemented)",
            "domain_syntheses": domain_syntheses,
            "raw_findings": raw_findings,
            "status": "partial_synthesis"
        }
        
        return final_synthesis

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
