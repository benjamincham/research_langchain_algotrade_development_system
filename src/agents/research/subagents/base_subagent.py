from typing import Dict, Any, Optional
from src.core.base_agent import BaseAgent
from src.core.logging import logger
from src.memory.memory_manager import MemoryManager
from src.memory.collection_wrappers.research_findings import ResearchFindingMetadata
import json
from datetime import datetime
import uuid

class ResearchSubAgent(BaseAgent):
    """
    Abstract base class for all specialized research subagents.
    
    Handles the execution of a specific research task and stores the finding
    in the ResearchFindingsCollection.
    """
    
    def __init__(
        self,
        name: str,
        role: str,
        llm: Optional[Any] = None,
        memory_manager: Optional[MemoryManager] = None
    ):
        super().__init__(name=name, role=role, llm=llm)
        self.memory_manager = memory_manager
        if not self.memory_manager:
            logger.warning(f"MemoryManager not provided to {self.name}. Findings will not be stored.")

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the specific research task.
        
        Args:
            input_data: Dictionary containing 'ticker', 'timeframe', 'task_description', 'focus_area'.
            
        Returns:
            A dictionary containing the research finding and metadata.
        """
        ticker = input_data.get("ticker")
        timeframe = input_data.get("timeframe")
        task_description = input_data.get("task_description")
        focus_area = input_data.get("focus_area")
        
        if not all([ticker, timeframe, task_description, focus_area]):
            raise ValueError("Missing required input data for ResearchSubAgent.")
            
        logger.info(f"[{self.name}] Starting task: {task_description} for {ticker} ({timeframe})")
        
        # Step 1: Perform research (LLM call with tool-use capability)
        research_prompt = self._generate_research_prompt(ticker, timeframe, task_description, focus_area)
        
        # Get all registered tools from the registry
        tools = self.tool_registry.get_all_tools() if self.tool_registry else []
        
        # First LLM call to decide on tool use
        raw_finding = await self._call_llm(research_prompt, tools=tools)
        
        # Check for tool calls and execute them
        # NOTE: This is a simplified tool-use loop. A full implementation would require
        # parsing the LLM response for tool calls, executing them, and feeding the
        # results back to the LLM in a loop. For this phase, we'll assume the LLM
        # returns the final finding after considering the available tools.
        # The actual tool-use loop will be implemented in a later phase or a more
        # specialized agent. For now, we'll focus on making the tools available.
        
        # Step 2: Parse and validate finding
        finding_data = self._parse_finding(raw_finding, ticker, timeframe, focus_area)
        
        # Step 3: Store finding in memory
        if self.memory_manager:
            self._store_finding(finding_data)
            
        return finding_data

    def _generate_research_prompt(self, ticker: str, timeframe: str, task_description: str, focus_area: str) -> str:
        """Generates the prompt for the LLM to perform the research."""
        # This will be overridden by specialized agents, but a base prompt is useful
        return f"""
        You are the {self.role} agent. Your task is to execute the following research task:
        "{task_description}"
        
        Focus Area: {focus_area}
        Ticker: {ticker}
        Timeframe: {timeframe}
        
        Your output MUST be a JSON object with the following structure:
        {{
            "finding_text": "A concise summary of the research finding (max 500 characters).",
            "confidence": "A float between 0.0 and 1.0 representing your confidence in the finding.",
            "type": "The type of finding (e.g., technical, fundamental, sentiment, pattern).",
            "source": "The primary data source used (e.g., Yahoo Finance, SEC Filings, Twitter)."
        }}
        
        The final output MUST be only the JSON object.
        """

    def _parse_finding(self, raw_finding: str, ticker: str, timeframe: str, focus_area: str) -> Dict[str, Any]:
        """Parses the raw LLM finding and validates it against the schema."""
        try:
            # Attempt to parse the JSON
            data = json.loads(raw_finding)
            
            # Construct full metadata for validation
            metadata = {
                "ticker": ticker,
                "type": data.get("type", "unknown"),
                "confidence": data.get("confidence", 0.0),
                "agent_id": self.name,
                "timestamp": datetime.now().isoformat(),
                "source": data.get("source", "LLM Synthesis"),
                "timeframe": timeframe,
                "tags": [focus_area, self.role.lower().replace(" ", "_")]
            }
            
            # Validate metadata using Pydantic schema (assuming it's available)
            # Note: This requires the Pydantic schema to be imported and used, 
            # which is not fully implemented yet, so we'll use a placeholder for now.
            # validated_metadata = ResearchFindingMetadata(**metadata)
            
            return {
                "id": f"{self.name}-{uuid.uuid4()}",
                "content": data.get("finding_text", "No finding text provided."),
                "metadata": metadata
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"[{self.name}] Failed to parse finding JSON: {e}. Raw: {raw_finding}")
            # Fallback to a safe, low-confidence finding
            return {
                "id": f"{self.name}-{uuid.uuid4()}",
                "content": f"Error: Failed to parse LLM output. Raw: {raw_finding[:100]}...",
                "metadata": {
                    "ticker": ticker,
                    "type": "error",
                    "confidence": 0.0,
                    "agent_id": self.name,
                    "timestamp": datetime.now().isoformat(),
                    "source": "Error Handler",
                    "timeframe": timeframe,
                    "tags": ["parsing_error"]
                }
            }

    def _store_finding(self, finding_data: Dict[str, Any]) -> None:
        """Stores the validated finding in the ResearchFindingsCollection."""
        try:
            # Assuming memory_manager has a method to get the correct collection
            # and the collection has an 'add' method that takes id, content, metadata
            research_collection = self.memory_manager.get_collection("research_findings")
            research_collection.add(
                doc_id=finding_data["id"],
                content=finding_data["content"],
                metadata=finding_data["metadata"]
            )
            logger.info(f"[{self.name}] Stored finding {finding_data['id']} in memory.")
        except Exception as e:
            logger.error(f"[{self.name}] Failed to store finding in memory: {e}")

# Need to import datetime and uuid for the base class
from datetime import datetime
import uuid
