from typing import Dict, Any, List, Optional
from src.core.base_agent import BaseAgent
from src.core.logging import logger
from src.memory.memory_manager import MemoryManager
import json

class DomainSynthesizer(BaseAgent):
    """
    Abstract base class for specialized domain synthesizers (Tier 2 agents).
    
    These agents aggregate and synthesize findings from multiple subagents
    within a specific domain (e.g., Technical, Fundamental, Sentiment).
    """
    
    def __init__(
        self,
        name: str,
        role: str,
        domain: str,
        llm: Optional[Any] = None,
        memory_manager: Optional[MemoryManager] = None
    ):
        super().__init__(name=name, role=role, llm=llm)
        self.domain = domain
        self.memory_manager = memory_manager
        if not self.memory_manager:
            logger.warning(f"MemoryManager not provided to {self.name}. Synthesis will not query memory.")

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the synthesis workflow.
        
        Args:
            input_data: Dictionary containing 'objective', 'ticker', 'timeframe', and 'raw_findings' (List[Dict]).
            
        Returns:
            A dictionary containing the synthesized finding and metadata.
        """
        objective = input_data.get("objective")
        ticker = input_data.get("ticker")
        timeframe = input_data.get("timeframe")
        raw_findings: List[Dict[str, Any]] = input_data.get("raw_findings", [])
        
        if not all([objective, ticker, timeframe]):
            raise ValueError("Missing required input data for DomainSynthesizer.")
            
        logger.info(f"[{self.name}] Starting synthesis for {self.domain} domain on {ticker}.")
        
        # Step 1: Generate synthesis prompt
        synthesis_prompt = self._generate_synthesis_prompt(objective, ticker, timeframe, raw_findings)
        
        # Step 2: Call LLM for synthesis
        raw_synthesis = await self._call_llm(synthesis_prompt)
        
        # Step 3: Parse and validate synthesis
        synthesized_finding = self._parse_synthesis(raw_synthesis, ticker, timeframe)
        
        # Step 4: Store finding in memory (optional, typically stored by ResearchAgent)
        # For now, just return the result
        
        return synthesized_finding

    def _generate_synthesis_prompt(self, objective: str, ticker: str, timeframe: str, raw_findings: List[Dict[str, Any]]) -> str:
        """Generates the prompt for the LLM to perform the synthesis."""
        findings_text = "\n".join([
            f"- Source: {f['metadata']['agent_id']} (Confidence: {f['metadata']['confidence']:.2f}): {f['content']}"
            for f in raw_findings
        ])
        
        return f"""
        You are the {self.role} agent. Your task is to synthesize the following raw research findings
        related to the {self.domain} domain for the objective: "{objective}".
        
        Ticker: {ticker}
        Timeframe: {timeframe}
        
        Raw Findings:
        {findings_text}
        
        Your synthesis must be a JSON object with the following structure:
        {{
            "synthesis_text": "A comprehensive, consolidated summary of all findings in the {self.domain} domain.",
            "overall_confidence": "A float between 0.0 and 1.0 representing the overall confidence in the synthesized conclusion.",
            "type": "{self.domain.lower()}",
            "key_takeaways": ["List of 3-5 most important conclusions."]
        }}
        
        The final output MUST be only the JSON object.
        """

    def _parse_synthesis(self, raw_synthesis: str, ticker: str, timeframe: str) -> Dict[str, Any]:
        """Parses the raw LLM synthesis and adds metadata."""
        try:
            data = json.loads(raw_synthesis)
            
            # Construct metadata
            metadata = {
                "ticker": ticker,
                "type": data.get("type", self.domain.lower()),
                "confidence": data.get("overall_confidence", 0.0),
                "agent_id": self.name,
                "timestamp": datetime.now().isoformat(),
                "source": "Domain Synthesis",
                "timeframe": timeframe,
                "tags": [self.domain.lower(), "synthesis"] + data.get("key_takeaways", [])
            }
            
            return {
                "id": f"{self.name}-{uuid.uuid4()}",
                "content": data.get("synthesis_text", "No synthesis text provided."),
                "metadata": metadata
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"[{self.name}] Failed to parse synthesis JSON: {e}. Raw: {raw_synthesis}")
            # Fallback to a safe, low-confidence finding
            return {
                "id": f"{self.name}-{uuid.uuid4()}",
                "content": f"Error: Failed to parse LLM output. Raw: {raw_synthesis[:100]}...",
                "metadata": {
                    "ticker": ticker,
                    "type": "error",
                    "confidence": 0.0,
                    "agent_id": self.name,
                    "timestamp": datetime.now().isoformat(),
                    "source": "Error Handler",
                    "timeframe": timeframe,
                    "tags": ["parsing_error", self.domain.lower()]
                }
            }

from datetime import datetime
import uuid
