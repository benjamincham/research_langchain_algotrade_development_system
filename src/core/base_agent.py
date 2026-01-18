from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from src.core.llm_client import get_default_llm
from src.core.logging import logger

class BaseAgent(ABC):
    """Abstract base class for all agents in the system."""
    
    def __init__(
        self, 
        name: str, 
        role: str, 
        llm: Optional[Any] = None,
        system_prompt: Optional[str] = None
    ):
        self.name = name
        self.role = role
        self.llm = llm or get_default_llm()
        self.system_prompt = system_prompt or f"You are {name}, a {role}."
        logger.info(f"Initialized agent: {self.name} ({self.role})")

    @abstractmethod
    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent's primary logic."""
        pass

    def _format_messages(self, user_input: str, history: List[BaseMessage] = None) -> List[BaseMessage]:
        """Format messages for the LLM including system prompt and history."""
        messages = [HumanMessage(content=self.system_prompt)]
        if history:
            messages.extend(history)
        messages.append(HumanMessage(content=user_input))
        return messages

    async def _call_llm(self, user_input: str, history: List[BaseMessage] = None) -> str:
        """Helper to call the LLM with formatted messages."""
        messages = self._format_messages(user_input, history)
        try:
            response = await self.llm.ainvoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Error calling LLM for agent {self.name}: {e}")
            raise
