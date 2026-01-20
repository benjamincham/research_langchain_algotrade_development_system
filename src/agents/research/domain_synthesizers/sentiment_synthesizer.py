from .base_synthesizer import DomainSynthesizer
from typing import Any, Optional
from src.memory.memory_manager import MemoryManager

class SentimentSynthesizer(DomainSynthesizer):
    """
    Tier 2 agent for synthesizing sentiment analysis findings.
    """
    
    def __init__(
        self,
        llm: Optional[Any] = None,
        memory_manager: Optional[MemoryManager] = None
    ):
        super().__init__(
            name="SentimentSynthesizer",
            role="Sentiment Domain Synthesizer",
            domain="Sentiment",
            llm=llm,
            memory_manager=memory_manager
        )
