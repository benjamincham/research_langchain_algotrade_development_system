from .base_synthesizer import DomainSynthesizer
from typing import Any, Optional
from src.memory.memory_manager import MemoryManager

class FundamentalSynthesizer(DomainSynthesizer):
    """
    Tier 2 agent for synthesizing fundamental analysis findings.
    """
    
    def __init__(
        self,
        llm: Optional[Any] = None,
        memory_manager: Optional[MemoryManager] = None
    ):
        super().__init__(
            name="FundamentalSynthesizer",
            role="Fundamental Domain Synthesizer",
            domain="Fundamental",
            llm=llm,
            memory_manager=memory_manager
        )
