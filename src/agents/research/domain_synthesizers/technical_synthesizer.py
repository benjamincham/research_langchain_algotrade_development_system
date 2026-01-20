from .base_synthesizer import DomainSynthesizer
from typing import Any, Optional
from src.memory.memory_manager import MemoryManager

class TechnicalSynthesizer(DomainSynthesizer):
    """
    Tier 2 agent for synthesizing technical analysis findings.
    """
    
    def __init__(
        self,
        llm: Optional[Any] = None,
        memory_manager: Optional[MemoryManager] = None
    ):
        super().__init__(
            name="TechnicalSynthesizer",
            role="Technical Domain Synthesizer",
            domain="Technical",
            llm=llm,
            memory_manager=memory_manager
        )
