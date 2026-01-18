from typing import Optional, Any
from langchain_openai import ChatOpenAI
from src.config.settings import settings
from src.core.logging import logger

class LLMClient:
    """Wrapper for LLM interactions."""
    
    def __init__(
        self, 
        model: Optional[str] = None, 
        temperature: float = 0.0,
        **kwargs: Any
    ):
        self.model = model or settings.DEFAULT_MODEL
        self.temperature = temperature
        
        try:
            self.llm = ChatOpenAI(
                model=self.model,
                temperature=self.temperature,
                openai_api_key=settings.OPENAI_API_KEY,
                openai_api_base=settings.OPENAI_API_BASE,
                **kwargs
            )
            logger.info(f"Initialized LLMClient with model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize LLMClient: {e}")
            raise

    def get_llm(self):
        """Return the underlying LangChain LLM instance."""
        return self.llm

# Singleton instance for default use
default_llm_client = None

def get_default_llm():
    """Get or initialize the default LLM instance."""
    global default_llm_client
    if default_llm_client is None:
        default_llm_client = LLMClient()
    return default_llm_client.get_llm()
