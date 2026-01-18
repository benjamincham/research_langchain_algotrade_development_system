import asyncio
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.llm_client import get_default_llm
from src.core.base_agent import BaseAgent
from src.core.logging import logger

class TestAgent(BaseAgent):
    async def run(self, input_data: dict):
        response = await self._call_llm(input_data.get("query", "Hello"))
        return {"response": response}

async def main():
    logger.info("Starting Phase 1 verification...")
    
    try:
        # Test LLM
        llm = get_default_llm()
        logger.info("LLM client initialized successfully.")
        
        # Test Agent
        agent = TestAgent(name="Verifier", role="Testing Agent")
        result = await agent.run({"query": "Say 'Phase 1 Verified'"})
        
        logger.info(f"Agent response: {result['response']}")
        
        if "Verified" in result['response']:
            logger.info("Phase 1 Verification SUCCESSFUL!")
        else:
            logger.warning("Phase 1 Verification completed but response was unexpected.")
            
    except Exception as e:
        logger.error(f"Phase 1 Verification FAILED: {e}")

if __name__ == "__main__":
    asyncio.run(main())
