import pytest
from unittest.mock import MagicMock, AsyncMock
from src.core.base_agent import BaseAgent
from typing import Dict, Any

class MockAgent(BaseAgent):
    """Concrete implementation of BaseAgent for testing."""
    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        response = await self._call_llm(input_data.get("query", ""))
        return {"response": response}

@pytest.mark.asyncio
async def test_base_agent_initialization():
    """Test that BaseAgent initializes correctly."""
    mock_llm = MagicMock()
    agent = MockAgent(name="TestAgent", role="Tester", llm=mock_llm)
    assert agent.name == "TestAgent"
    assert agent.role == "Tester"
    assert agent.llm == mock_llm
    assert "Tester" in agent.system_prompt

@pytest.mark.asyncio
async def test_base_agent_call_llm():
    """Test that _call_llm correctly interacts with the LLM."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Mocked LLM Response"
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)
    
    agent = MockAgent(name="TestAgent", role="Tester", llm=mock_llm)
    result = await agent.run({"query": "Hello"})
    
    assert result["response"] == "Mocked LLM Response"
    mock_llm.ainvoke.assert_called_once()

from src.agents.research_agent import ResearchAgent

@pytest.mark.asyncio
async def test_research_agent_initialization():
    """Test that ResearchAgent initializes correctly."""
    mock_llm = MagicMock()
    agent = ResearchAgent(llm=mock_llm)
    assert agent.name == "ResearchLeader"
    assert "Research Leader" in agent.role
    assert agent.llm == mock_llm
    assert "Research Leader Agent" in agent.system_prompt

@pytest.mark.asyncio
async def test_research_agent_run_basic():
    """Test basic run method of ResearchAgent."""
    mock_llm = MagicMock()
    agent = ResearchAgent(llm=mock_llm)
    result = await agent.run({"objective": "Analyze AAPL momentum"})
    assert result["objective"] == "Analyze AAPL momentum"
    assert result["status"] == "initialized"
