import pytest
from unittest.mock import AsyncMock, MagicMock
from src.agents.research.subagents.base_subagent import ResearchSubAgent
from src.agents.research.subagents.technical_analysis import TechnicalAnalysisSubAgent
from src.agents.research.subagents.fundamental_analysis import FundamentalAnalysisSubAgent
from src.agents.research.subagents.sentiment_analysis import SentimentAnalysisSubAgent
from src.agents.research.subagents.pattern_mining import PatternMiningSubAgent
from src.agents.research.subagents.market_research import MarketResearchSubAgent
from src.memory.memory_manager import MemoryManager
from src.core.base_agent import BaseAgent # For mocking

# Mock the MemoryManager and its collection
@pytest.fixture
def mock_memory_manager():
    mock_collection = MagicMock()
    mock_manager = MagicMock(spec=MemoryManager)
    mock_manager.get_collection.return_value = mock_collection
    return mock_manager

# Mock the LLM response for a successful finding
@pytest.fixture
def mock_llm_response():
    return MagicMock(content='{"finding_text": "Test finding.", "confidence": 0.9, "type": "technical", "source": "Test Source"}')

# Mock the LLM for a successful run
@pytest.fixture
def mock_llm_success(mock_llm_response):
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = mock_llm_response
    return mock_llm

# Mock the LLM for a parsing failure
@pytest.fixture
def mock_llm_failure():
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = MagicMock(content='This is not JSON.') # Invalid JSON
    return mock_llm

# Input data for testing
@pytest.fixture
def input_data():
    return {
        "ticker": "AAPL",
        "timeframe": "1d",
        "task_description": "Analyze the 50-day moving average.",
        "focus_area": "MA_50"
    }

@pytest.mark.asyncio
async def test_base_subagent_initialization(mock_llm_success, mock_memory_manager):
    agent = ResearchSubAgent(name="TestAgent", role="Tester", llm=mock_llm_success, memory_manager=mock_memory_manager)
    assert agent.name == "TestAgent"
    assert agent.role == "Tester"
    assert agent.memory_manager == mock_memory_manager

@pytest.mark.asyncio
async def test_base_subagent_run_success(mock_llm_success, mock_memory_manager, input_data):
    agent = ResearchSubAgent(name="TestAgent", role="Tester", llm=mock_llm_success, memory_manager=mock_memory_manager)
    result = await agent.run(input_data)
    
    assert result["content"] == "Test finding."
    assert result["metadata"]["ticker"] == "AAPL"
    assert result["metadata"]["confidence"] == 0.9
    assert result["metadata"]["agent_id"] == "TestAgent"
    
    # Check if finding was stored
    mock_memory_manager.get_collection.assert_called_once_with("research_findings")
    mock_memory_manager.get_collection.return_value.add.assert_called_once()

@pytest.mark.asyncio
async def test_base_subagent_run_parsing_failure(mock_llm_failure, mock_memory_manager, input_data):
    agent = ResearchSubAgent(name="TestAgent", role="Tester", llm=mock_llm_failure, memory_manager=mock_memory_manager)
    result = await agent.run(input_data)
    
    assert "Error: Failed to parse LLM output" in result["content"]
    assert result["metadata"]["confidence"] == 0.0
    assert result["metadata"]["type"] == "error"
    
    # Check if the fallback finding was stored
    mock_memory_manager.get_collection.assert_called_once_with("research_findings")
    mock_memory_manager.get_collection.return_value.add.assert_called_once()

@pytest.mark.asyncio
async def test_technical_analysis_agent_prompt(mock_llm_success, mock_memory_manager, input_data):
    agent = TechnicalAnalysisSubAgent(llm=mock_llm_success, memory_manager=mock_memory_manager)
    prompt = agent._generate_research_prompt(**input_data)
    assert "technical analysis" in prompt
    assert '\"type\": \"technical\"' in prompt

@pytest.mark.asyncio
async def test_fundamental_analysis_agent_prompt(mock_llm_success, mock_memory_manager, input_data):
    agent = FundamentalAnalysisSubAgent(llm=mock_llm_success, memory_manager=mock_memory_manager)
    prompt = agent._generate_research_prompt(**input_data)
    assert "fundamental analysis" in prompt
    assert '\"type\": \"fundamental\"' in prompt

@pytest.mark.asyncio
async def test_sentiment_analysis_agent_prompt(mock_llm_success, mock_memory_manager, input_data):
    agent = SentimentAnalysisSubAgent(llm=mock_llm_success, memory_manager=mock_memory_manager)
    prompt = agent._generate_research_prompt(**input_data)
    assert "sentiment analysis" in prompt
    assert '\"type\": \"sentiment\"' in prompt

@pytest.mark.asyncio
async def test_pattern_mining_agent_prompt(mock_llm_success, mock_memory_manager, input_data):
    agent = PatternMiningSubAgent(llm=mock_llm_success, memory_manager=mock_memory_manager)
    prompt = agent._generate_research_prompt(**input_data)
    assert "pattern mining" in prompt
    assert '\"type\": \"pattern\"' in prompt

@pytest.mark.asyncio
async def test_market_research_agent_prompt(mock_llm_success, mock_memory_manager, input_data):
    agent = MarketResearchSubAgent(llm=mock_llm_success, memory_manager=mock_memory_manager)
    prompt = agent._generate_research_prompt(**input_data)
    assert "general market research" in prompt
    assert '\"type\": \"market\"' in prompt
