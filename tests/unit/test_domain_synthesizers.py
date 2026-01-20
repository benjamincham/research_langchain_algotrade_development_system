import pytest
from unittest.mock import AsyncMock, MagicMock
from src.agents.research.domain_synthesizers.base_synthesizer import DomainSynthesizer
from src.agents.research.domain_synthesizers.technical_synthesizer import TechnicalSynthesizer
from src.agents.research.domain_synthesizers.fundamental_synthesizer import FundamentalSynthesizer
from src.agents.research.domain_synthesizers.sentiment_synthesizer import SentimentSynthesizer
from src.memory.memory_manager import MemoryManager

# Mock the MemoryManager
@pytest.fixture
def mock_memory_manager():
    return MagicMock(spec=MemoryManager)

# Mock the LLM response for a successful synthesis
@pytest.fixture
def mock_llm_response_tech():
    return MagicMock(content='{"synthesis_text": "Tech synthesis complete.", "overall_confidence": 0.95, "type": "technical", "key_takeaways": ["Takeaway 1", "Takeaway 2"]}')

# Mock the LLM for a successful run
@pytest.fixture
def mock_llm_success(mock_llm_response_tech):
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = mock_llm_response_tech
    return mock_llm

# Input data for testing
@pytest.fixture
def input_data():
    return {
        "objective": "Determine if AAPL is a buy based on technicals.",
        "ticker": "AAPL",
        "timeframe": "1d",
        "raw_findings": [
            {"content": "RSI is overbought.", "metadata": {"agent_id": "RSI_Agent", "confidence": 0.8}},
            {"content": "MACD is crossing over.", "metadata": {"agent_id": "MACD_Agent", "confidence": 0.9}}
        ]
    }

@pytest.mark.asyncio
async def test_base_synthesizer_initialization(mock_llm_success, mock_memory_manager):
    agent = DomainSynthesizer(name="TestSynth", role="Tester", domain="Test", llm=mock_llm_success, memory_manager=mock_memory_manager)
    assert agent.name == "TestSynth"
    assert agent.domain == "Test"

@pytest.mark.asyncio
async def test_base_synthesizer_run_success(mock_llm_success, mock_memory_manager, input_data):
    agent = TechnicalSynthesizer(llm=mock_llm_success, memory_manager=mock_memory_manager)
    result = await agent.run(input_data)
    
    assert result["content"] == "Tech synthesis complete."
    assert result["metadata"]["ticker"] == "AAPL"
    assert result["metadata"]["confidence"] == 0.95
    assert result["metadata"]["agent_id"] == "TechnicalSynthesizer"
    assert "Takeaway 1" in result["metadata"]["tags"]

@pytest.mark.asyncio
async def test_technical_synthesizer_prompt(mock_llm_success, input_data):
    agent = TechnicalSynthesizer(llm=mock_llm_success)
    prompt = agent._generate_synthesis_prompt(**input_data)
    assert "Technical Domain Synthesizer" in prompt
    assert "RSI is overbought" in prompt
    assert '"type": "technical"' in prompt

@pytest.mark.asyncio
async def test_fundamental_synthesizer_prompt(mock_llm_success, input_data):
    agent = FundamentalSynthesizer(llm=mock_llm_success)
    prompt = agent._generate_synthesis_prompt(**input_data)
    assert "Fundamental Domain Synthesizer" in prompt
    assert "RSI is overbought" in prompt # Raw findings are passed regardless of domain
    assert '"type": "fundamental"' in prompt

@pytest.mark.asyncio
async def test_sentiment_synthesizer_prompt(mock_llm_success, input_data):
    agent = SentimentSynthesizer(llm=mock_llm_success)
    prompt = agent._generate_synthesis_prompt(**input_data)
    assert "Sentiment Domain Synthesizer" in prompt
    assert "RSI is overbought" in prompt # Raw findings are passed regardless of domain
    assert '"type": "sentiment"' in prompt
