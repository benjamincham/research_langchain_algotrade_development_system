import pytest
from unittest.mock import AsyncMock, MagicMock
from src.agents.validation_agent import ValidationAgent
import json

# Mock the LLM response for a successful validation
@pytest.fixture
def mock_llm_response_passed():
    return MagicMock(content=json.dumps({
        "validation_status": "PASSED",
        "confidence_score": 0.99,
        "report": "Data is valid and complete."
    }))

# Mock the LLM response for a failed validation
@pytest.fixture
def mock_llm_response_failed():
    return MagicMock(content=json.dumps({
        "validation_status": "FAILED",
        "confidence_score": 0.1,
        "report": "Missing key financial metrics."
    }))

# Mock the LLM response for invalid JSON
@pytest.fixture
def mock_llm_response_invalid_json():
    return MagicMock(content="This is not JSON.")

# Mock the LLM itself
@pytest.fixture
def mock_llm():
    return AsyncMock()

@pytest.mark.asyncio
async def test_validation_agent_initialization():
    agent = ValidationAgent(name="TestValidator")
    assert agent.name == "TestValidator"
    assert "Data Validation Agent" in agent.system_prompt

@pytest.mark.asyncio
async def test_validation_agent_run_passed(mock_llm, mock_llm_response_passed):
    mock_llm.ainvoke.return_value = mock_llm_response_passed
    agent = ValidationAgent(llm=mock_llm)
    input_data = {
        "data_to_validate": {"ticker": "AAPL", "price": 150.0},
        "context": {"source": "FinancialDataTool", "expected_format": "JSON"}
    }
    result = await agent.run(input_data)
    
    assert result["validation_status"] == "PASSED"
    assert result["confidence_score"] == 0.99
    assert "valid and complete" in result["report"]

@pytest.mark.asyncio
async def test_validation_agent_run_failed(mock_llm, mock_llm_response_failed):
    mock_llm.ainvoke.return_value = mock_llm_response_failed
    agent = ValidationAgent(llm=mock_llm)
    input_data = {
        "data_to_validate": {"ticker": "AAPL", "price": 150.0},
        "context": {"source": "FinancialDataTool", "expected_format": "JSON"}
    }
    result = await agent.run(input_data)
    
    assert result["validation_status"] == "FAILED"
    assert result["confidence_score"] == 0.1
    assert "Missing key financial metrics" in result["report"]

@pytest.mark.asyncio
async def test_validation_agent_run_invalid_json(mock_llm, mock_llm_response_invalid_json):
    mock_llm.ainvoke.return_value = mock_llm_response_invalid_json
    agent = ValidationAgent(llm=mock_llm)
    input_data = {
        "data_to_validate": {"ticker": "AAPL", "price": 150.0},
        "context": {"source": "FinancialDataTool", "expected_format": "JSON"}
    }
    result = await agent.run(input_data)
    
    assert result["validation_status"] == "ERROR"
    assert result["confidence_score"] == 0.0
    assert "failed to produce a valid JSON report" in result["report"]

@pytest.mark.asyncio
async def test_validation_agent_run_no_data(mock_llm):
    agent = ValidationAgent(llm=mock_llm)
    input_data = {"context": {"source": "Test"}}
    result = await agent.run(input_data)
    
    assert result["validation_status"] == "FAILED"
    assert result["confidence_score"] == 0.0
    assert "No data provided for validation" in result["report"]
