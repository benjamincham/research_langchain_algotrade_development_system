import pytest
from pydantic import BaseModel, Field
from src.tools.tool_registry import BaseTool, ToolInputSchema, ToolRegistry, ExampleTool, ExampleToolInput
from typing import Type
from unittest.mock import MagicMock

# --- Custom Tool for Testing ---

class TestToolInput(ToolInputSchema):
    """Input schema for the TestTool."""
    param1: str = Field(..., description="A required string parameter.")
    param2: int = Field(10, description="An optional integer parameter.")

class TestTool(BaseTool):
    """A custom tool for testing purposes."""
    
    name: str = "custom_test_tool"
    description: str = "A tool for testing the registry."
    input_schema: Type[ToolInputSchema] = TestToolInput

    async def run(self, param1: str, param2: int = 10) -> str:
        return f"Ran with {param1} and {param2}"

# --- Tests for BaseTool and ToolInputSchema ---

def test_base_tool_initialization():
    tool = TestTool()
    assert tool.name == "custom_test_tool"
    assert tool.description == "A tool for testing the registry."
    assert tool.input_schema == TestToolInput

@pytest.mark.asyncio
async def test_base_tool_run_not_implemented():
    # BaseTool run should raise NotImplementedError
    class IncompleteTool(BaseTool):
        name: str = "incomplete"
        description: str = "desc"
        input_schema: Type[ToolInputSchema] = ToolInputSchema
        
    tool = IncompleteTool()
    with pytest.raises(NotImplementedError):
        await tool.run()

@pytest.mark.asyncio
async def test_custom_tool_run():
    tool = TestTool()
    result = await tool.run(param1="hello", param2=5)
    assert result == "Ran with hello and 5"
    result_default = await tool.run(param1="world")
    assert result_default == "Ran with world and 10"

# --- Tests for ToolRegistry ---

@pytest.fixture
def registry():
    return ToolRegistry()

def test_registry_initialization(registry):
    assert registry._tools == {}

def test_register_and_get_tool(registry):
    tool = TestTool()
    registry.register_tool(tool)
    
    retrieved_tool = registry.get_tool("custom_test_tool")
    assert retrieved_tool == tool

def test_register_tool_not_base_tool(registry):
    with pytest.raises(TypeError):
        registry.register_tool(MagicMock())

def test_get_tool_not_found(registry):
    with pytest.raises(ValueError):
        registry.get_tool("non_existent_tool")

def test_get_all_tools(registry):
    tool1 = TestTool()
    tool2 = ExampleTool()
    registry.register_tool(tool1)
    registry.register_tool(tool2)
    
    all_tools = registry.get_all_tools()
    assert len(all_tools) == 2
    assert tool1 in all_tools
    assert tool2 in all_tools

def test_get_tool_definitions(registry):
    tool = TestTool()
    registry.register_tool(tool)
    
    definitions = registry.get_tool_definitions()
    assert len(definitions) == 1
    definition = definitions[0]
    
    assert definition["type"] == "function"
    assert definition["function"]["name"] == "custom_test_tool"
    assert "required" in definition["function"]["parameters"]
    assert "param1" in definition["function"]["parameters"]["properties"]
    assert "param2" in definition["function"]["parameters"]["properties"]
