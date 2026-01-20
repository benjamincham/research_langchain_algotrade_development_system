from typing import Dict, Type, Any, List
from pydantic import BaseModel, Field
from loguru import logger

# --- 1. Tool Schema Definition ---

class ToolInputSchema(BaseModel):
    """Base schema for tool input parameters."""
    pass

class BaseTool(BaseModel):
    """Base class for all tools in the system."""
    name: str = Field(..., description="The unique name of the tool.")
    description: str = Field(..., description="A detailed description of the tool's purpose and functionality.")
    input_schema: Type[ToolInputSchema] = Field(..., description="The Pydantic schema for the tool's input parameters.")

    async def run(self, **kwargs) -> Any:
        """
        The core logic of the tool. Must be implemented by subclasses.
        
        Args:
            **kwargs: Parameters matching the tool's input_schema.
            
        Returns:
            The result of the tool's operation.
        """
        raise NotImplementedError("The 'run' method must be implemented by subclasses.")

# --- 2. Tool Registry Implementation ---

class ToolRegistry:
    """
    A central registry for managing and providing access to all available tools.
    """
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        logger.info("ToolRegistry initialized.")

    def register_tool(self, tool_instance: BaseTool):
        """
        Registers a tool instance with the registry.
        
        Args:
            tool_instance: An instance of a class inheriting from BaseTool.
        """
        if not isinstance(tool_instance, BaseTool):
            raise TypeError("Only instances of BaseTool subclasses can be registered.")
            
        if tool_instance.name in self._tools:
            logger.warning(f"Tool '{tool_instance.name}' is already registered. Overwriting.")
            
        self._tools[tool_instance.name] = tool_instance
        logger.info(f"Tool '{tool_instance.name}' registered successfully.")

    def get_tool(self, name: str) -> BaseTool:
        """
        Retrieves a tool instance by its unique name.
        
        Args:
            name: The unique name of the tool.
            
        Returns:
            The requested BaseTool instance.
            
        Raises:
            ValueError: If the tool is not found.
        """
        if name not in self._tools:
            raise ValueError(f"Tool '{name}' not found in the registry.")
        return self._tools[name]

    def get_all_tools(self) -> List[BaseTool]:
        """
        Returns a list of all registered tool instances.
        """
        return list(self._tools.values())

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        Returns a list of tool definitions suitable for LLM function calling.
        """
        definitions = []
        for tool in self._tools.values():
            # Convert Pydantic schema to a format suitable for LLM tool calling (e.g., OpenAI format)
            schema_dict = tool.input_schema.model_json_schema()
            
            definitions.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": schema_dict
                }
            })
        return definitions

# --- 3. Example Tool for Testing ---

class ExampleToolInput(ToolInputSchema):
    """Input schema for the ExampleTool."""
    query: str = Field(..., description="The search query for the example tool.")
    limit: int = Field(1, description="The maximum number of results to return.")

class ExampleTool(BaseTool):
    """A simple example tool to demonstrate the registry functionality."""
    
    name: str = "example_search"
    description: str = "A tool for performing a simple, mocked search operation."
    input_schema: Type[ToolInputSchema] = ExampleToolInput

    async def run(self, query: str, limit: int = 1) -> Dict[str, Any]:
        """Mocks a search operation."""
        logger.info(f"ExampleTool running with query: '{query}' and limit: {limit}")
        return {
            "query": query,
            "limit": limit,
            "result": [f"Mocked result {i} for {query}" for i in range(limit)]
        }
