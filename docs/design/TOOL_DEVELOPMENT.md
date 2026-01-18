# Tool Development Meta-System Design

## Overview

The Tool Development Meta-System is a critical phase that runs BEFORE research and development. It generates, validates, and registers tools for metric calculation, ensuring that all tools are properly tested and documented before being used in the pipeline.

## Purpose

1. **Generate Custom Metric Tools**: Create Python functions for user-defined metrics
2. **Validate Tool Correctness**: Ensure tools produce accurate results
3. **Register Tools with Lifecycle**: Manage tool versions and deprecation
4. **Enable Systematic Expansion**: Allow safe addition of new tools

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    TOOL DEVELOPMENT META-SYSTEM                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                 METRIC TOOL GENERATOR AGENT                      │   │
│  │  • Analyze user-defined metric specifications                    │   │
│  │  • Generate Python functions                                     │   │
│  │  • Create tool wrappers with schemas                            │   │
│  │  • Generate test cases                                           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    TOOL VALIDATOR                                │   │
│  │  • Syntax validation                                             │   │
│  │  • Unit test execution                                           │   │
│  │  • Edge case testing                                             │   │
│  │  • Performance benchmarking                                      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    TOOL REGISTRY                                 │   │
│  │  • Version management                                            │   │
│  │  • Lifecycle states (Draft → Active → Deprecated)               │   │
│  │  • Schema storage                                                │   │
│  │  • Test case storage                                             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                 TOOLCHAIN VALIDATOR                              │   │
│  │  • Integration tests                                             │   │
│  │  • Workflow tests                                                │   │
│  │  • Regression tests                                              │   │
│  │  • Compatibility checks                                          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Tool Schema

```python
from pydantic import BaseModel
from typing import Any, Callable, Optional
from enum import Enum

class ToolLifecycle(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class ToolParameter(BaseModel):
    """Parameter definition for a tool."""
    name: str
    type: str  # "float", "int", "str", "list", "dict", "DataFrame"
    description: str
    required: bool = True
    default: Optional[Any] = None
    constraints: Optional[dict] = None  # e.g., {"min": 0, "max": 100}


class ToolDefinition(BaseModel):
    """Complete tool definition."""
    
    # Identification
    name: str
    version: str
    description: str
    category: str  # "metric", "indicator", "data", "analysis"
    
    # Schema
    parameters: list[ToolParameter]
    return_type: str
    return_description: str
    
    # Implementation
    code: str  # Python function code
    dependencies: list[str] = []  # Required packages
    
    # Lifecycle
    lifecycle: ToolLifecycle = ToolLifecycle.DRAFT
    created_at: str
    updated_at: str
    deprecated_at: Optional[str] = None
    deprecation_reason: Optional[str] = None
    
    # Testing
    test_cases: list[dict] = []
    validation_results: Optional[dict] = None


class ToolRegistry(BaseModel):
    """Registry of all tools."""
    tools: dict[str, ToolDefinition] = {}
    
    def register(self, tool: ToolDefinition) -> bool:
        """Register a new tool or update existing."""
        self.tools[tool.name] = tool
        return True
    
    def get(self, name: str, version: Optional[str] = None) -> Optional[ToolDefinition]:
        """Get a tool by name and optionally version."""
        tool = self.tools.get(name)
        if tool and version and tool.version != version:
            return None
        return tool
    
    def get_active(self) -> list[ToolDefinition]:
        """Get all active tools."""
        return [t for t in self.tools.values() if t.lifecycle == ToolLifecycle.ACTIVE]
    
    def deprecate(self, name: str, reason: str) -> bool:
        """Deprecate a tool."""
        if name in self.tools:
            self.tools[name].lifecycle = ToolLifecycle.DEPRECATED
            self.tools[name].deprecation_reason = reason
            return True
        return False
```

## Metric Tool Generator Agent

### Agent Design

```python
METRIC_GENERATOR_PROMPT = """
You are a Metric Tool Generator Agent. Your task is to create Python functions 
that calculate trading metrics based on user specifications.

For each metric specification, you must:
1. Understand the mathematical definition
2. Generate clean, efficient Python code
3. Include proper type hints and docstrings
4. Handle edge cases (empty data, NaN values, etc.)
5. Generate comprehensive test cases

Output format for each tool:
{
    "name": "metric_name",
    "description": "What this metric measures",
    "code": "def metric_name(...):\\n    ...",
    "parameters": [...],
    "return_type": "float",
    "test_cases": [
        {"input": {...}, "expected": ...},
        ...
    ]
}

Guidelines:
- Use numpy/pandas for efficient calculations
- Always validate inputs
- Return NaN for invalid calculations, don't raise exceptions
- Include at least 5 test cases covering normal and edge cases
"""
```

### Generation Flow

```python
class MetricToolGenerator:
    """Agent that generates metric calculation tools."""
    
    def __init__(self, llm):
        self.llm = llm
        self.validator = ToolValidator()
    
    async def generate_tool(
        self,
        metric_spec: dict
    ) -> ToolDefinition:
        """
        Generate a tool from metric specification.
        
        Args:
            metric_spec: User-defined metric specification
                {
                    "name": "custom_sharpe",
                    "description": "Modified Sharpe ratio with...",
                    "formula": "...",
                    "inputs": ["returns", "risk_free_rate"],
                    "output": "float"
                }
        
        Returns:
            Complete tool definition
        """
        # Generate initial code
        response = await self.llm.generate(
            prompt=self._build_prompt(metric_spec),
            system=METRIC_GENERATOR_PROMPT
        )
        
        tool_def = self._parse_response(response)
        
        # Validate generated code
        validation_result = await self.validator.validate(tool_def)
        
        if not validation_result.passed:
            # Retry with error feedback
            tool_def = await self._retry_with_feedback(
                metric_spec=metric_spec,
                tool_def=tool_def,
                errors=validation_result.errors
            )
        
        tool_def.validation_results = validation_result.to_dict()
        return tool_def
    
    async def generate_batch(
        self,
        metric_specs: list[dict]
    ) -> list[ToolDefinition]:
        """Generate multiple tools in batch."""
        tools = []
        for spec in metric_specs:
            tool = await self.generate_tool(spec)
            tools.append(tool)
        return tools
```

## Tool Validator

### Validation Pipeline

```python
class ToolValidator:
    """Validates generated tools."""
    
    async def validate(self, tool: ToolDefinition) -> ValidationResult:
        """
        Run full validation pipeline on a tool.
        """
        results = ValidationResult(tool_name=tool.name)
        
        # Stage 1: Syntax validation
        syntax_result = self._validate_syntax(tool.code)
        results.add_stage("syntax", syntax_result)
        if not syntax_result.passed:
            return results
        
        # Stage 2: Static analysis
        static_result = self._static_analysis(tool.code)
        results.add_stage("static_analysis", static_result)
        
        # Stage 3: Unit tests
        unit_result = await self._run_unit_tests(tool)
        results.add_stage("unit_tests", unit_result)
        if not unit_result.passed:
            return results
        
        # Stage 4: Edge case tests
        edge_result = await self._run_edge_case_tests(tool)
        results.add_stage("edge_cases", edge_result)
        
        # Stage 5: Performance benchmark
        perf_result = await self._benchmark_performance(tool)
        results.add_stage("performance", perf_result)
        
        results.passed = all(
            stage.passed for stage in results.stages.values()
        )
        
        return results
    
    def _validate_syntax(self, code: str) -> StageResult:
        """Check Python syntax."""
        try:
            ast.parse(code)
            return StageResult(passed=True)
        except SyntaxError as e:
            return StageResult(
                passed=False,
                errors=[f"Syntax error: {e}"]
            )
    
    def _static_analysis(self, code: str) -> StageResult:
        """Run static analysis (type checking, linting)."""
        issues = []
        
        # Check for common issues
        if "import os" in code or "import subprocess" in code:
            issues.append("Security: Potentially dangerous imports detected")
        
        if "eval(" in code or "exec(" in code:
            issues.append("Security: eval/exec usage detected")
        
        # Check for type hints
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.returns is None:
                    issues.append(f"Missing return type hint for {node.name}")
        
        return StageResult(
            passed=len([i for i in issues if "Security" in i]) == 0,
            warnings=issues
        )
    
    async def _run_unit_tests(self, tool: ToolDefinition) -> StageResult:
        """Execute unit tests."""
        # Create isolated execution environment
        exec_globals = {
            "np": np,
            "pd": pd,
            "math": math,
        }
        
        # Execute tool code
        exec(tool.code, exec_globals)
        func = exec_globals[tool.name]
        
        # Run test cases
        results = []
        for test in tool.test_cases:
            try:
                actual = func(**test["input"])
                expected = test["expected"]
                
                if isinstance(expected, float):
                    passed = abs(actual - expected) < 1e-6
                else:
                    passed = actual == expected
                
                results.append({
                    "test": test,
                    "actual": actual,
                    "passed": passed
                })
            except Exception as e:
                results.append({
                    "test": test,
                    "error": str(e),
                    "passed": False
                })
        
        all_passed = all(r["passed"] for r in results)
        return StageResult(
            passed=all_passed,
            details=results
        )
```

## Toolchain Validator

### Integration Testing

```python
class ToolchainValidator:
    """Validates tool combinations and workflows."""
    
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
    
    async def validate_toolchain(self) -> ToolchainReport:
        """
        Run comprehensive toolchain validation.
        """
        report = ToolchainReport()
        
        # Get all active tools
        tools = self.registry.get_active()
        
        # Stage 1: Individual tool validation
        for tool in tools:
            result = await self._validate_tool(tool)
            report.add_tool_result(tool.name, result)
        
        # Stage 2: Integration tests
        integration_results = await self._run_integration_tests()
        report.add_integration_results(integration_results)
        
        # Stage 3: Workflow tests
        workflow_results = await self._run_workflow_tests()
        report.add_workflow_results(workflow_results)
        
        # Stage 4: Regression tests
        regression_results = await self._run_regression_tests()
        report.add_regression_results(regression_results)
        
        return report
    
    async def _run_integration_tests(self) -> list[TestResult]:
        """Test tool combinations."""
        tests = [
            # Test: Calculate Sharpe from returns
            {
                "name": "sharpe_from_returns",
                "steps": [
                    ("fetch_market_data", {"symbol": "SPY", "period": "1y"}),
                    ("calculate_returns", {"prices": "$prev.close"}),
                    ("calculate_sharpe", {"returns": "$prev"})
                ],
                "expected_type": "float"
            },
            # Test: Full backtest metrics pipeline
            {
                "name": "backtest_metrics_pipeline",
                "steps": [
                    ("run_backtest", {"strategy": "test_strategy"}),
                    ("calculate_all_metrics", {"results": "$prev"}),
                    ("evaluate_quality_gate", {"metrics": "$prev"})
                ],
                "expected_keys": ["passed", "score", "feedback"]
            }
        ]
        
        results = []
        for test in tests:
            result = await self._execute_integration_test(test)
            results.append(result)
        
        return results
    
    async def _run_regression_tests(self) -> list[TestResult]:
        """
        Run regression tests to ensure tool updates don't break existing functionality.
        """
        # Load saved regression test cases
        regression_cases = await self._load_regression_cases()
        
        results = []
        for case in regression_cases:
            tool = self.registry.get(case["tool_name"])
            if tool:
                result = await self._execute_regression_test(tool, case)
                results.append(result)
        
        return results
```

## Tool Lifecycle Management

### Lifecycle States

```
┌─────────┐     validate     ┌─────────┐     deprecate    ┌────────────┐
│  DRAFT  │ ───────────────► │ ACTIVE  │ ────────────────► │ DEPRECATED │
└─────────┘                  └─────────┘                   └────────────┘
     │                            │                              │
     │                            │                              │
     │         reject             │         archive              │
     └────────────────────────────┴──────────────────────────────┘
                                  │
                                  ▼
                            ┌──────────┐
                            │ ARCHIVED │
                            └──────────┘
```

### Lifecycle Manager

```python
class ToolLifecycleManager:
    """Manages tool lifecycle transitions."""
    
    def __init__(self, registry: ToolRegistry, validator: ToolValidator):
        self.registry = registry
        self.validator = validator
    
    async def activate_tool(self, tool_name: str) -> ActivationResult:
        """
        Transition tool from DRAFT to ACTIVE.
        
        Requires:
        - All validation tests pass
        - No security issues
        - Documentation complete
        """
        tool = self.registry.get(tool_name)
        if not tool:
            return ActivationResult(success=False, error="Tool not found")
        
        if tool.lifecycle != ToolLifecycle.DRAFT:
            return ActivationResult(
                success=False, 
                error=f"Tool must be in DRAFT state, currently: {tool.lifecycle}"
            )
        
        # Run validation
        validation = await self.validator.validate(tool)
        if not validation.passed:
            return ActivationResult(
                success=False,
                error="Validation failed",
                details=validation.to_dict()
            )
        
        # Activate
        tool.lifecycle = ToolLifecycle.ACTIVE
        tool.updated_at = datetime.now().isoformat()
        self.registry.register(tool)
        
        return ActivationResult(success=True)
    
    async def deprecate_tool(
        self,
        tool_name: str,
        reason: str,
        replacement: Optional[str] = None
    ) -> DeprecationResult:
        """
        Deprecate a tool with optional replacement.
        
        Deprecated tools:
        - Emit warnings when used
        - Remain usable for 30 days
        - Then transition to ARCHIVED
        """
        tool = self.registry.get(tool_name)
        if not tool:
            return DeprecationResult(success=False, error="Tool not found")
        
        tool.lifecycle = ToolLifecycle.DEPRECATED
        tool.deprecated_at = datetime.now().isoformat()
        tool.deprecation_reason = reason
        
        if replacement:
            tool.deprecation_reason += f" Use {replacement} instead."
        
        self.registry.register(tool)
        
        # Schedule archival
        await self._schedule_archival(tool_name, days=30)
        
        return DeprecationResult(success=True)
```

## Built-in Tools

The system comes with pre-built tools that are automatically registered:

### Data Tools

| Tool | Description | Category |
|------|-------------|----------|
| `fetch_market_data` | Get OHLCV data via yfinance | data |
| `fetch_fundamentals` | Get fundamental data | data |
| `calculate_returns` | Calculate returns from prices | data |

### Metric Tools

| Tool | Description | Category |
|------|-------------|----------|
| `calculate_sharpe` | Sharpe ratio | metric |
| `calculate_sortino` | Sortino ratio | metric |
| `calculate_max_drawdown` | Maximum drawdown | metric |
| `calculate_profit_factor` | Profit factor | metric |
| `calculate_win_rate` | Win rate | metric |
| `calculate_var` | Value at Risk | metric |

### Analysis Tools

| Tool | Description | Category |
|------|-------------|----------|
| `run_backtest` | Execute Backtrader backtest | analysis |
| `walk_forward_optimize` | Walk-forward optimization | analysis |
| `monte_carlo_simulate` | Monte Carlo simulation | analysis |

## Integration with Pipeline

```python
# In main workflow
async def tool_development_phase(state: PipelineState) -> PipelineState:
    """Execute tool development phase."""
    
    generator = MetricToolGenerator(llm=get_llm())
    validator = ToolValidator()
    lifecycle = ToolLifecycleManager(registry, validator)
    
    # Generate tools for custom metrics
    if state["user_config"].custom_metric_definitions:
        for metric_spec in state["user_config"].custom_metric_definitions:
            # Generate tool
            tool = await generator.generate_tool(metric_spec)
            
            # Register as draft
            registry.register(tool)
            
            # Activate if validation passes
            result = await lifecycle.activate_tool(tool.name)
            if not result.success:
                # Log warning but continue
                logger.warning(f"Tool {tool.name} activation failed: {result.error}")
    
    # Validate entire toolchain
    toolchain_report = await ToolchainValidator(registry).validate_toolchain()
    
    return {
        **state,
        "tool_registry": registry.to_dict(),
        "tool_validation_results": toolchain_report.to_dict()
    }
```
