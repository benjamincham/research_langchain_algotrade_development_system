# Quality Gate Schemas Documentation

**Document**: Quality Gate Agent Input/Output Schemas  
**Created**: 2026-01-18  
**Status**: Complete  
**Implementation**: `src/agents/quality_gate/schemas.py`

## Overview

This document provides comprehensive documentation for the input and output schemas used by the Quality Gate Agent and its two sub-agents (Failure Analysis Agent and Trajectory Analyzer Agent). These schemas ensure type-safe, validated data flow throughout the quality gate evaluation process.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      QUALITY_GATE NODE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: WorkflowState (from LangGraph)                          │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  1. Evaluate all variants against quality criteria       │ │
│  │     → QualityGateResult[] (passed/failed for each)       │ │
│  └───────────────────────────────────────────────────────────┘ │
│                          ↓                                      │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  2. If any variant passed → SUCCESS                       │ │
│  │     Else → Invoke Failure Analysis Agent                  │ │
│  └───────────────────────────────────────────────────────────┘ │
│                          ↓                                      │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  FAILURE ANALYSIS AGENT                                   │ │
│  │  Input: FailureAnalysisInput                              │ │
│  │  Output: FailureAnalysisOutput                            │ │
│  │    - Classification (5 categories)                        │ │
│  │    - Recommendation (5 actions)                           │ │
│  │    - Specific actions to take                             │ │
│  └───────────────────────────────────────────────────────────┘ │
│                          ↓                                      │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  3. If experiment_history >= 2 iterations                 │ │
│  │     → Invoke Trajectory Analyzer Agent                    │ │
│  └───────────────────────────────────────────────────────────┘ │
│                          ↓                                      │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  TRAJECTORY ANALYZER AGENT                                │ │
│  │  Input: TrajectoryAnalysisInput                           │ │
│  │  Output: TrajectoryAnalysisOutput                         │ │
│  │    - Metric trajectories (IMPROVING/DECLINING/etc.)       │ │
│  │    - Convergence analysis                                 │ │
│  │    - Parameter impact analysis                            │ │
│  └───────────────────────────────────────────────────────────┘ │
│                          ↓                                      │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  4. Combine analyses → Determine next_action              │ │
│  │     → QualityGateNodeOutput                               │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
│  Output: QualityGateNodeOutput (merged into WorkflowState)      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Schema Catalog

### Core Schemas

| Schema | Purpose | Used By |
|--------|---------|---------|
| `StrategyVariant` | Represents a single strategy variant | Failure Analysis Agent |
| `BacktestMetrics` | Backtest results for a variant | Failure Analysis, Trajectory Analyzer |
| `QualityGateResult` | Quality gate evaluation result | Quality Gate Node |
| `ResearchFinding` | Research finding from research swarm | Failure Analysis Agent |
| `IterationHistory` | Single iteration record | Failure Analysis, Trajectory Analyzer |

### Agent Input/Output Schemas

| Agent | Input Schema | Output Schema |
|-------|--------------|---------------|
| Failure Analysis Agent | `FailureAnalysisInput` | `FailureAnalysisOutput` |
| Trajectory Analyzer Agent | `TrajectoryAnalysisInput` | `TrajectoryAnalysisOutput` |
| Quality Gate Node | WorkflowState (dict) | `QualityGateNodeOutput` |

---

## Failure Analysis Agent Schemas

### FailureAnalysisInput

**Purpose**: Provides complete context for failure analysis including strategy code, backtest results, research findings, and iteration history.

**Fields**:

```python
class FailureAnalysisInput(BaseModel):
    # Strategy information
    strategy_variants: List[StrategyVariant]  # All tested variants
    
    # Backtest results
    backtest_results: List[BacktestMetrics]  # Metrics for all variants
    
    # Quality gate evaluation
    quality_gate_results: List[QualityGateResult]  # Pass/fail for each variant
    
    # Research context
    research_findings: List[ResearchFinding]  # Findings from research swarm
    
    # Quality criteria (thresholds)
    quality_criteria: Dict[str, float]  # e.g., {"sharpe_ratio": 1.0, "max_drawdown": 0.20}
    
    # Iteration context
    current_iteration: int  # Current strategy iteration (>= 1)
    max_iterations: int  # Maximum allowed iterations
    iteration_history: List[IterationHistory]  # Previous iterations
    
    # Market context
    ticker: str  # e.g., "AAPL"
    timeframe: str  # e.g., "1d"
```

**Validation**:
- All lists must be non-empty
- `backtest_results` must match `strategy_variants` by `variant_id`
- `current_iteration` must be >= 1
- `max_iterations` must be >= 1

**Example**:

```python
failure_input = FailureAnalysisInput(
    strategy_variants=[
        StrategyVariant(
            variant_id="v1",
            code="class MyStrategy(bt.Strategy): ...",
            parameters={"rsi_period": 14, "position_size": 0.1},
            description="RSI momentum strategy"
        )
    ],
    backtest_results=[
        BacktestMetrics(
            variant_id="v1",
            sharpe_ratio=0.75,
            total_return=0.12,
            max_drawdown=0.18,
            win_rate=0.52,
            total_trades=45
        )
    ],
    quality_gate_results=[
        QualityGateResult(
            variant_id="v1",
            passed=False,
            gate_score=0.65,
            failed_criteria=["sharpe_ratio"],
            criteria_scores={"sharpe_ratio": 0.75, "max_drawdown": 0.90}
        )
    ],
    research_findings=[...],
    quality_criteria={"sharpe_ratio": 1.0, "max_drawdown": 0.20},
    current_iteration=1,
    max_iterations=5,
    iteration_history=[],
    ticker="AAPL",
    timeframe="1d"
)
```

---

### FailureAnalysisOutput

**Purpose**: Provides classification, root cause analysis, recommendation, and specific actions for addressing strategy failures.

**Fields**:

```python
class FailureAnalysisOutput(BaseModel):
    # Classification (5 categories)
    classification: Literal[
        "PARAMETER_ISSUE",       # Close to passing, needs parameter tuning
        "ALGORITHM_BUG",         # Implementation error (e.g., incorrect RSI calculation)
        "DESIGN_FLAW",           # Fundamental design issue (e.g., missing regime awareness)
        "RESEARCH_GAP",          # Wrong research hypothesis
        "FUNDAMENTAL_IMPOSSIBILITY"  # Alpha doesn't exist
    ]
    
    # Root cause analysis
    root_cause: str  # Detailed explanation
    
    # Statistical assessment
    statistical_assessment: Dict[str, Any]  # Distance from threshold, trajectory, etc.
    
    # Recommendation (5 actions)
    recommendation: Literal[
        "TUNE_PARAMETERS",    # Adjust parameters (Tier 1)
        "FIX_BUG",            # Fix implementation bug (Tier 1)
        "REFINE_ALGORITHM",   # Redesign strategy logic (Tier 1)
        "REFINE_RESEARCH",    # Go back to research swarm (Tier 2)
        "ABANDON"             # Give up (Tier 3)
    ]
    
    # Specific actions
    specific_actions: List[str]  # Concrete steps to take
    
    # Confidence and reasoning
    confidence: float  # 0.0 to 1.0
    reasoning: str  # Detailed reasoning
    
    # Likelihood of success
    likelihood_of_success: float  # 0.0 to 1.0
    
    # Iteration recommendation
    should_continue: bool  # Continue or abandon?
    estimated_iterations_needed: Optional[int]  # How many more iterations?
```

**Example**:

```python
failure_output = FailureAnalysisOutput(
    classification="PARAMETER_ISSUE",
    root_cause="Sharpe ratio is 0.75, just 0.25 below threshold of 1.0. Strategy shows consistent positive returns but needs optimization.",
    statistical_assessment={
        "distance_from_threshold": {"sharpe_ratio": 0.25, "max_drawdown": 0.02},
        "improvement_trajectory": "STABLE",
        "consistency": "CONSISTENT",
        "alignment_with_research": "GOOD"
    },
    recommendation="TUNE_PARAMETERS",
    specific_actions=[
        "Increase position size from 0.1 to 0.15 to boost returns",
        "Add volatility filter (ATR > 2.0) to reduce drawdown",
        "Tighten stop-loss from 3% to 2.5%"
    ],
    confidence=0.85,
    reasoning="Strategy is very close to passing. The research hypothesis (momentum in AAPL) is sound, and the implementation is correct. Parameter tuning should push Sharpe ratio above 1.0.",
    likelihood_of_success=0.75,
    should_continue=True,
    estimated_iterations_needed=2
)
```

---

## Trajectory Analyzer Agent Schemas

### TrajectoryAnalysisInput

**Purpose**: Provides experiment history and current state for trajectory analysis to determine if iterations are converging toward success.

**Fields**:

```python
class TrajectoryAnalysisInput(BaseModel):
    # Experiment history (minimum 2 iterations required)
    experiment_history: List[IterationHistory]  # Complete history
    
    # Current state
    current_iteration: int  # >= 2
    current_best_metrics: BacktestMetrics  # Current best variant
    
    # Quality criteria
    quality_criteria: Dict[str, float]  # Thresholds
    
    # Iteration limits
    max_strategy_iterations: int
    max_research_iterations: int
    max_total_iterations: int
    
    # Current iteration counts
    strategy_iteration: int
    research_iteration: int
    total_iterations: int
```

**Validation**:
- `experiment_history` must have >= 2 items
- `current_iteration` must be >= 2

**Example**:

```python
trajectory_input = TrajectoryAnalysisInput(
    experiment_history=[
        IterationHistory(
            iteration=1,
            timestamp="2026-01-18T10:00:00",
            best_sharpe=0.75,
            best_variant_id="v1",
            action_taken="TUNE_PARAMETERS",
            parameters_changed=["position_size"],
            improvement=None
        ),
        IterationHistory(
            iteration=2,
            timestamp="2026-01-18T10:15:00",
            best_sharpe=0.85,
            best_variant_id="v2",
            action_taken="TUNE_PARAMETERS",
            parameters_changed=["position_size", "stop_loss"],
            improvement=0.10
        )
    ],
    current_iteration=2,
    current_best_metrics=BacktestMetrics(
        variant_id="v2",
        sharpe_ratio=0.85,
        total_return=0.15,
        max_drawdown=0.16,
        win_rate=0.54,
        total_trades=52
    ),
    quality_criteria={"sharpe_ratio": 1.0, "max_drawdown": 0.20},
    max_strategy_iterations=5,
    max_research_iterations=3,
    max_total_iterations=15,
    strategy_iteration=2,
    research_iteration=0,
    total_iterations=2
)
```

---

### TrajectoryAnalysisOutput

**Purpose**: Provides trajectory analysis including metric trends, convergence status, parameter impact, and recommendations.

**Fields**:

```python
class TrajectoryAnalysisOutput(BaseModel):
    # Metric trajectories
    metric_trajectories: List[MetricTrajectory]  # Trajectory for each metric
    
    # Convergence analysis
    convergence_analysis: ConvergenceAnalysis  # Converging/diverging/etc.
    
    # Parameter impact
    parameter_impacts: List[ParameterImpactAnalysis]  # Impact of each parameter
    
    # Overall assessment
    overall_trajectory: Literal["IMPROVING", "DECLINING", "STABLE", "OSCILLATING"]
    
    # Recommendation
    recommendation: Literal["CONTINUE", "PIVOT", "ABANDON"]
    reasoning: str
    
    # Confidence
    confidence: float  # 0.0 to 1.0
    
    # Insights
    key_insights: List[str]  # Key findings
    
    # Predictions
    predicted_next_iteration_metrics: Optional[Dict[str, float]]  # Predicted values
```

**Example**:

```python
trajectory_output = TrajectoryAnalysisOutput(
    metric_trajectories=[
        MetricTrajectory(
            metric_name="sharpe_ratio",
            values=[0.75, 0.85],
            trend="IMPROVING",
            improvement_rate=0.10,
            volatility=0.05,
            distance_to_target=0.15
        ),
        MetricTrajectory(
            metric_name="max_drawdown",
            values=[0.18, 0.16],
            trend="IMPROVING",
            improvement_rate=-0.02,
            volatility=0.01,
            distance_to_target=0.04
        )
    ],
    convergence_analysis=ConvergenceAnalysis(
        status="CONVERGING",
        confidence=0.80,
        estimated_iterations_to_convergence=2,
        reasoning="Sharpe ratio improving steadily at +0.10 per iteration. At this rate, will reach 1.0 in 2 more iterations."
    ),
    parameter_impacts=[
        ParameterImpactAnalysis(
            parameter_name="position_size",
            impact_score=0.85,
            direction="POSITIVE",
            recommendation="Continue increasing position size gradually"
        ),
        ParameterImpactAnalysis(
            parameter_name="stop_loss",
            impact_score=0.60,
            direction="POSITIVE",
            recommendation="Tightening stop-loss reduced drawdown, continue optimizing"
        )
    ],
    overall_trajectory="IMPROVING",
    recommendation="CONTINUE",
    reasoning="Strong convergence pattern. Sharpe ratio improving steadily, drawdown decreasing. Parameter changes are effective. High likelihood of success within 2 more iterations.",
    confidence=0.85,
    key_insights=[
        "Sharpe ratio improving steadily (+0.10 per iteration)",
        "Max drawdown decreasing (-0.02 per iteration)",
        "Position size changes have highest impact on performance",
        "Strategy is converging toward quality gate thresholds"
    ],
    predicted_next_iteration_metrics={
        "sharpe_ratio": 0.95,
        "max_drawdown": 0.14
    }
)
```

---

## Quality Gate Node Output Schema

### QualityGateNodeOutput

**Purpose**: Complete output from the quality_gate node that gets merged back into WorkflowState for conditional routing.

**Fields**:

```python
class QualityGateNodeOutput(BaseModel):
    # Best variant selection
    best_variant_id: Optional[str]  # ID of best variant (None if all failed)
    best_metrics: Optional[BacktestMetrics]  # Metrics of best variant
    
    # Quality gate results
    any_variant_passed: bool  # Did any variant pass?
    quality_gate_results: List[QualityGateResult]  # Results for all variants
    
    # Failure analysis (if all failed)
    failure_analysis: Optional[FailureAnalysisOutput]
    
    # Trajectory analysis (if history >= 2)
    trajectory_analysis: Optional[TrajectoryAnalysisOutput]
    
    # Routing decision
    next_action: Literal[
        "SUCCESS",
        "TUNE_PARAMETERS",
        "FIX_BUG",
        "REFINE_ALGORITHM",
        "REFINE_RESEARCH",
        "ABANDON"
    ]
    
    # Iteration counters
    total_iterations: int  # Incremented
    
    # Status
    final_status: Optional[Literal["SUCCESS", "ABANDONED"]]
    
    # Reasoning
    decision_reasoning: str  # Explanation of routing decision
```

**Example**:

```python
quality_gate_output = QualityGateNodeOutput(
    best_variant_id="v2",
    best_metrics=BacktestMetrics(
        variant_id="v2",
        sharpe_ratio=0.85,
        total_return=0.15,
        max_drawdown=0.16,
        win_rate=0.54,
        total_trades=52
    ),
    any_variant_passed=False,
    quality_gate_results=[...],
    failure_analysis=FailureAnalysisOutput(...),
    trajectory_analysis=TrajectoryAnalysisOutput(...),
    next_action="TUNE_PARAMETERS",
    total_iterations=3,
    final_status=None,
    decision_reasoning="Failure analysis classified as PARAMETER_ISSUE. Trajectory analysis shows CONVERGING pattern. Recommendation: TUNE_PARAMETERS. Estimated 2 more iterations needed."
)
```

---

## Helper Functions

Two helper functions are provided to extract agent inputs from WorkflowState:

### create_failure_analysis_input_from_state()

```python
def create_failure_analysis_input_from_state(state: dict) -> FailureAnalysisInput:
    """Extract FailureAnalysisInput from WorkflowState"""
    return FailureAnalysisInput(
        strategy_variants=[...],
        backtest_results=[...],
        quality_gate_results=[...],
        research_findings=[...],
        quality_criteria=state["quality_criteria"],
        current_iteration=state["strategy_iteration"],
        max_iterations=state["max_strategy_iterations"],
        iteration_history=state.get("experiment_history", []),
        ticker=state["ticker"],
        timeframe=state["timeframe"]
    )
```

### create_trajectory_analysis_input_from_state()

```python
def create_trajectory_analysis_input_from_state(state: dict) -> TrajectoryAnalysisInput:
    """Extract TrajectoryAnalysisInput from WorkflowState"""
    best_result = max(
        state["backtest_results"],
        key=lambda x: x.get("sharpe_ratio", -999)
    )
    
    return TrajectoryAnalysisInput(
        experiment_history=state["experiment_history"],
        current_iteration=state["total_iterations"],
        current_best_metrics=BacktestMetrics(**best_result),
        quality_criteria=state["quality_criteria"],
        max_strategy_iterations=state["max_strategy_iterations"],
        max_research_iterations=state["max_research_iterations"],
        max_total_iterations=state["max_total_iterations"],
        strategy_iteration=state["strategy_iteration"],
        research_iteration=state["research_iteration"],
        total_iterations=state["total_iterations"]
    )
```

---

## Data Flow Example

### Complete Quality Gate Node Execution

```python
async def quality_gate_node(state: WorkflowState) -> QualityGateNodeOutput:
    """Quality gate node implementation"""
    
    # 1. Evaluate all variants against quality criteria
    quality_gate_results = []
    for variant, metrics in zip(state["strategy_variants"], state["backtest_results"]):
        result = evaluate_quality_gates(metrics, state["quality_criteria"])
        quality_gate_results.append(result)
    
    # 2. Check if any variant passed
    any_passed = any(r.passed for r in quality_gate_results)
    
    if any_passed:
        # SUCCESS path
        best_variant = select_best_variant(quality_gate_results)
        return QualityGateNodeOutput(
            best_variant_id=best_variant.variant_id,
            best_metrics=best_variant.metrics,
            any_variant_passed=True,
            quality_gate_results=quality_gate_results,
            failure_analysis=None,
            trajectory_analysis=None,
            next_action="SUCCESS",
            total_iterations=state["total_iterations"] + 1,
            final_status="SUCCESS",
            decision_reasoning="Variant passed all quality gates"
        )
    
    # 3. All variants failed - invoke Failure Analysis Agent
    failure_input = create_failure_analysis_input_from_state(state)
    failure_analysis = await failure_analysis_agent.invoke(failure_input)
    
    # 4. If history >= 2, invoke Trajectory Analyzer Agent
    trajectory_analysis = None
    if len(state.get("experiment_history", [])) >= 2:
        trajectory_input = create_trajectory_analysis_input_from_state(state)
        trajectory_analysis = await trajectory_analyzer_agent.invoke(trajectory_input)
    
    # 5. Combine analyses to determine next_action
    next_action = determine_next_action(failure_analysis, trajectory_analysis)
    
    # 6. Return complete output
    return QualityGateNodeOutput(
        best_variant_id=None,
        best_metrics=None,
        any_variant_passed=False,
        quality_gate_results=quality_gate_results,
        failure_analysis=failure_analysis,
        trajectory_analysis=trajectory_analysis,
        next_action=next_action,
        total_iterations=state["total_iterations"] + 1,
        final_status="ABANDONED" if next_action == "ABANDON" else None,
        decision_reasoning=f"Failure: {failure_analysis.classification}. Trajectory: {trajectory_analysis.overall_trajectory if trajectory_analysis else 'N/A'}. Action: {next_action}"
    )
```

---

## Implementation Status

**File**: `src/agents/quality_gate/schemas.py`  
**Status**: ✅ Complete and tested  
**Tests**: Example usage in `__main__` block passes  
**Validation**: All Pydantic validators working correctly

## Next Steps

1. Implement `failure_analysis_agent.py` using `FailureAnalysisInput` and `FailureAnalysisOutput`
2. Implement `trajectory_analyzer_agent.py` using `TrajectoryAnalysisInput` and `TrajectoryAnalysisOutput`
3. Implement `quality_gate_node()` function in `src/workflows/workflow.py`
4. Write unit tests for all schemas
5. Write integration tests for quality_gate node

## Related Documents

- [AGENT_CATALOG.md](./AGENT_CATALOG.md) - Complete agent definitions
- [FAILURE_ANALYSIS_SYSTEM.md](./FAILURE_ANALYSIS_SYSTEM.md) - Failure analysis design
- [EXPERIMENT_TRACKING.md](./EXPERIMENT_TRACKING.md) - Experiment tracking design
- [LANGGRAPH_IMPLEMENTATION.md](./LANGGRAPH_IMPLEMENTATION.md) - LangGraph workflow implementation
