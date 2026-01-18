"""
Pydantic schemas for Quality Gate sub-agents (Failure Analysis and Trajectory Analyzer).

These schemas ensure type safety and validation for data flow between the quality_gate node
and its sub-agents.
"""

from typing import Dict, List, Optional, Literal, Any
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


# ============================================================================
# FAILURE ANALYSIS AGENT SCHEMAS
# ============================================================================

class StrategyVariant(BaseModel):
    """Schema for a single strategy variant"""
    variant_id: str = Field(..., description="Unique identifier for this variant")
    code: str = Field(..., description="Complete strategy code")
    parameters: Dict[str, Any] = Field(..., description="Strategy parameters")
    description: str = Field(..., description="Human-readable description")


class BacktestMetrics(BaseModel):
    """Schema for backtest metrics of a single variant"""
    variant_id: str = Field(..., description="Variant identifier")
    sharpe_ratio: Optional[float] = Field(None, description="Sharpe ratio")
    total_return: Optional[float] = Field(None, description="Total return (e.g., 0.15 = 15%)")
    max_drawdown: Optional[float] = Field(None, description="Max drawdown (e.g., 0.20 = 20%)")
    win_rate: Optional[float] = Field(None, description="Win rate (e.g., 0.55 = 55%)")
    total_trades: Optional[int] = Field(None, description="Total number of trades")
    profit_factor: Optional[float] = Field(None, description="Profit factor")
    avg_trade_duration: Optional[float] = Field(None, description="Average trade duration in days")
    execution_error: Optional[str] = Field(None, description="Error message if backtest failed")
    
    @field_validator('sharpe_ratio')
    @classmethod
    def validate_sharpe_ratio(cls, v):
        if v is not None and (v < -10 or v > 10):
            raise ValueError(f"Sharpe ratio {v} is outside reasonable range [-10, 10]")
        return v
    
    @field_validator('max_drawdown')
    @classmethod
    def validate_max_drawdown(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError(f"Max drawdown {v} must be between 0 and 1")
        return v
    
    @field_validator('win_rate')
    @classmethod
    def validate_win_rate(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError(f"Win rate {v} must be between 0 and 1")
        return v


class QualityGateResult(BaseModel):
    """Schema for quality gate evaluation result of a single variant"""
    variant_id: str = Field(..., description="Variant identifier")
    passed: bool = Field(..., description="Whether variant passed quality gates")
    gate_score: float = Field(..., description="Overall gate score (0.0 to 1.0)")
    failed_criteria: List[str] = Field(default_factory=list, description="List of failed criteria names")
    criteria_scores: Dict[str, float] = Field(..., description="Individual criterion scores")


class ResearchFinding(BaseModel):
    """Schema for a research finding (from research swarm)"""
    finding_type: Literal["technical", "fundamental", "sentiment"] = Field(..., description="Type of finding")
    content: str = Field(..., description="Finding content")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    evidence: List[str] = Field(default_factory=list, description="Supporting evidence")
    source_agent: str = Field(..., description="Agent that generated this finding")


class IterationHistory(BaseModel):
    """Schema for a single iteration in experiment history"""
    iteration: int = Field(..., description="Iteration number")
    timestamp: str = Field(..., description="ISO format timestamp")
    best_sharpe: Optional[float] = Field(None, description="Best Sharpe ratio in this iteration")
    best_variant_id: Optional[str] = Field(None, description="ID of best variant")
    action_taken: str = Field(..., description="Action taken (TUNE, FIX, REFINE, RESEARCH, etc.)")
    parameters_changed: List[str] = Field(default_factory=list, description="Parameters that were changed")
    improvement: Optional[float] = Field(None, description="Improvement from previous iteration")


class FailureAnalysisInput(BaseModel):
    """Input schema for Failure Analysis Agent"""
    
    # Strategy information
    strategy_variants: List[StrategyVariant] = Field(
        ..., 
        description="All strategy variants that were tested"
    )
    
    # Backtest results
    backtest_results: List[BacktestMetrics] = Field(
        ..., 
        description="Backtest metrics for all variants"
    )
    
    # Quality gate evaluation
    quality_gate_results: List[QualityGateResult] = Field(
        ..., 
        description="Quality gate evaluation for all variants"
    )
    
    # Research context
    research_findings: List[ResearchFinding] = Field(
        ..., 
        description="Research findings from research swarm"
    )
    
    # Quality criteria (what the strategy needs to pass)
    quality_criteria: Dict[str, float] = Field(
        ..., 
        description="Quality criteria thresholds (e.g., {'sharpe_ratio': 1.0, 'max_drawdown': 0.20})"
    )
    
    # Iteration context
    current_iteration: int = Field(..., ge=1, description="Current strategy iteration number")
    max_iterations: int = Field(..., ge=1, description="Maximum allowed iterations")
    iteration_history: List[IterationHistory] = Field(
        default_factory=list, 
        description="History of previous iterations"
    )
    
    # Market context
    ticker: str = Field(..., description="Ticker symbol being analyzed")
    timeframe: str = Field(..., description="Timeframe (e.g., '1d', '1h')")
    
    @field_validator('strategy_variants', 'backtest_results', 'quality_gate_results')
    @classmethod
    def validate_non_empty(cls, v):
        """Ensure critical lists are not empty"""
        if not v:
            raise ValueError(f"List cannot be empty")
        return v


class FailureAnalysisOutput(BaseModel):
    """Output schema for Failure Analysis Agent"""
    
    # Classification
    classification: Literal[
        "PARAMETER_ISSUE",
        "ALGORITHM_BUG", 
        "DESIGN_FLAW",
        "RESEARCH_GAP",
        "FUNDAMENTAL_IMPOSSIBILITY"
    ] = Field(..., description="Failure classification category")
    
    # Root cause analysis
    root_cause: str = Field(
        ..., 
        description="Detailed explanation of why strategies failed"
    )
    
    # Statistical assessment
    statistical_assessment: Dict[str, Any] = Field(
        ...,
        description="Statistical analysis of failure patterns",
        example={
            "distance_from_threshold": {"sharpe_ratio": 0.15, "max_drawdown": 0.05},
            "improvement_trajectory": "DECLINING",
            "consistency": "ERRATIC",
            "alignment_with_research": "POOR"
        }
    )
    
    # Recommendation
    recommendation: Literal[
        "TUNE_PARAMETERS",
        "FIX_BUG",
        "REFINE_ALGORITHM",
        "REFINE_RESEARCH",
        "ABANDON"
    ] = Field(..., description="Recommended next action")
    
    # Specific actions
    specific_actions: List[str] = Field(
        ...,
        description="Concrete actions to take",
        example=[
            "Increase position size from 0.1 to 0.15",
            "Add stop-loss at 2% below entry",
            "Filter trades by volume > 1M shares"
        ]
    )
    
    # Confidence and reasoning
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Confidence in this analysis (0.0 to 1.0)"
    )
    
    reasoning: str = Field(
        ...,
        description="Detailed reasoning for the recommendation"
    )
    
    # Likelihood of success
    likelihood_of_success: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Estimated probability that following recommendation will lead to success"
    )
    
    # Iteration recommendation
    should_continue: bool = Field(
        ...,
        description="Whether to continue iterating or abandon"
    )
    
    estimated_iterations_needed: Optional[int] = Field(
        None,
        ge=1,
        description="Estimated number of additional iterations needed"
    )


# ============================================================================
# TRAJECTORY ANALYZER AGENT SCHEMAS
# ============================================================================

class TrajectoryAnalysisInput(BaseModel):
    """Input schema for Trajectory Analyzer Agent"""
    
    # Experiment history (required, must have >= 2 iterations)
    experiment_history: List[IterationHistory] = Field(
        ...,
        min_items=2,
        description="Complete experiment history (minimum 2 iterations required)"
    )
    
    # Current state
    current_iteration: int = Field(..., ge=2, description="Current iteration number")
    current_best_metrics: BacktestMetrics = Field(
        ...,
        description="Metrics of current best variant"
    )
    
    # Quality criteria
    quality_criteria: Dict[str, float] = Field(
        ...,
        description="Quality criteria thresholds"
    )
    
    # Iteration limits
    max_strategy_iterations: int = Field(..., ge=1, description="Max strategy iterations allowed")
    max_research_iterations: int = Field(..., ge=1, description="Max research iterations allowed")
    max_total_iterations: int = Field(..., ge=1, description="Max total iterations allowed")
    
    # Current iteration counts
    strategy_iteration: int = Field(..., ge=0, description="Current strategy iteration count")
    research_iteration: int = Field(..., ge=0, description="Current research iteration count")
    total_iterations: int = Field(..., ge=1, description="Total iterations so far")
    
    @field_validator('experiment_history')
    @classmethod
    def validate_sufficient_history(cls, v):
        """Ensure we have enough history for trajectory analysis"""
        if len(v) < 2:
            raise ValueError("Trajectory analysis requires at least 2 iterations of history")
        return v


class MetricTrajectory(BaseModel):
    """Schema for trajectory of a single metric"""
    metric_name: str = Field(..., description="Name of the metric (e.g., 'sharpe_ratio')")
    values: List[float] = Field(..., description="Historical values")
    trend: Literal["IMPROVING", "DECLINING", "STABLE", "OSCILLATING"] = Field(
        ...,
        description="Overall trend direction"
    )
    improvement_rate: float = Field(
        ...,
        description="Rate of improvement per iteration (can be negative)"
    )
    volatility: float = Field(
        ...,
        ge=0.0,
        description="Volatility of the metric (standard deviation)"
    )
    distance_to_target: float = Field(
        ...,
        description="Distance from current value to target threshold"
    )


class ConvergenceAnalysis(BaseModel):
    """Schema for convergence analysis"""
    status: Literal["CONVERGING", "DIVERGING", "OSCILLATING", "STAGNANT"] = Field(
        ...,
        description="Convergence status"
    )
    
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in convergence assessment"
    )
    
    estimated_iterations_to_convergence: Optional[int] = Field(
        None,
        ge=1,
        description="Estimated iterations until convergence (if converging)"
    )
    
    reasoning: str = Field(
        ...,
        description="Explanation of convergence assessment"
    )


class ParameterImpactAnalysis(BaseModel):
    """Schema for parameter impact analysis"""
    parameter_name: str = Field(..., description="Name of the parameter")
    impact_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Impact score (0.0 = no impact, 1.0 = high impact)"
    )
    direction: Literal["POSITIVE", "NEGATIVE", "NEUTRAL"] = Field(
        ...,
        description="Direction of impact on performance"
    )
    recommendation: str = Field(
        ...,
        description="Recommendation for this parameter"
    )


class TrajectoryAnalysisOutput(BaseModel):
    """Output schema for Trajectory Analyzer Agent"""
    
    # Metric trajectories
    metric_trajectories: List[MetricTrajectory] = Field(
        ...,
        description="Trajectory analysis for each key metric"
    )
    
    # Convergence analysis
    convergence_analysis: ConvergenceAnalysis = Field(
        ...,
        description="Analysis of convergence patterns"
    )
    
    # Parameter impact
    parameter_impacts: List[ParameterImpactAnalysis] = Field(
        default_factory=list,
        description="Impact analysis of parameter changes"
    )
    
    # Overall assessment
    overall_trajectory: Literal["IMPROVING", "DECLINING", "STABLE", "OSCILLATING"] = Field(
        ...,
        description="Overall trajectory across all metrics"
    )
    
    # Recommendation
    recommendation: Literal["CONTINUE", "PIVOT", "ABANDON"] = Field(
        ...,
        description="High-level recommendation based on trajectory"
    )
    
    reasoning: str = Field(
        ...,
        description="Detailed reasoning for the recommendation"
    )
    
    # Confidence
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in trajectory analysis"
    )
    
    # Insights
    key_insights: List[str] = Field(
        ...,
        description="Key insights from trajectory analysis",
        example=[
            "Sharpe ratio improving steadily (+0.1 per iteration)",
            "Max drawdown oscillating, needs stabilization",
            "Position size changes have highest impact on performance"
        ]
    )
    
    # Predictions
    predicted_next_iteration_metrics: Optional[Dict[str, float]] = Field(
        None,
        description="Predicted metrics for next iteration (if trend continues)"
    )


# ============================================================================
# QUALITY GATE NODE OUTPUT SCHEMA
# ============================================================================

class QualityGateNodeOutput(BaseModel):
    """
    Complete output schema for the quality_gate node.
    This is what gets merged back into WorkflowState.
    """
    
    # Best variant selection
    best_variant_id: Optional[str] = Field(
        None,
        description="ID of best performing variant (None if all failed)"
    )
    
    best_metrics: Optional[BacktestMetrics] = Field(
        None,
        description="Metrics of best variant"
    )
    
    # Quality gate results
    any_variant_passed: bool = Field(
        ...,
        description="Whether any variant passed quality gates"
    )
    
    quality_gate_results: List[QualityGateResult] = Field(
        ...,
        description="Quality gate results for all variants"
    )
    
    # Failure analysis (populated if all variants failed)
    failure_analysis: Optional[FailureAnalysisOutput] = Field(
        None,
        description="Failure analysis from Failure Analysis Agent"
    )
    
    # Trajectory analysis (populated if history >= 2)
    trajectory_analysis: Optional[TrajectoryAnalysisOutput] = Field(
        None,
        description="Trajectory analysis from Trajectory Analyzer Agent"
    )
    
    # Routing decision
    next_action: Literal[
        "SUCCESS",
        "TUNE_PARAMETERS",
        "FIX_BUG",
        "REFINE_ALGORITHM",
        "REFINE_RESEARCH",
        "ABANDON"
    ] = Field(
        ...,
        description="Next action for conditional routing"
    )
    
    # Iteration counters (incremented)
    total_iterations: int = Field(..., ge=1, description="Total iterations completed")
    
    # Status
    final_status: Optional[Literal["SUCCESS", "ABANDONED"]] = Field(
        None,
        description="Final status if workflow is ending"
    )
    
    # Reasoning
    decision_reasoning: str = Field(
        ...,
        description="Explanation of routing decision"
    )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_failure_analysis_input_from_state(state: dict) -> FailureAnalysisInput:
    """
    Helper function to extract FailureAnalysisInput from WorkflowState.
    
    Args:
        state: WorkflowState dictionary
        
    Returns:
        FailureAnalysisInput instance
    """
    return FailureAnalysisInput(
        strategy_variants=[
            StrategyVariant(
                variant_id=v["variant_id"],
                code=v["code"],
                parameters=v["parameters"],
                description=v.get("description", "")
            )
            for v in state["strategy_variants"]
        ],
        backtest_results=[
            BacktestMetrics(**result)
            for result in state["backtest_results"]
        ],
        quality_gate_results=[
            QualityGateResult(**result)
            for result in state.get("quality_gate_results", [])
        ],
        research_findings=[
            ResearchFinding(**finding)
            for finding in state["research_findings"]
        ],
        quality_criteria=state["quality_criteria"],
        current_iteration=state["strategy_iteration"],
        max_iterations=state["max_strategy_iterations"],
        iteration_history=[
            IterationHistory(**record)
            for record in state.get("experiment_history", [])
        ],
        ticker=state["ticker"],
        timeframe=state["timeframe"]
    )


def create_trajectory_analysis_input_from_state(state: dict) -> TrajectoryAnalysisInput:
    """
    Helper function to extract TrajectoryAnalysisInput from WorkflowState.
    
    Args:
        state: WorkflowState dictionary
        
    Returns:
        TrajectoryAnalysisInput instance
    """
    # Get best metrics from current iteration
    best_result = max(
        state["backtest_results"],
        key=lambda x: x.get("sharpe_ratio", -999)
    )
    
    return TrajectoryAnalysisInput(
        experiment_history=[
            IterationHistory(**record)
            for record in state["experiment_history"]
        ],
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


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """Example usage of schemas"""
    
    # Example: Create FailureAnalysisInput
    failure_input = FailureAnalysisInput(
        strategy_variants=[
            StrategyVariant(
                variant_id="v1",
                code="class MyStrategy(bt.Strategy): pass",
                parameters={"rsi_period": 14},
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
        research_findings=[
            ResearchFinding(
                finding_type="technical",
                content="Strong momentum in AAPL",
                confidence=0.85,
                evidence=["RSI > 70 for 5 days"],
                source_agent="TechnicalAnalysisAgent"
            )
        ],
        quality_criteria={"sharpe_ratio": 1.0, "max_drawdown": 0.20},
        current_iteration=1,
        max_iterations=5,
        iteration_history=[],
        ticker="AAPL",
        timeframe="1d"
    )
    
    print("✅ FailureAnalysisInput created successfully")
    print(f"   Variants: {len(failure_input.strategy_variants)}")
    print(f"   Results: {len(failure_input.backtest_results)}")
    
    # Example: Create FailureAnalysisOutput
    failure_output = FailureAnalysisOutput(
        classification="PARAMETER_ISSUE",
        root_cause="Sharpe ratio is 0.75, just 0.25 below threshold of 1.0",
        statistical_assessment={
            "distance_from_threshold": {"sharpe_ratio": 0.25},
            "improvement_trajectory": "STABLE",
            "consistency": "CONSISTENT",
            "alignment_with_research": "GOOD"
        },
        recommendation="TUNE_PARAMETERS",
        specific_actions=[
            "Increase position size from 0.1 to 0.15",
            "Add volatility filter to reduce drawdown"
        ],
        confidence=0.85,
        reasoning="Strategy is close to passing, parameter tuning should work",
        likelihood_of_success=0.75,
        should_continue=True,
        estimated_iterations_needed=2
    )
    
    print("✅ FailureAnalysisOutput created successfully")
    print(f"   Classification: {failure_output.classification}")
    print(f"   Recommendation: {failure_output.recommendation}")
