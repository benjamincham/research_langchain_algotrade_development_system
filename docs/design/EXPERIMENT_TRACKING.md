# Experiment Tracking System

## Problem Statement

At the Quality Gate Validation Phase, the system needs to track all iterations and experiments to enable the LLM to:

1. **Analyze improvement trajectories**: Is the strategy getting better or worse?
2. **Detect convergence patterns**: Are we approaching success or diverging?
3. **Identify gradient descent**: Are parameter changes moving in the right direction?
4. **Assess iteration efficiency**: Is each iteration providing value?
5. **Make informed decisions**: Should we continue iterating or change approach?

**Key Question**: How does the LLM know if fine-tuning is going in the right direction?

**Answer**: A comprehensive experiment tracking system that records every iteration with full context and enables trajectory analysis.

---

## Solution: Multi-Layer Experiment Tracking

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    EXPERIMENT TRACKING SYSTEM                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    LAYER 1: EXPERIMENT LOGGER                    │   │
│  │  Records every iteration with:                                   │   │
│  │  • Strategy code and parameters                                  │   │
│  │  • Backtest metrics                                              │   │
│  │  • Quality gate results                                          │   │
│  │  • Actions taken                                                 │   │
│  │  • Timestamp and metadata                                        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    LAYER 2: TRAJECTORY ANALYZER                  │   │
│  │  Computes improvement metrics:                                   │   │
│  │  • Metric deltas (iteration N vs N-1)                            │   │
│  │  • Improvement rates                                             │   │
│  │  • Convergence indicators                                        │   │
│  │  • Gradient direction                                            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    LAYER 3: LLM ANALYZER                         │   │
│  │  LLM analyzes trajectory and determines:                         │   │
│  │  • Is improvement consistent?                                    │   │
│  │  • Are we converging to success?                                 │   │
│  │  • Which parameters are helping/hurting?                         │   │
│  │  • Should we continue or pivot?                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    LAYER 4: VISUALIZATION                        │   │
│  │  Generates charts and reports:                                   │   │
│  │  • Metric trajectories over iterations                           │   │
│  │  • Parameter sensitivity analysis                                │   │
│  │  • Convergence plots                                             │   │
│  │  • Experiment comparison tables                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Schema

### Experiment Record

```python
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime

class ExperimentRecord(BaseModel):
    """Complete record of a single experiment iteration."""
    
    # Identification
    experiment_id: str = Field(description="Unique experiment ID (e.g., 'exp_001')")
    iteration: int = Field(description="Iteration number within this experiment")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Context
    research_directive: str = Field(description="Original research objective")
    research_iteration: int = Field(description="Research iteration number")
    strategy_iteration: int = Field(description="Strategy iteration number")
    
    # Strategy
    strategy_name: str
    strategy_code: str
    strategy_description: str
    parameters: Dict[str, Any] = Field(description="Strategy parameters")
    
    # Backtest Results
    metrics: Dict[str, float] = Field(description="All backtest metrics")
    
    # Quality Gate
    gate_passed: bool
    gate_score: float = Field(description="Overall quality gate score (0-1)")
    failed_criteria: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Actions Taken
    action_taken: str = Field(description="What action was taken after this iteration")
    parameter_changes: Dict[str, Any] = Field(
        default_factory=dict,
        description="What parameters were changed"
    )
    
    # Analysis
    failure_classification: Optional[str] = None
    failure_analysis: Optional[Dict[str, Any]] = None
    
    # Metadata
    compute_time: float = Field(description="Time taken for this iteration (seconds)")
    cost: float = Field(default=0.0, description="LLM API cost for this iteration")


class ExperimentTrajectory(BaseModel):
    """Analysis of experiment trajectory across iterations."""
    
    experiment_id: str
    total_iterations: int
    
    # Metric Trajectories
    metric_history: Dict[str, List[float]] = Field(
        description="History of each metric across iterations"
    )
    
    # Improvement Analysis
    improvement_rates: Dict[str, float] = Field(
        description="Rate of improvement for each metric (per iteration)"
    )
    
    convergence_status: str = Field(
        description="CONVERGING | DIVERGING | OSCILLATING | STAGNANT"
    )
    
    # Gradient Analysis
    gradient_direction: str = Field(
        description="ASCENDING (improving) | DESCENDING (worsening) | FLAT"
    )
    
    distance_to_goal: float = Field(
        description="How far from passing quality gates (0 = passing)"
    )
    
    estimated_iterations_to_success: Optional[int] = Field(
        description="Estimated iterations needed to pass (if converging)"
    )
    
    # Parameter Sensitivity
    parameter_impact: Dict[str, float] = Field(
        description="Impact of each parameter change on performance"
    )


class ExperimentSummary(BaseModel):
    """High-level summary of an experiment."""
    
    experiment_id: str
    status: str = Field(description="SUCCESS | FAILED | IN_PROGRESS | ABANDONED")
    
    total_iterations: int
    best_iteration: int
    best_metrics: Dict[str, float]
    
    final_recommendation: str
    lessons_learned: List[str]
    
    total_compute_time: float
    total_cost: float
```

---

## Layer 1: Experiment Logger

### Implementation

```python
import json
from pathlib import Path
from typing import Optional
from src.core.logging import get_logger

class ExperimentLogger:
    """
    Logs all experiment iterations to enable trajectory analysis.
    """
    
    def __init__(self, experiment_id: str, storage_dir: str = "experiments"):
        self.experiment_id = experiment_id
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_file = self.storage_dir / f"{experiment_id}.jsonl"
        self.logger = get_logger(__name__)
        
        self.current_iteration = 0
        self.records: List[ExperimentRecord] = []
    
    def log_iteration(
        self,
        strategy_name: str,
        strategy_code: str,
        strategy_description: str,
        parameters: Dict[str, Any],
        metrics: Dict[str, float],
        gate_passed: bool,
        gate_score: float,
        failed_criteria: List[Dict],
        action_taken: str,
        parameter_changes: Dict[str, Any],
        failure_classification: Optional[str] = None,
        failure_analysis: Optional[Dict] = None,
        compute_time: float = 0.0,
        cost: float = 0.0,
        research_directive: str = "",
        research_iteration: int = 1,
        strategy_iteration: int = 1
    ) -> ExperimentRecord:
        """
        Log a single experiment iteration.
        """
        self.current_iteration += 1
        
        record = ExperimentRecord(
            experiment_id=self.experiment_id,
            iteration=self.current_iteration,
            research_directive=research_directive,
            research_iteration=research_iteration,
            strategy_iteration=strategy_iteration,
            strategy_name=strategy_name,
            strategy_code=strategy_code,
            strategy_description=strategy_description,
            parameters=parameters,
            metrics=metrics,
            gate_passed=gate_passed,
            gate_score=gate_score,
            failed_criteria=failed_criteria,
            action_taken=action_taken,
            parameter_changes=parameter_changes,
            failure_classification=failure_classification,
            failure_analysis=failure_analysis,
            compute_time=compute_time,
            cost=cost
        )
        
        # Store in memory
        self.records.append(record)
        
        # Append to file (JSONL format for streaming)
        with open(self.experiment_file, 'a') as f:
            f.write(record.model_dump_json() + '\n')
        
        self.logger.info(
            f"Logged iteration {self.current_iteration}: "
            f"gate_score={gate_score:.2f}, action={action_taken}"
        )
        
        return record
    
    def get_all_records(self) -> List[ExperimentRecord]:
        """Get all records for this experiment."""
        return self.records
    
    def get_latest_record(self) -> Optional[ExperimentRecord]:
        """Get the most recent record."""
        return self.records[-1] if self.records else None
    
    def load_from_file(self) -> List[ExperimentRecord]:
        """Load experiment records from file."""
        if not self.experiment_file.exists():
            return []
        
        records = []
        with open(self.experiment_file, 'r') as f:
            for line in f:
                record_dict = json.loads(line)
                records.append(ExperimentRecord(**record_dict))
        
        self.records = records
        self.current_iteration = len(records)
        return records
```

---

## Layer 2: Trajectory Analyzer

### Implementation

```python
import numpy as np
from typing import List, Dict, Tuple
from scipy import stats

class TrajectoryAnalyzer:
    """
    Analyzes experiment trajectories to detect convergence and improvement patterns.
    """
    
    def __init__(self, records: List[ExperimentRecord]):
        self.records = records
        self.logger = get_logger(__name__)
    
    def analyze_trajectory(self) -> ExperimentTrajectory:
        """
        Perform comprehensive trajectory analysis.
        """
        if len(self.records) < 2:
            raise ValueError("Need at least 2 iterations for trajectory analysis")
        
        # Extract metric histories
        metric_history = self._extract_metric_history()
        
        # Compute improvement rates
        improvement_rates = self._compute_improvement_rates(metric_history)
        
        # Detect convergence status
        convergence_status = self._detect_convergence(metric_history)
        
        # Analyze gradient direction
        gradient_direction = self._analyze_gradient(metric_history)
        
        # Compute distance to goal
        distance_to_goal = self._compute_distance_to_goal()
        
        # Estimate iterations to success
        estimated_iterations = self._estimate_iterations_to_success(
            improvement_rates, distance_to_goal
        )
        
        # Analyze parameter impact
        parameter_impact = self._analyze_parameter_impact()
        
        return ExperimentTrajectory(
            experiment_id=self.records[0].experiment_id,
            total_iterations=len(self.records),
            metric_history=metric_history,
            improvement_rates=improvement_rates,
            convergence_status=convergence_status,
            gradient_direction=gradient_direction,
            distance_to_goal=distance_to_goal,
            estimated_iterations_to_success=estimated_iterations,
            parameter_impact=parameter_impact
        )
    
    def _extract_metric_history(self) -> Dict[str, List[float]]:
        """Extract time series of each metric."""
        metric_history = {}
        
        for record in self.records:
            for metric_name, metric_value in record.metrics.items():
                if metric_name not in metric_history:
                    metric_history[metric_name] = []
                metric_history[metric_name].append(metric_value)
        
        return metric_history
    
    def _compute_improvement_rates(
        self, metric_history: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """
        Compute rate of improvement for each metric.
        
        Uses linear regression to find slope of metric over iterations.
        Positive slope = improving, negative = worsening.
        """
        improvement_rates = {}
        
        for metric_name, values in metric_history.items():
            if len(values) < 2:
                improvement_rates[metric_name] = 0.0
                continue
            
            # Linear regression: metric = slope * iteration + intercept
            iterations = np.arange(len(values))
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                iterations, values
            )
            
            # For metrics where higher is better (Sharpe, returns)
            # positive slope = improvement
            # For metrics where lower is better (drawdown, volatility)
            # negative slope = improvement
            
            # We'll store raw slope and let LLM interpret based on metric type
            improvement_rates[metric_name] = slope
        
        return improvement_rates
    
    def _detect_convergence(
        self, metric_history: Dict[str, List[float]]
    ) -> str:
        """
        Detect if metrics are converging, diverging, oscillating, or stagnant.
        """
        # Focus on key metric: gate_score
        gate_scores = [r.gate_score for r in self.records]
        
        if len(gate_scores) < 3:
            return "INSUFFICIENT_DATA"
        
        # Compute variance of recent iterations vs early iterations
        mid_point = len(gate_scores) // 2
        early_variance = np.var(gate_scores[:mid_point])
        recent_variance = np.var(gate_scores[mid_point:])
        
        # Compute trend
        iterations = np.arange(len(gate_scores))
        slope, _, r_value, _, _ = stats.linregress(iterations, gate_scores)
        
        # Decision logic
        if recent_variance < early_variance * 0.5 and slope > 0:
            return "CONVERGING"  # Variance decreasing, trend upward
        elif recent_variance > early_variance * 2:
            return "OSCILLATING"  # High variance
        elif abs(slope) < 0.01:
            return "STAGNANT"  # No trend
        else:
            return "DIVERGING"  # Getting worse
    
    def _analyze_gradient(
        self, metric_history: Dict[str, List[float]]
    ) -> str:
        """
        Analyze if we're in gradient ascent (improving) or descent (worsening).
        """
        gate_scores = [r.gate_score for r in self.records]
        
        if len(gate_scores) < 2:
            return "FLAT"
        
        # Compute average change
        changes = np.diff(gate_scores)
        avg_change = np.mean(changes)
        
        if avg_change > 0.02:  # Improving by >2% per iteration
            return "ASCENDING"
        elif avg_change < -0.02:  # Worsening by >2% per iteration
            return "DESCENDING"
        else:
            return "FLAT"
    
    def _compute_distance_to_goal(self) -> float:
        """
        Compute how far we are from passing quality gates.
        
        Returns 0.0 if passing, positive value if failing.
        """
        latest_record = self.records[-1]
        
        if latest_record.gate_passed:
            return 0.0
        
        # Distance = 1.0 - gate_score (assuming gate_score is 0-1)
        return 1.0 - latest_record.gate_score
    
    def _estimate_iterations_to_success(
        self,
        improvement_rates: Dict[str, float],
        distance_to_goal: float
    ) -> Optional[int]:
        """
        Estimate how many more iterations needed to pass quality gates.
        
        Uses linear extrapolation based on improvement rate.
        """
        if distance_to_goal == 0.0:
            return 0
        
        # Use gate_score improvement rate
        gate_scores = [r.gate_score for r in self.records]
        
        if len(gate_scores) < 2:
            return None
        
        # Compute average improvement per iteration
        improvements = np.diff(gate_scores)
        avg_improvement = np.mean(improvements)
        
        if avg_improvement <= 0:
            return None  # Not improving, can't estimate
        
        # Estimate iterations needed
        iterations_needed = distance_to_goal / avg_improvement
        
        return int(np.ceil(iterations_needed))
    
    def _analyze_parameter_impact(self) -> Dict[str, float]:
        """
        Analyze which parameter changes had the most impact on performance.
        
        Correlates parameter changes with metric improvements.
        """
        if len(self.records) < 3:
            return {}
        
        parameter_impact = {}
        
        # For each parameter that was changed
        for i in range(1, len(self.records)):
            prev_record = self.records[i - 1]
            curr_record = self.records[i]
            
            # Get parameter changes
            param_changes = curr_record.parameter_changes
            
            # Get metric improvement
            metric_improvement = (
                curr_record.gate_score - prev_record.gate_score
            )
            
            # Attribute improvement to each changed parameter
            for param_name, param_change in param_changes.items():
                if param_name not in parameter_impact:
                    parameter_impact[param_name] = []
                
                parameter_impact[param_name].append(metric_improvement)
        
        # Average impact for each parameter
        return {
            param: np.mean(impacts)
            for param, impacts in parameter_impact.items()
        }
```

---

## Layer 3: LLM Trajectory Analyzer

### Prompt Template

```python
TRAJECTORY_ANALYSIS_PROMPT = """
You are an expert quantitative analyst reviewing experiment iterations to determine if the optimization is converging toward success.

## Experiment Context

**Experiment ID:** {experiment_id}
**Total Iterations:** {total_iterations}
**Current Status:** {current_status}

## Metric Trajectories

{metric_trajectories}

## Statistical Analysis

**Improvement Rates (per iteration):**
{improvement_rates}

**Convergence Status:** {convergence_status}
**Gradient Direction:** {gradient_direction}
**Distance to Goal:** {distance_to_goal:.2f}
**Estimated Iterations to Success:** {estimated_iterations}

## Parameter Impact Analysis

{parameter_impact}

## Iteration History

{iteration_history}

## Your Task

Analyze the experiment trajectory and answer:

1. **Is the optimization converging?** Are we getting closer to passing quality gates?

2. **Is the gradient descent working?** Are parameter changes moving in the right direction?

3. **Which parameters are helping/hurting?** What should we keep changing vs. stop changing?

4. **Should we continue iterating?** Or should we pivot to a different approach?

5. **What is the likelihood of success?** If we continue, will we pass quality gates?

6. **Specific recommendations:** What should the next iteration do?

## Output Format

Provide your analysis in JSON format:

```json
{
  "convergence_assessment": "CONVERGING | DIVERGING | OSCILLATING | STAGNANT",
  "gradient_effectiveness": "EFFECTIVE | INEFFECTIVE | MIXED",
  "likelihood_of_success": "HIGH | MEDIUM | LOW",
  "estimated_iterations_needed": 3,
  "key_findings": [
    "Finding 1",
    "Finding 2"
  ],
  "parameter_recommendations": {
    "position_size": "INCREASE - positively correlated with performance",
    "stop_loss": "KEEP - stable and effective",
    "entry_threshold": "DECREASE - negatively correlated"
  },
  "next_action": "CONTINUE | PIVOT | ABANDON",
  "specific_recommendations": [
    "Increase position size from 0.15 to 0.20",
    "Keep stop loss at current level",
    "Decrease entry threshold from 75 to 70"
  ],
  "confidence": 0.85,
  "reasoning": "Detailed explanation of your analysis"
}
```

## Guidelines

- Look for consistent improvement trends
- Identify which parameter changes are working
- Detect if we're stuck in local optima
- Be honest about likelihood of success
- Provide actionable recommendations
"""
```

### Implementation

```python
class LLMTrajectoryAnalyzer:
    """
    Uses LLM to analyze experiment trajectories and provide recommendations.
    """
    
    def __init__(self):
        self.llm = create_powerful_llm()
        self.logger = get_logger(__name__)
    
    async def analyze_trajectory(
        self,
        records: List[ExperimentRecord],
        trajectory: ExperimentTrajectory
    ) -> Dict[str, Any]:
        """
        Use LLM to analyze trajectory and provide recommendations.
        """
        # Format data for prompt
        metric_trajectories = self._format_metric_trajectories(trajectory)
        improvement_rates = self._format_improvement_rates(trajectory)
        parameter_impact = self._format_parameter_impact(trajectory)
        iteration_history = self._format_iteration_history(records)
        
        # Build prompt
        prompt = TRAJECTORY_ANALYSIS_PROMPT.format(
            experiment_id=trajectory.experiment_id,
            total_iterations=trajectory.total_iterations,
            current_status="IN_PROGRESS",
            metric_trajectories=metric_trajectories,
            improvement_rates=improvement_rates,
            convergence_status=trajectory.convergence_status,
            gradient_direction=trajectory.gradient_direction,
            distance_to_goal=trajectory.distance_to_goal,
            estimated_iterations=trajectory.estimated_iterations_to_success or "Unknown",
            parameter_impact=parameter_impact,
            iteration_history=iteration_history
        )
        
        # Call LLM
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        
        # Parse response
        analysis = self._parse_json_response(response.content)
        
        self.logger.info(
            f"Trajectory analysis: {analysis['convergence_assessment']}, "
            f"next action: {analysis['next_action']}"
        )
        
        return analysis
    
    def _format_metric_trajectories(self, trajectory: ExperimentTrajectory) -> str:
        """Format metric trajectories as a table."""
        lines = []
        
        for metric_name, values in trajectory.metric_history.items():
            # Show last 5 iterations
            recent_values = values[-5:]
            values_str = " → ".join([f"{v:.2f}" for v in recent_values])
            lines.append(f"- {metric_name}: {values_str}")
        
        return "\n".join(lines)
    
    def _format_improvement_rates(self, trajectory: ExperimentTrajectory) -> str:
        """Format improvement rates."""
        lines = []
        
        for metric_name, rate in trajectory.improvement_rates.items():
            direction = "↑" if rate > 0 else "↓" if rate < 0 else "→"
            lines.append(f"- {metric_name}: {rate:+.4f} {direction}")
        
        return "\n".join(lines)
    
    def _format_parameter_impact(self, trajectory: ExperimentTrajectory) -> str:
        """Format parameter impact analysis."""
        if not trajectory.parameter_impact:
            return "No parameter changes yet"
        
        lines = []
        
        for param_name, impact in trajectory.parameter_impact.items():
            effect = "POSITIVE" if impact > 0 else "NEGATIVE" if impact < 0 else "NEUTRAL"
            lines.append(f"- {param_name}: {impact:+.4f} ({effect})")
        
        return "\n".join(lines)
    
    def _format_iteration_history(self, records: List[ExperimentRecord]) -> str:
        """Format iteration history."""
        lines = []
        
        for record in records[-5:]:  # Last 5 iterations
            lines.append(f"\nIteration {record.iteration}:")
            lines.append(f"  Gate Score: {record.gate_score:.2f}")
            lines.append(f"  Action: {record.action_taken}")
            if record.parameter_changes:
                changes_str = ", ".join([
                    f"{k}={v}" for k, v in record.parameter_changes.items()
                ])
                lines.append(f"  Changes: {changes_str}")
        
        return "\n".join(lines)
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response."""
        import json
        import re
        
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                raise ValueError("Could not find JSON in LLM response")
        
        return json.loads(json_str)
```

---

## Integration with Quality Gate System

### Updated Workflow

```python
async def quality_gate_with_experiment_tracking(
    strategy: TradingStrategy,
    research_findings: List[dict],
    gate: QualityGate,
    max_iterations: int = 5
) -> tuple[bool, TradingStrategy, ExperimentSummary]:
    """
    Quality gate loop with comprehensive experiment tracking.
    """
    # Initialize tracking
    experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = ExperimentLogger(experiment_id)
    failure_agent = FailureAnalysisAgent()
    trajectory_analyzer = TrajectoryAnalyzer([])
    llm_analyzer = LLMTrajectoryAnalyzer()
    
    current_strategy = strategy
    
    for iteration in range(max_iterations):
        start_time = time.time()
        
        # Run backtest
        metrics = await run_backtest(current_strategy)
        
        # Evaluate quality gate
        result = await evaluate_quality_gate(metrics, gate)
        
        compute_time = time.time() - start_time
        
        if result.passed:
            # SUCCESS! Log final iteration
            logger.log_iteration(
                strategy_name=current_strategy.name,
                strategy_code=current_strategy.code,
                strategy_description=current_strategy.description,
                parameters=current_strategy.parameters,
                metrics=metrics,
                gate_passed=True,
                gate_score=result.overall_score,
                failed_criteria=[],
                action_taken="SUCCESS",
                parameter_changes={},
                compute_time=compute_time
            )
            
            # Generate summary
            summary = _generate_experiment_summary(logger, "SUCCESS")
            return True, current_strategy, summary
        
        # FAILED - Perform failure analysis
        analysis = await failure_agent.analyze_failure(
            strategy_code=current_strategy.code,
            research_findings=research_findings,
            backtest_metrics=metrics,
            quality_gate_results=result.to_dict(),
            iteration_history=[r.dict() for r in logger.get_all_records()],
            current_iteration=iteration + 1,
            max_iterations=max_iterations
        )
        
        # Log iteration
        logger.log_iteration(
            strategy_name=current_strategy.name,
            strategy_code=current_strategy.code,
            strategy_description=current_strategy.description,
            parameters=current_strategy.parameters,
            metrics=metrics,
            gate_passed=False,
            gate_score=result.overall_score,
            failed_criteria=result.failed_criteria,
            action_taken=analysis.recommendation,
            parameter_changes={},  # Will be filled after refinement
            failure_classification=analysis.failure_classification,
            failure_analysis=analysis.dict(),
            compute_time=compute_time
        )
        
        # Analyze trajectory (if we have enough iterations)
        if iteration >= 1:
            trajectory = trajectory_analyzer.analyze_trajectory()
            llm_trajectory_analysis = await llm_analyzer.analyze_trajectory(
                logger.get_all_records(),
                trajectory
            )
            
            # Use LLM trajectory analysis to inform decision
            if llm_trajectory_analysis['next_action'] == 'ABANDON':
                summary = _generate_experiment_summary(logger, "ABANDONED")
                return False, current_strategy, summary
            
            elif llm_trajectory_analysis['next_action'] == 'PIVOT':
                # Go back to research
                summary = _generate_experiment_summary(logger, "PIVOT_TO_RESEARCH")
                return False, current_strategy, summary
        
        # Refine strategy based on analysis
        if analysis.recommendation == "TUNE_PARAMETERS":
            current_strategy, param_changes = await refine_strategy_parameters(
                current_strategy,
                analysis.specific_actions
            )
            # Update last record with parameter changes
            logger.records[-1].parameter_changes = param_changes
        
        elif analysis.recommendation == "FIX_BUG":
            current_strategy = await fix_strategy_bug(
                current_strategy,
                analysis.bug_detection
            )
        
        elif analysis.recommendation in ["REFINE_RESEARCH", "ABANDON"]:
            summary = _generate_experiment_summary(logger, "FAILED")
            return False, current_strategy, summary
    
    # Max iterations reached
    summary = _generate_experiment_summary(logger, "MAX_ITERATIONS")
    return False, current_strategy, summary


def _generate_experiment_summary(
    logger: ExperimentLogger,
    status: str
) -> ExperimentSummary:
    """Generate experiment summary."""
    records = logger.get_all_records()
    
    # Find best iteration
    best_iteration = max(records, key=lambda r: r.gate_score)
    
    # Extract lessons learned
    lessons = []
    for record in records:
        if record.failure_analysis:
            lessons.append(record.failure_analysis.get('root_cause', ''))
    
    return ExperimentSummary(
        experiment_id=logger.experiment_id,
        status=status,
        total_iterations=len(records),
        best_iteration=best_iteration.iteration,
        best_metrics=best_iteration.metrics,
        final_recommendation=records[-1].action_taken,
        lessons_learned=lessons,
        total_compute_time=sum(r.compute_time for r in records),
        total_cost=sum(r.cost for r in records)
    )
```

---

## Example Output

### Experiment Log (JSONL file)

```jsonl
{"experiment_id":"exp_20260118_143022","iteration":1,"timestamp":"2026-01-18T14:30:22Z","strategy_name":"AAPL_Momentum_v1","parameters":{"position_size":0.1,"rsi_threshold":70,"stop_loss":0.15},"metrics":{"sharpe_ratio":0.75,"max_drawdown":0.22,"win_rate":0.48},"gate_passed":false,"gate_score":0.65,"action_taken":"TUNE_PARAMETERS","parameter_changes":{},"compute_time":45.2}
{"experiment_id":"exp_20260118_143022","iteration":2,"timestamp":"2026-01-18T14:35:10Z","strategy_name":"AAPL_Momentum_v2","parameters":{"position_size":0.15,"rsi_threshold":70,"stop_loss":0.15},"metrics":{"sharpe_ratio":0.85,"max_drawdown":0.20,"win_rate":0.50},"gate_passed":false,"gate_score":0.75,"action_taken":"TUNE_PARAMETERS","parameter_changes":{"position_size":0.15},"compute_time":43.8}
{"experiment_id":"exp_20260118_143022","iteration":3,"timestamp":"2026-01-18T14:39:55Z","strategy_name":"AAPL_Momentum_v3","parameters":{"position_size":0.20,"rsi_threshold":75,"stop_loss":0.12},"metrics":{"sharpe_ratio":1.05,"max_drawdown":0.18,"win_rate":0.52},"gate_passed":true,"gate_score":0.92,"action_taken":"SUCCESS","parameter_changes":{"position_size":0.20,"rsi_threshold":75,"stop_loss":0.12},"compute_time":44.1}
```

### LLM Trajectory Analysis

```json
{
  "convergence_assessment": "CONVERGING",
  "gradient_effectiveness": "EFFECTIVE",
  "likelihood_of_success": "HIGH",
  "estimated_iterations_needed": 1,
  "key_findings": [
    "Gate score improving consistently: 0.65 → 0.75 → 0.92",
    "Sharpe ratio showing strong upward trend: 0.75 → 0.85 → 1.05",
    "Max drawdown decreasing: 22% → 20% → 18%",
    "Position size increases positively correlated with performance"
  ],
  "parameter_recommendations": {
    "position_size": "INCREASE - strong positive correlation (+0.10 per 0.05 increase)",
    "rsi_threshold": "INCREASE - helped improve win rate",
    "stop_loss": "TIGHTEN - successfully reduced drawdown"
  },
  "next_action": "CONTINUE",
  "specific_recommendations": [
    "Continue with current parameter trajectory",
    "Consider slight increase in position size to 0.22 if needed",
    "Strategy is very close to passing, one more iteration should succeed"
  ],
  "confidence": 0.90,
  "reasoning": "The experiment shows clear convergence with consistent improvement across all key metrics. The gradient descent is working effectively - each parameter change has moved metrics closer to thresholds. The improvement rate suggests we should pass quality gates within 1-2 more iterations. High confidence in success."
}
```

---

## Summary

### How the System Tracks Experiments

1. **ExperimentLogger**: Records every iteration to JSONL file
2. **TrajectoryAnalyzer**: Computes statistical metrics (improvement rates, convergence)
3. **LLMTrajectoryAnalyzer**: Uses LLM to interpret trajectories and recommend actions
4. **Visualization**: Generates charts showing metric evolution

### What the LLM Can Analyze

- ✅ **Improvement trajectory**: Is each iteration getting better?
- ✅ **Convergence detection**: Are we approaching success?
- ✅ **Gradient effectiveness**: Are parameter changes helping?
- ✅ **Parameter sensitivity**: Which parameters matter most?
- ✅ **Iteration efficiency**: Is each iteration providing value?
- ✅ **Success likelihood**: Will we pass if we continue?

### Key Advantages

- ✅ **Complete history**: Every iteration recorded with full context
- ✅ **Statistical analysis**: Quantitative metrics on improvement
- ✅ **LLM reasoning**: Intelligent interpretation of patterns
- ✅ **Actionable insights**: Specific recommendations for next steps
- ✅ **Transparent**: Can visualize and review all experiments

---

**End of Experiment Tracking System Design**
