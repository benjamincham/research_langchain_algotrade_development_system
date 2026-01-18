# Dynamic Quality Gate System Design

## Overview

The Quality Gate System validates trading strategies against user-defined and dynamically evolving criteria. Unlike static boolean pass/fail gates, this system uses fuzzy logic scoring with confidence intervals and adaptive thresholds.

## Key Features

1. **User-Defined Criteria**: Initial criteria set by human at project start
2. **Fuzzy Logic Scoring**: Continuous 0-1 scores instead of boolean pass/fail
3. **Statistical Significance**: Confidence intervals for all metrics
4. **Adaptive Thresholds**: Criteria can evolve based on market conditions
5. **Detailed Feedback**: Actionable improvement suggestions for failures

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        QUALITY GATE SYSTEM                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    CRITERIA REGISTRY                             │   │
│  │  • User-defined criteria from initialization                     │   │
│  │  • Built-in standard criteria                                    │   │
│  │  • Custom metric criteria                                        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    FUZZY LOGIC EVALUATOR                         │   │
│  │  • Continuous scoring (0-1)                                      │   │
│  │  • Soft thresholds with gradual penalties                       │   │
│  │  • Weighted aggregation                                          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                 STATISTICAL VALIDATOR                            │   │
│  │  • Confidence intervals                                          │   │
│  │  • Statistical significance tests                                │   │
│  │  • Monte Carlo validation                                        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                  FEEDBACK GENERATOR                              │   │
│  │  • Identify weakest criteria                                     │   │
│  │  • Generate improvement suggestions                              │   │
│  │  • Prioritize fixes by impact                                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Criterion Schema

```python
from pydantic import BaseModel
from typing import Literal, Optional, Callable

class Criterion(BaseModel):
    """A single quality gate criterion."""
    
    # Identification
    name: str
    description: str
    category: Literal["performance", "risk", "statistical", "custom"]
    
    # Evaluation
    metric: str  # Name of metric to evaluate
    operator: Literal[">=", "<=", "==", "!=", ">", "<", "between"]
    threshold: float
    threshold_upper: Optional[float] = None  # For "between" operator
    
    # Fuzzy logic parameters
    soft_threshold: Optional[float] = None  # Gradual penalty starts here
    penalty_curve: Literal["linear", "exponential", "sigmoid"] = "linear"
    
    # Weighting
    weight: float = 1.0
    is_required: bool = True  # Required criteria must pass
    
    # Statistical
    require_significance: bool = False
    significance_level: float = 0.05
    min_samples: int = 30


class QualityGate(BaseModel):
    """A collection of criteria forming a quality gate."""
    
    name: str
    description: str
    criteria: list[Criterion]
    
    # Gate-level settings
    min_overall_score: float = 0.7  # Minimum weighted average to pass
    max_required_failures: int = 0  # Max required criteria that can fail
```

## Standard Criteria Library

### Performance Criteria

| Criterion | Metric | Threshold | Description |
|-----------|--------|-----------|-------------|
| Sharpe Ratio | sharpe_ratio | >= 1.0 | Risk-adjusted return |
| Profit Factor | profit_factor | >= 1.5 | Gross profit / Gross loss |
| Win Rate | win_rate | >= 0.5 | Winning trades / Total trades |
| Annual Return | annual_return | >= target | User-defined target |
| Recovery Factor | recovery_factor | >= 2.0 | Net profit / Max drawdown |

### Risk Criteria

| Criterion | Metric | Threshold | Description |
|-----------|--------|-----------|-------------|
| Max Drawdown | max_drawdown | <= 0.20 | Maximum peak-to-trough decline |
| VaR (95%) | var_95 | <= 0.05 | Value at Risk |
| Expected Shortfall | expected_shortfall | <= 0.07 | Average loss beyond VaR |
| Correlation | benchmark_correlation | <= 0.7 | Correlation with benchmark |
| Volatility | annual_volatility | <= target | User-defined target |

### Statistical Criteria

| Criterion | Metric | Threshold | Description |
|-----------|--------|-----------|-------------|
| Trade Count | trade_count | >= 30 | Minimum trades for significance |
| T-Statistic | t_statistic | >= 2.0 | Statistical significance |
| P-Value | p_value | <= 0.05 | Probability of random result |
| OOS Performance | oos_sharpe | >= 0.8 | Out-of-sample Sharpe |

## Fuzzy Logic Evaluation

### Scoring Function

```python
def fuzzy_score(
    value: float,
    threshold: float,
    soft_threshold: Optional[float],
    operator: str,
    penalty_curve: str = "linear"
) -> float:
    """
    Calculate fuzzy score for a metric value.
    
    Returns:
        Score between 0.0 (complete failure) and 1.0 (full pass)
    """
    if operator in [">=", ">"]:
        if value >= threshold:
            return 1.0
        elif soft_threshold and value >= soft_threshold:
            # Gradual penalty zone
            range_size = threshold - soft_threshold
            distance = threshold - value
            penalty = apply_curve(distance / range_size, penalty_curve)
            return 1.0 - (penalty * 0.5)  # Max 50% penalty in soft zone
        else:
            # Hard failure zone
            if soft_threshold:
                base = soft_threshold
            else:
                base = threshold
            if value <= 0:
                return 0.0
            return max(0.0, value / base * 0.5)
    
    elif operator in ["<=", "<"]:
        if value <= threshold:
            return 1.0
        elif soft_threshold and value <= soft_threshold:
            range_size = soft_threshold - threshold
            distance = value - threshold
            penalty = apply_curve(distance / range_size, penalty_curve)
            return 1.0 - (penalty * 0.5)
        else:
            # Hard failure
            return max(0.0, 1.0 - (value - threshold) / threshold)
    
    # ... handle other operators


def apply_curve(x: float, curve: str) -> float:
    """Apply penalty curve transformation."""
    if curve == "linear":
        return x
    elif curve == "exponential":
        return x ** 2
    elif curve == "sigmoid":
        return 1 / (1 + math.exp(-10 * (x - 0.5)))
    return x
```

### Weighted Aggregation

```python
def aggregate_scores(
    criterion_results: list[CriterionResult],
    gate: QualityGate
) -> GateResult:
    """
    Aggregate criterion scores into overall gate result.
    """
    # Check required criteria
    required_failures = sum(
        1 for r in criterion_results 
        if r.criterion.is_required and r.score < 0.5
    )
    
    if required_failures > gate.max_required_failures:
        return GateResult(
            passed=False,
            reason="Required criteria failed",
            failed_criteria=[r for r in criterion_results if r.score < 0.5]
        )
    
    # Calculate weighted average
    total_weight = sum(r.criterion.weight for r in criterion_results)
    weighted_sum = sum(
        r.score * r.criterion.weight 
        for r in criterion_results
    )
    overall_score = weighted_sum / total_weight
    
    return GateResult(
        passed=overall_score >= gate.min_overall_score,
        overall_score=overall_score,
        criterion_results=criterion_results,
        feedback=generate_feedback(criterion_results, overall_score)
    )
```

## Statistical Validation

### Confidence Intervals

```python
def calculate_confidence_interval(
    metric_values: list[float],
    confidence_level: float = 0.95
) -> tuple[float, float, float]:
    """
    Calculate confidence interval for a metric.
    
    Returns:
        (mean, lower_bound, upper_bound)
    """
    n = len(metric_values)
    mean = np.mean(metric_values)
    std_error = np.std(metric_values, ddof=1) / np.sqrt(n)
    
    # t-distribution for small samples
    t_value = stats.t.ppf((1 + confidence_level) / 2, n - 1)
    margin = t_value * std_error
    
    return mean, mean - margin, mean + margin
```

### Statistical Significance Test

```python
def test_significance(
    strategy_returns: list[float],
    benchmark_returns: list[float],
    significance_level: float = 0.05
) -> SignificanceResult:
    """
    Test if strategy returns are significantly different from benchmark.
    """
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(strategy_returns, benchmark_returns)
    
    # Effect size (Cohen's d)
    diff = np.array(strategy_returns) - np.array(benchmark_returns)
    cohens_d = np.mean(diff) / np.std(diff, ddof=1)
    
    return SignificanceResult(
        is_significant=p_value < significance_level,
        t_statistic=t_stat,
        p_value=p_value,
        effect_size=cohens_d,
        interpretation=interpret_effect_size(cohens_d)
    )
```

## Feedback Generation

### Feedback Structure

```python
class GateFeedback(BaseModel):
    """Feedback for quality gate evaluation."""
    
    overall_assessment: str
    passed: bool
    overall_score: float
    
    # Detailed breakdown
    strengths: list[str]
    weaknesses: list[str]
    
    # Improvement suggestions
    priority_improvements: list[ImprovementSuggestion]
    
    # Iteration guidance
    recommended_changes: list[str]
    estimated_impact: dict[str, float]


class ImprovementSuggestion(BaseModel):
    """A specific improvement suggestion."""
    
    criterion: str
    current_value: float
    target_value: float
    gap: float
    
    suggestion: str
    expected_impact: float
    difficulty: Literal["easy", "medium", "hard"]
```

### Feedback Generation Logic

```python
def generate_feedback(
    criterion_results: list[CriterionResult],
    overall_score: float
) -> GateFeedback:
    """
    Generate actionable feedback from gate evaluation.
    """
    # Identify strengths and weaknesses
    strengths = [r for r in criterion_results if r.score >= 0.8]
    weaknesses = [r for r in criterion_results if r.score < 0.7]
    
    # Sort weaknesses by impact potential
    weaknesses.sort(key=lambda r: r.criterion.weight * (1 - r.score), reverse=True)
    
    # Generate improvement suggestions
    suggestions = []
    for weakness in weaknesses[:3]:  # Top 3 priorities
        suggestion = create_improvement_suggestion(weakness)
        suggestions.append(suggestion)
    
    return GateFeedback(
        overall_assessment=assess_overall(overall_score),
        passed=overall_score >= 0.7,
        overall_score=overall_score,
        strengths=[describe_strength(s) for s in strengths],
        weaknesses=[describe_weakness(w) for w in weaknesses],
        priority_improvements=suggestions,
        recommended_changes=generate_changes(weaknesses),
        estimated_impact=estimate_impact(suggestions)
    )
```

## Iteration Loop

```python
async def quality_gate_loop(
    strategy: TradingStrategy,
    gate: QualityGate,
    max_iterations: int = 5
) -> tuple[bool, TradingStrategy, list[GateFeedback]]:
    """
    Iteratively refine strategy until it passes quality gate.
    
    Args:
        strategy: Initial trading strategy
        gate: Quality gate to pass
        max_iterations: Maximum refinement attempts
        
    Returns:
        (passed, final_strategy, feedback_history)
    """
    feedback_history = []
    current_strategy = strategy
    
    for iteration in range(max_iterations):
        # Run backtest
        metrics = await run_backtest(current_strategy)
        
        # Evaluate quality gate
        result = gate.evaluate(metrics)
        feedback_history.append(result.feedback)
        
        if result.passed:
            # Store successful strategy
            await memory.store_strategy(
                strategy=current_strategy,
                metrics=metrics,
                gate_result=result
            )
            return True, current_strategy, feedback_history
        
        # Store lesson learned
        await memory.store_lesson(
            strategy=current_strategy,
            metrics=metrics,
            feedback=result.feedback,
            iteration=iteration
        )
        
        # Refine strategy based on feedback
        current_strategy = await refine_strategy(
            strategy=current_strategy,
            feedback=result.feedback,
            memory=memory
        )
    
    # Max iterations reached
    return False, current_strategy, feedback_history
```

## Adaptive Thresholds

### Market Regime Adjustment

```python
class AdaptiveThresholdManager:
    """Adjust thresholds based on market conditions."""
    
    def __init__(self, memory: ChromaDB):
        self.memory = memory
    
    async def adjust_thresholds(
        self,
        gate: QualityGate,
        current_regime: MarketRegime
    ) -> QualityGate:
        """
        Adjust gate thresholds for current market regime.
        """
        adjusted_criteria = []
        
        for criterion in gate.criteria:
            # Get historical performance in similar regimes
            historical = await self.memory.query_regime_performance(
                regime=current_regime,
                metric=criterion.metric
            )
            
            if historical:
                # Adjust threshold based on regime
                adjustment = self.calculate_adjustment(
                    criterion=criterion,
                    regime=current_regime,
                    historical=historical
                )
                
                adjusted_criterion = criterion.copy()
                adjusted_criterion.threshold *= adjustment
                adjusted_criteria.append(adjusted_criterion)
            else:
                adjusted_criteria.append(criterion)
        
        return QualityGate(
            name=gate.name,
            description=f"{gate.description} (adjusted for {current_regime.name})",
            criteria=adjusted_criteria
        )
```

## Integration with Pipeline

```python
# In main workflow
async def quality_gate_phase(state: PipelineState) -> PipelineState:
    """Execute quality gate validation phase."""
    
    # Build gate from user config
    gate = build_quality_gate(state["user_config"])
    
    # Adjust for market regime if available
    if state.get("market_regime"):
        gate = await adaptive_manager.adjust_thresholds(
            gate=gate,
            current_regime=state["market_regime"]
        )
    
    # Run quality gate loop
    passed, final_strategy, feedback_history = await quality_gate_loop(
        strategy=state["current_strategy"],
        gate=gate,
        max_iterations=state.get("max_iterations", 5)
    )
    
    return {
        **state,
        "gate_results": {
            "passed": passed,
            "final_strategy": final_strategy.dict(),
            "feedback_history": [f.dict() for f in feedback_history]
        }
    }
```
