# Feedback Loops and Iteration Logic

## Executive Summary

This document defines the feedback loops and iteration logic between the Quality Gate Validation Phase, Strategy Development Phase, and Research Swarm Phase. It establishes clear rules for when the system should iterate within a phase, when to go back to a previous phase, and when to abandon a research direction entirely.

---

## Current State Analysis

### What Exists in Current Design

The current design includes a `quality_gate_loop()` function that iterates up to `max_iterations` times (default: 5) within the **Strategy Development and Backtesting phases**. This loop:

1. Runs backtest on current strategy
2. Evaluates metrics against quality gate criteria
3. If PASS → exits loop and proceeds to output
4. If FAIL → generates feedback and refines strategy
5. Repeats until PASS or max iterations reached

**Key Finding**: The current design only iterates within Strategy Development. There is **no explicit logic** for:
- Going back to Research Swarm Phase
- Deciding when research findings are insufficient
- Abandoning a research direction entirely

---

## Problem Statement

We need to answer three critical questions:

### Question 1: When does the system go back to RESEARCH SWARM PHASE?

**Scenario**: After multiple iterations of strategy development and optimization, the strategies consistently fail quality gates. The problem may not be the strategy implementation, but the underlying research findings or research direction.

**Current Gap**: No mechanism exists to detect this scenario and trigger new research.

### Question 2: When does the system go back to STRATEGY DEVELOPMENT PHASE?

**Scenario**: A strategy fails quality gates, but the failure is fixable through parameter tuning, logic adjustments, or code refinement.

**Current State**: This is handled by the existing `quality_gate_loop()`.

### Question 3: When does the system abandon a research direction?

**Scenario**: After exhausting both research iterations and strategy iterations, the system cannot produce a viable strategy. Continuing would waste resources.

**Current Gap**: No mechanism exists to detect futility and abandon a research direction.

---

## Proposed Solution: Three-Tier Iteration System

### Tier 1: Strategy Refinement Loop (Existing)

**Scope**: Strategy Development → Backtesting → Quality Gates → Strategy Development

**Max Iterations**: 5 (configurable)

**Trigger**: Quality gate failure with actionable feedback

**Actions**:
- Adjust strategy parameters
- Refine entry/exit logic
- Modify risk management rules
- Re-run backtest and re-evaluate

**Exit Conditions**:
- ✅ **SUCCESS**: Strategy passes quality gates → Proceed to Output Phase
- ❌ **FAILURE**: Max iterations reached → Escalate to Tier 2

**Example**:
```
Iteration 1: Sharpe 0.8 (target: 1.0) → Feedback: "Increase position sizing"
Iteration 2: Sharpe 1.1, Max DD 25% (target: 20%) → Feedback: "Tighten stop loss"
Iteration 3: Sharpe 1.2, Max DD 18% → PASS
```

---

### Tier 2: Research Refinement Loop (NEW)

**Scope**: Research Swarm → Strategy Development → Backtesting → Quality Gates → Research Swarm

**Max Iterations**: 3 (configurable)

**Trigger**: Tier 1 exhausted without success

**Decision Logic**:

The system analyzes the failure pattern to determine if the problem is **strategic** (bad research direction) or **tactical** (bad strategy implementation).

**Indicators for Research Refinement**:

1. **Consistent Failure Across Multiple Strategies**: If 3+ different strategy variants all fail on the same criteria, the research findings may be flawed.

2. **Fundamental Metric Failures**: If strategies consistently fail on fundamental metrics (e.g., negative returns, extremely high drawdown), the underlying alpha hypothesis is likely wrong.

3. **Conflicting Research Findings**: If the research swarm produced conflicting or low-confidence findings, the research may need to be deeper or more focused.

4. **Missing Research Dimensions**: If strategies fail due to lack of regime awareness, sentiment analysis, or other dimensions not covered in initial research.

**Actions**:
- Store failed strategies as "lessons learned" with failure analysis
- Generate refined research directive based on failure patterns
- Re-run Research Swarm with:
  - Different subagent focus areas
  - Additional data sources
  - Longer time horizons
  - Different market regimes
- Generate new strategies from refined research

**Exit Conditions**:
- ✅ **SUCCESS**: New strategies pass quality gates → Proceed to Output Phase
- ❌ **FAILURE**: Max iterations reached → Escalate to Tier 3

**Example**:
```
Research Iteration 1: "AAPL momentum" → 5 strategies fail (all negative Sharpe)
  Analysis: Momentum hypothesis may be wrong
  
Research Iteration 2: "AAPL mean reversion" → 3 strategies fail (high drawdown)
  Analysis: Need regime awareness
  
Research Iteration 3: "AAPL regime-adaptive" → 2 strategies pass
  SUCCESS
```

---

### Tier 3: Abandonment Decision (NEW)

**Scope**: Entire research direction

**Trigger**: Tier 2 exhausted without success

**Decision Logic**:

After exhausting both strategy refinement (Tier 1) and research refinement (Tier 2), the system must decide whether to:
1. **Abandon** this research direction entirely
2. **Escalate** to human review
3. **Pivot** to a completely different approach

**Abandonment Criteria**:

1. **Total Iterations Exceeded**: (Tier 1 iterations × Tier 2 iterations) > threshold (default: 15)

2. **Negative ROI**: Estimated time/cost to continue > expected value of potential strategy

3. **Fundamental Impossibility**: Research findings indicate the alpha opportunity doesn't exist (e.g., market is efficient in this dimension)

4. **Resource Constraints**: Computational budget or time budget exhausted

**Actions**:
- Store comprehensive failure report in "lessons learned"
- Document why this research direction failed
- Extract generalizable insights (e.g., "momentum strategies don't work in low-volatility regimes")
- Recommend alternative research directions
- **Optionally**: Escalate to human for review

**Exit Conditions**:
- 🛑 **ABANDON**: Research direction abandoned, move to next research objective
- 👤 **ESCALATE**: Human reviews and decides next steps

**Example**:
```
Total Attempts: 15 (5 strategy iterations × 3 research iterations)
Failure Pattern: All strategies have negative returns
Root Cause: "AAPL has been in a strong downtrend; long-only strategies cannot work"
Lesson Learned: "Always check market regime before developing long-only strategies"
Recommendation: "Try short-only or market-neutral strategies, or pivot to different ticker"
Decision: ABANDON
```

---

## Decision Tree

```
┌─────────────────────────────────────────────────────────────────────┐
│                      QUALITY GATE EVALUATION                        │
└─────────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
                  PASS                FAIL
                    │                   │
                    ▼                   ▼
            ┌───────────────┐   ┌──────────────────┐
            │ OUTPUT PHASE  │   │ Analyze Failure  │
            └───────────────┘   └──────────────────┘
                                        │
                        ┌───────────────┼───────────────┐
                        │               │               │
                   Fixable?      Fundamental?    Exhausted?
                        │               │               │
                        ▼               ▼               ▼
              ┌─────────────────┐ ┌─────────────┐ ┌─────────────┐
              │ TIER 1:         │ │ TIER 2:     │ │ TIER 3:     │
              │ Strategy        │ │ Research    │ │ Abandonment │
              │ Refinement      │ │ Refinement  │ │ Decision    │
              └─────────────────┘ └─────────────┘ └─────────────┘
                        │               │               │
                        │               │               │
                  Max 5 iter      Max 3 iter      Abandon/Escalate
                        │               │               │
                        └───────────────┴───────────────┘
                                        │
                                        ▼
                        ┌───────────────────────────────┐
                        │ Back to Quality Gate or Exit  │
                        └───────────────────────────────┘
```

---

## Routing Rules

### Rule 1: Strategy Refinement (Tier 1)

**Condition**: Quality gate failure with `iteration_count < max_strategy_iterations`

**Failure Type Classification**:
- **Parameter Issue**: Metrics close to threshold (within 20%)
- **Logic Issue**: Specific criteria failing consistently
- **Risk Issue**: Risk metrics failing but performance metrics passing

**Action**: 
```python
if failure_type in ["parameter", "logic", "risk"]:
    feedback = generate_improvement_feedback(metrics, criteria)
    refined_strategy = refine_strategy(strategy, feedback)
    return "STRATEGY_DEVELOPMENT_PHASE"
```

**Example Feedback**:
- "Sharpe ratio is 0.85 (target: 1.0). Try increasing position size or tightening entry conditions."
- "Max drawdown is 22% (target: 20%). Implement tighter stop losses or reduce leverage."

---

### Rule 2: Research Refinement (Tier 2)

**Condition**: Tier 1 exhausted AND `research_iteration_count < max_research_iterations`

**Failure Pattern Classification**:
- **Consistent Failure**: 3+ strategy variants fail on same criteria
- **Fundamental Failure**: All strategies have negative returns or extreme risk
- **Missing Dimension**: Strategies fail due to lack of specific analysis (e.g., regime, sentiment)

**Action**:
```python
if failure_pattern in ["consistent", "fundamental", "missing_dimension"]:
    failure_analysis = analyze_failure_patterns(failed_strategies)
    research_directive = generate_research_directive(failure_analysis)
    store_lesson_learned(failure_analysis)
    return "RESEARCH_SWARM_PHASE"
```

**Example Research Directives**:
- "Original research focused on momentum. Failure analysis shows momentum doesn't work in current regime. New directive: Research mean reversion strategies for AAPL."
- "Strategies lack regime awareness. New directive: Research market regimes for AAPL and develop regime-adaptive strategies."
- "Sentiment analysis was missing. New directive: Include sentiment subagent in research swarm."

---

### Rule 3: Abandonment (Tier 3)

**Condition**: Tier 2 exhausted OR abandonment criteria met

**Abandonment Criteria**:
```python
def should_abandon(state: WorkflowState) -> bool:
    total_iterations = state.strategy_iterations * state.research_iterations
    
    # Criterion 1: Total iterations exceeded
    if total_iterations > state.max_total_iterations:
        return True
    
    # Criterion 2: All strategies have negative returns
    if all(s.annual_return < 0 for s in state.failed_strategies):
        return True
    
    # Criterion 3: No improvement across research iterations
    if len(state.research_iterations) >= 2:
        iter1_best = max(s.sharpe for s in state.research_iter_1_strategies)
        iter2_best = max(s.sharpe for s in state.research_iter_2_strategies)
        if iter2_best <= iter1_best * 1.1:  # Less than 10% improvement
            return True
    
    # Criterion 4: Computational budget exhausted
    if state.compute_time_used > state.max_compute_time:
        return True
    
    return False
```

**Action**:
```python
if should_abandon(state):
    failure_report = generate_failure_report(state)
    store_lesson_learned(failure_report)
    
    if state.require_human_review:
        return "HUMAN_REVIEW"
    else:
        return "ABANDON"
```

---

## State Management

### Workflow State Schema

```python
class WorkflowState(BaseModel):
    # Current phase
    current_phase: Literal["research", "strategy", "backtest", "quality_gate", "output"]
    
    # Iteration counters
    strategy_iteration: int = 0
    research_iteration: int = 0
    total_iterations: int = 0
    
    # Limits
    max_strategy_iterations: int = 5
    max_research_iterations: int = 3
    max_total_iterations: int = 15
    
    # Research state
    research_directive: str
    research_findings: list[ResearchFinding]
    
    # Strategy state
    current_strategy: TradingStrategy
    failed_strategies: list[TradingStrategy]
    strategy_variants: list[TradingStrategy]
    
    # Quality gate state
    gate_results: list[GateResult]
    feedback_history: list[str]
    
    # Failure analysis
    failure_patterns: list[str]
    lessons_learned: list[Lesson]
    
    # Resource tracking
    compute_time_used: float
    max_compute_time: float
    
    # Human-in-the-loop
    require_human_review: bool = False
```

---

## Implementation Example

```python
async def execute_workflow_with_feedback_loops(
    user_config: UserConfiguration
) -> WorkflowResult:
    """
    Execute the complete workflow with three-tier feedback loops.
    """
    state = WorkflowState(
        research_directive=user_config.alpha_direction,
        max_strategy_iterations=user_config.max_strategy_iterations,
        max_research_iterations=user_config.max_research_iterations,
        max_total_iterations=user_config.max_total_iterations
    )
    
    # Start with research
    state.current_phase = "research"
    
    while True:
        # RESEARCH SWARM PHASE
        if state.current_phase == "research":
            state.research_iteration += 1
            
            research_findings = await run_research_swarm(
                directive=state.research_directive,
                lessons_learned=state.lessons_learned
            )
            state.research_findings = research_findings
            state.current_phase = "strategy"
        
        # STRATEGY DEVELOPMENT PHASE
        if state.current_phase == "strategy":
            state.strategy_iteration += 1
            state.total_iterations += 1
            
            strategy = await develop_strategy(
                research_findings=state.research_findings,
                feedback=state.feedback_history[-1] if state.feedback_history else None
            )
            state.current_strategy = strategy
            state.current_phase = "backtest"
        
        # BACKTESTING PHASE
        if state.current_phase == "backtest":
            metrics = await run_backtest(state.current_strategy)
            state.current_phase = "quality_gate"
        
        # QUALITY GATE PHASE
        if state.current_phase == "quality_gate":
            result = await evaluate_quality_gate(
                metrics=metrics,
                criteria=user_config.quality_criteria
            )
            state.gate_results.append(result)
            
            if result.passed:
                # SUCCESS!
                state.current_phase = "output"
                break
            
            # FAILED - Determine next action
            next_action = determine_next_action(state, result)
            
            if next_action == "REFINE_STRATEGY":
                # TIER 1: Strategy Refinement
                feedback = generate_strategy_feedback(result)
                state.feedback_history.append(feedback)
                state.current_phase = "strategy"
            
            elif next_action == "REFINE_RESEARCH":
                # TIER 2: Research Refinement
                state.failed_strategies.append(state.current_strategy)
                failure_analysis = analyze_failure_patterns(state.failed_strategies)
                state.lessons_learned.append(failure_analysis)
                
                new_directive = generate_research_directive(failure_analysis)
                state.research_directive = new_directive
                state.strategy_iteration = 0  # Reset strategy counter
                state.current_phase = "research"
            
            elif next_action == "ABANDON":
                # TIER 3: Abandonment
                failure_report = generate_failure_report(state)
                store_lesson_learned(failure_report)
                
                if state.require_human_review:
                    return WorkflowResult(
                        status="ESCALATED",
                        message="Research direction exhausted. Human review required.",
                        state=state
                    )
                else:
                    return WorkflowResult(
                        status="ABANDONED",
                        message="Research direction abandoned after exhausting all iterations.",
                        state=state
                    )
        
        # OUTPUT PHASE
        if state.current_phase == "output":
            return WorkflowResult(
                status="SUCCESS",
                strategy=state.current_strategy,
                metrics=metrics,
                state=state
            )


def determine_next_action(
    state: WorkflowState,
    result: GateResult
) -> Literal["REFINE_STRATEGY", "REFINE_RESEARCH", "ABANDON"]:
    """
    Determine the next action based on current state and failure result.
    """
    # Check abandonment first
    if should_abandon(state):
        return "ABANDON"
    
    # Check if we should refine research
    if state.strategy_iteration >= state.max_strategy_iterations:
        if state.research_iteration < state.max_research_iterations:
            return "REFINE_RESEARCH"
        else:
            return "ABANDON"
    
    # Check failure pattern
    failure_pattern = classify_failure_pattern(state.gate_results)
    
    if failure_pattern in ["consistent_fundamental", "missing_dimension"]:
        if state.research_iteration < state.max_research_iterations:
            return "REFINE_RESEARCH"
    
    # Default: refine strategy
    return "REFINE_STRATEGY"
```

---

## Summary

### When to go back to RESEARCH SWARM PHASE?

**Answer**: When Tier 1 (Strategy Refinement) is exhausted AND failure analysis indicates the problem is **strategic** (bad research direction) rather than **tactical** (bad strategy implementation).

**Specific Triggers**:
- 3+ strategy variants fail on the same criteria
- All strategies have negative returns or extreme risk
- Strategies lack a specific research dimension (regime, sentiment, etc.)
- Research iteration count < max (default: 3)

### When to go back to STRATEGY DEVELOPMENT PHASE?

**Answer**: When quality gates fail AND the failure is **fixable** through parameter tuning, logic adjustments, or code refinement.

**Specific Triggers**:
- Metrics are close to thresholds (within 20%)
- Specific criteria failing consistently
- Strategy iteration count < max (default: 5)

### When to abandon a research direction?

**Answer**: When both Tier 1 and Tier 2 are exhausted OR abandonment criteria are met.

**Specific Triggers**:
- Total iterations > threshold (default: 15)
- All strategies have negative returns
- No improvement across research iterations (< 10% improvement)
- Computational budget exhausted
- Human review required

---

**End of Feedback Loops Document**
