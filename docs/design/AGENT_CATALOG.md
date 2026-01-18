# Agent Catalog

## Overview

This document defines all agents in the LangChain AlgoTrade Development System. Each agent has a clear role, defined inputs/outputs, and specific tools it uses.

**Total Agents**: 6 primary agents + 18 subagents = 24 agents

---

## Agent Hierarchy

```
LangGraph Workflow (Orchestrator)
│
├── Research Swarm Agent (Primary Agent)
│   ├── Research Leader Agent (Coordinator)
│   ├── Technical Research Subagents (5 agents)
│   ├── Fundamental Research Subagents (5 agents)
│   ├── Sentiment Research Subagents (5 agents)
│   ├── Technical Domain Synthesizer Agent
│   ├── Fundamental Domain Synthesizer Agent
│   └── Sentiment Domain Synthesizer Agent
│
├── Strategy Development Agent (Primary Agent)
│
├── Quality Gate Agent (Primary Agent)
│   ├── Failure Analysis Agent (Sub-agent)
│   └── Trajectory Analyzer Agent (Sub-agent)
│
└── (Parallel Backtest is NOT an agent - it's a LangGraph map-reduce node)
```

---

## Primary Agents

### 1. Research Swarm Agent

**Role**: Coordinate comprehensive market research across technical, fundamental, and sentiment dimensions using hierarchical synthesis.

**Type**: Coordinator Agent (manages subagents)

**Invoked By**: LangGraph workflow (research_swarm node)

**Architecture**: 3-tier hierarchical synthesis
- **Tier 1**: 15 Research Subagents (parallel execution)
- **Tier 2**: 3 Domain Synthesizers (parallel execution)
- **Tier 3**: Research Leader Agent (final synthesis)

**Input**:
```python
{
    "research_directive": str,  # e.g., "Find momentum alpha in tech stocks"
    "ticker": str,              # e.g., "AAPL"
    "timeframe": str,           # e.g., "1d"
    "research_iteration": int,  # Current research iteration
    "previous_findings": List[Dict],  # From memory (if re-researching)
    "failure_feedback": Optional[Dict]  # From Failure Analysis Agent
}
```

**Output**:
```python
{
    "synthesized_findings": List[FactSheet],  # 3 fact sheets (technical, fundamental, sentiment)
    "confidence_score": float,  # 0.0 to 1.0
    "research_quality": str,    # "high" | "medium" | "low"
    "gaps_identified": List[str]  # Areas needing more research
}
```

**Tools Used**:
- Memory System (ChromaDB) - retrieve previous findings
- Market Data APIs (yfinance, news APIs)
- LLM (for synthesis and reasoning)

**Sub-Architecture**: See "Research Swarm Sub-Agents" section below

**Responsibilities**:
1. Spawn 15 research subagents (5 technical, 5 fundamental, 5 sentiment)
2. Collect findings from all subagents
3. Invoke 3 domain synthesizers to create fact sheets
4. Perform final synthesis and quality assessment
5. Store findings in memory
6. Return synthesized findings to LangGraph

---

### 2. Strategy Development Agent

**Role**: Generate trading strategy code variants based on research findings and iteration history.

**Type**: Code Generation Agent

**Invoked By**: LangGraph workflow (strategy_dev node)

**Input**:
```python
{
    "synthesized_findings": List[FactSheet],  # From Research Swarm
    "strategy_iteration": int,  # Current strategy iteration
    "previous_strategies": List[Dict],  # From memory
    "failure_analysis": Optional[Dict],  # From Failure Analysis Agent
    "quality_criteria": Dict[str, float],  # Target metrics
    "num_variants": int  # How many variants to generate (default: 5)
}
```

**Output**:
```python
{
    "strategy_variants": List[StrategyVariant],
    "generation_reasoning": str,  # Why these variants were chosen
    "expected_performance": Dict[str, float]  # Predicted metrics
}

# StrategyVariant schema:
{
    "variant_id": str,
    "name": str,
    "description": str,
    "code": str,  # Complete Python code
    "parameters": Dict[str, Any],
    "approach": str,  # "momentum" | "mean_reversion" | "breakout" | etc.
    "complexity": str  # "simple" | "moderate" | "complex"
}
```

**Tools Used**:
- Memory System (ChromaDB) - retrieve strategy library
- Code Generator Tool
- Code Validator Tool
- LLM (for code generation and reasoning)

**Responsibilities**:
1. Analyze research findings
2. Review previous strategy failures (if any)
3. Generate N strategy variants (different parameters or approaches)
4. Validate all generated code (syntax, logic, Backtrader compatibility)
5. Store strategies in memory
6. Return variants to LangGraph

**Variant Generation Strategy**:
- **First iteration**: Generate diverse approaches (momentum, mean reversion, breakout)
- **Subsequent iterations**: 
  - If TUNE recommended: Generate parameter variations
  - If FIX recommended: Fix bugs and regenerate
  - If REFINE recommended: Redesign algorithm logic
  - If RESEARCH recommended: Wait for new research findings

---

### 3. Quality Gate Agent

**Role**: Evaluate backtest results, make pass/fail decisions, and determine next action using three-tier feedback loops.

**Type**: Decision-Making Agent

**Invoked By**: LangGraph workflow (quality_gate node)

**Input**:
```python
{
    "backtest_results": List[BacktestResult],  # From parallel backtest node
    "quality_criteria": Dict[str, float],  # User-defined thresholds
    "experiment_history": List[ExperimentRecord],  # All previous iterations
    "strategy_iteration": int,
    "research_iteration": int,
    "total_iterations": int,
    "max_iterations": Dict[str, int]  # Limits
}
```

**Output**:
```python
{
    "decision": str,  # "SUCCESS" | "TUNE" | "FIX" | "REFINE" | "RESEARCH" | "ABANDON"
    "best_variant": Optional[StrategyVariant],  # If any passed
    "gate_evaluation": Dict[str, Any],  # Detailed evaluation
    "failure_analysis": Optional[Dict],  # From Failure Analysis Agent
    "trajectory_analysis": Optional[Dict],  # From Trajectory Analyzer Agent
    "reasoning": str,  # Why this decision was made
    "specific_actions": List[str]  # Recommended actions
}
```

**Tools Used**:
- Fuzzy Logic Scorer (evaluate metrics)
- Experiment Tracker (read history)
- Memory System (store lessons learned)
- LLM (for reasoning and decision-making)

**Sub-Agents**:
- **Failure Analysis Agent** (invoked if all variants failed)
- **Trajectory Analyzer Agent** (invoked if sufficient history exists)

**Responsibilities**:
1. Evaluate all backtest results against quality criteria
2. Compute fuzzy logic scores for each variant
3. Check if any variant passed (→ SUCCESS)
4. If all failed:
   a. Invoke Failure Analysis Agent
   b. Invoke Trajectory Analyzer Agent (if history ≥ 2 iterations)
   c. Combine analyses
   d. Make routing decision (three-tier feedback loops)
5. Check iteration limits
6. Store lessons learned
7. Return decision to LangGraph

**Three-Tier Feedback Loop Logic**:
```python
if any_variant_passed:
    return "SUCCESS"

# Tier 1: Strategy Refinement
if failure_classification == "PARAMETER_ISSUE":
    return "TUNE"
elif failure_classification == "ALGORITHM_BUG":
    return "FIX"
elif failure_classification == "DESIGN_FLAW" and strategy_iteration < 3:
    return "REFINE"

# Tier 2: Research Refinement
elif failure_classification == "RESEARCH_GAP":
    return "RESEARCH"
elif failure_classification == "DESIGN_FLAW" and strategy_iteration >= 3:
    return "RESEARCH"  # Multiple refinements failed, need better research

# Tier 3: Abandonment
elif failure_classification == "FUNDAMENTAL_IMPOSSIBILITY":
    return "ABANDON"
elif total_iterations >= max_total_iterations:
    return "ABANDON"
elif trajectory_status == "DIVERGING":
    return "ABANDON"
else:
    return "ABANDON"  # Fallback
```

---

### 4. Failure Analysis Agent

**Role**: Diagnose why strategies failed and classify failures into actionable categories.

**Type**: Diagnostic Agent

**Invoked By**: Quality Gate Agent (when all variants fail)

**Input**:
```python
{
    "strategy_code": str,  # Latest strategy code
    "strategy_parameters": Dict[str, Any],
    "backtest_metrics": Dict[str, float],
    "quality_gate_results": Dict[str, Any],
    "research_findings": List[FactSheet],
    "iteration_history": List[ExperimentRecord],
    "current_iteration": int,
    "max_iterations": int
}
```

**Output**:
```python
{
    "failure_classification": str,  # One of 5 categories
    "root_cause": str,  # Detailed explanation
    "confidence": float,  # 0.0 to 1.0
    "specific_actions": List[str],  # Actionable recommendations
    "bug_detection": Optional[Dict],  # If bugs found
    "statistical_assessment": Dict[str, Any],
    "recommendation": str,  # "TUNE" | "FIX" | "REFINE" | "RESEARCH" | "ABANDON"
    "reasoning": str  # LLM's reasoning process
}
```

**Failure Classifications**:
1. **PARAMETER_ISSUE**: Logic is sound, parameters need tuning
   - Example: Sharpe 0.85 (close to 1.0 threshold), just need parameter adjustment
   - Recommendation: TUNE

2. **ALGORITHM_BUG**: Implementation error in code
   - Example: Incorrect RSI calculation, off-by-one error, wrong signal logic
   - Recommendation: FIX

3. **DESIGN_FLAW**: Missing features or conceptual issues
   - Example: No regime awareness, no risk management, no position sizing
   - Recommendation: REFINE

4. **RESEARCH_GAP**: Insufficient or incorrect research
   - Example: Research didn't identify key market patterns, wrong hypothesis
   - Recommendation: RESEARCH

5. **FUNDAMENTAL_IMPOSSIBILITY**: Alpha doesn't exist
   - Example: All approaches fail, no edge found, market too efficient
   - Recommendation: ABANDON

**Tools Used**:
- Code Analysis Tool (detect bugs, check logic)
- Statistical Analysis Tool (assess metrics)
- LLM (for reasoning and diagnosis)

**Responsibilities**:
1. Analyze strategy code for bugs
2. Analyze backtest metrics for patterns
3. Compare metrics to quality criteria (distance from threshold)
4. Review iteration history for consistent failures
5. Classify failure into one of 5 categories
6. Provide specific actionable recommendations
7. Return analysis to Quality Gate Agent

---

### 5. Trajectory Analyzer Agent

**Role**: Analyze experiment trajectory over multiple iterations to detect convergence, divergence, or stagnation patterns.

**Type**: Statistical Analysis Agent

**Invoked By**: Quality Gate Agent (when history ≥ 2 iterations)

**Input**:
```python
{
    "experiment_history": List[ExperimentRecord],  # All iterations
    "key_metrics": List[str],  # ["sharpe_ratio", "max_drawdown", "win_rate"]
    "quality_criteria": Dict[str, float]
}
```

**Output**:
```python
{
    "trajectory_status": str,  # "CONVERGING" | "DIVERGING" | "OSCILLATING" | "STAGNANT"
    "convergence_assessment": Dict[str, Any],
    "improvement_rates": Dict[str, float],  # Per metric
    "parameter_impact": Dict[str, float],  # Which parameters matter most
    "iterations_to_convergence": Optional[int],  # Estimated
    "next_action": str,  # "CONTINUE" | "PIVOT" | "ABANDON"
    "reasoning": str,  # LLM's interpretation
    "visualizations": List[str]  # Paths to generated charts
}
```

**Trajectory Statuses**:
1. **CONVERGING**: Metrics improving consistently
   - Improvement rate > 5% per iteration
   - Consistent direction
   - Recommendation: CONTINUE

2. **DIVERGING**: Metrics getting worse
   - Improvement rate < -5% per iteration
   - Moving away from thresholds
   - Recommendation: ABANDON

3. **OSCILLATING**: Metrics fluctuating without clear trend
   - High variance, no consistent direction
   - Recommendation: PIVOT or ABANDON

4. **STAGNANT**: Metrics not changing
   - Improvement rate near 0%
   - Stuck in local optimum
   - Recommendation: PIVOT

**Tools Used**:
- Experiment Tracker (read JSONL logs)
- Statistical Analysis Tool (compute trends, rates)
- Visualization Tool (generate charts)
- LLM (for interpretation and reasoning)

**Responsibilities**:
1. Load experiment history from Experiment Tracker
2. Compute statistical metrics (improvement rates, variance, trends)
3. Detect convergence/divergence patterns
4. Analyze parameter impact (which changes helped/hurt)
5. Estimate iterations to convergence
6. Generate visualizations
7. Use LLM to interpret patterns and recommend action
8. Return analysis to Quality Gate Agent

---

## Research Swarm Sub-Agents

### Research Leader Agent

**Role**: Coordinate research subagents and domain synthesizers.

**Type**: Coordinator Agent

**Invoked By**: Research Swarm Agent

**Responsibilities**:
1. Spawn 15 research subagents (5 technical, 5 fundamental, 5 sentiment)
2. Collect findings from all subagents
3. Detect conflicts between findings
4. Invoke 3 domain synthesizers
5. Perform final synthesis
6. Assess research quality
7. Return synthesized findings

---

### Technical Research Subagents (5 agents)

**Role**: Analyze technical indicators and price patterns.

**Type**: Research Agent

**Invoked By**: Research Leader Agent (parallel execution)

**Specializations**:
1. **Trend Analysis Agent**: Moving averages, trend strength, support/resistance
2. **Momentum Analysis Agent**: RSI, MACD, Stochastic, momentum indicators
3. **Volatility Analysis Agent**: Bollinger Bands, ATR, volatility regimes
4. **Volume Analysis Agent**: Volume patterns, OBV, accumulation/distribution
5. **Pattern Recognition Agent**: Chart patterns, candlestick patterns

**Input**:
```python
{
    "ticker": str,
    "timeframe": str,
    "lookback_period": int,  # Days of historical data
    "specialization": str  # Which aspect to focus on
}
```

**Output**:
```python
{
    "agent_id": str,
    "specialization": str,
    "findings": List[Finding],
    "confidence": float,
    "data_quality": str
}

# Finding schema:
{
    "type": "technical",
    "indicator": str,
    "value": float,
    "interpretation": str,
    "signal": str,  # "bullish" | "bearish" | "neutral"
    "confidence": float,
    "evidence": Dict[str, Any]
}
```

**Tools Used**:
- yfinance (price data)
- TA-Lib (technical indicators)
- pandas (data manipulation)

---

### Fundamental Research Subagents (5 agents)

**Role**: Analyze fundamental data and company metrics.

**Type**: Research Agent

**Invoked By**: Research Leader Agent (parallel execution)

**Specializations**:
1. **Financial Metrics Agent**: Revenue, earnings, margins, growth rates
2. **Valuation Agent**: P/E, P/B, P/S, DCF valuation
3. **Balance Sheet Agent**: Assets, liabilities, debt ratios, liquidity
4. **Cash Flow Agent**: Operating cash flow, free cash flow, capital allocation
5. **Industry Analysis Agent**: Sector trends, competitive position, market share

**Input**:
```python
{
    "ticker": str,
    "specialization": str
}
```

**Output**:
```python
{
    "agent_id": str,
    "specialization": str,
    "findings": List[Finding],
    "confidence": float,
    "data_quality": str
}

# Finding schema:
{
    "type": "fundamental",
    "metric": str,
    "value": float,
    "interpretation": str,
    "signal": str,  # "positive" | "negative" | "neutral"
    "confidence": float,
    "evidence": Dict[str, Any]
}
```

**Tools Used**:
- yfinance (fundamental data)
- Financial APIs (detailed metrics)
- LLM (for interpretation)

---

### Sentiment Research Subagents (5 agents)

**Role**: Analyze market sentiment from various sources.

**Type**: Research Agent

**Invoked By**: Research Leader Agent (parallel execution)

**Specializations**:
1. **News Sentiment Agent**: News articles, headlines, media coverage
2. **Social Media Agent**: Twitter, Reddit, StockTwits sentiment
3. **Analyst Sentiment Agent**: Analyst ratings, price targets, recommendations
4. **Options Flow Agent**: Put/call ratios, unusual options activity
5. **Insider Activity Agent**: Insider buying/selling, institutional ownership

**Input**:
```python
{
    "ticker": str,
    "specialization": str,
    "lookback_period": int  # Days
}
```

**Output**:
```python
{
    "agent_id": str,
    "specialization": str,
    "findings": List[Finding],
    "confidence": float,
    "data_quality": str
}

# Finding schema:
{
    "type": "sentiment",
    "source": str,
    "sentiment_score": float,  # -1.0 to 1.0
    "interpretation": str,
    "signal": str,  # "bullish" | "bearish" | "neutral"
    "confidence": float,
    "evidence": Dict[str, Any]
}
```

**Tools Used**:
- News APIs
- Social media APIs
- Financial data APIs
- LLM (for sentiment analysis)

---

### Domain Synthesizer Agents (3 agents)

**Role**: Synthesize findings within a specific domain (technical, fundamental, sentiment) into a cohesive fact sheet.

**Type**: Synthesis Agent

**Invoked By**: Research Leader Agent (parallel execution)

**Specializations**:
1. **Technical Domain Synthesizer**: Synthesizes 5 technical subagent findings
2. **Fundamental Domain Synthesizer**: Synthesizes 5 fundamental subagent findings
3. **Sentiment Domain Synthesizer**: Synthesizes 5 sentiment subagent findings

**Input**:
```python
{
    "domain": str,  # "technical" | "fundamental" | "sentiment"
    "subagent_findings": List[Finding],  # 5 findings from subagents
    "research_directive": str
}
```

**Output**:
```python
{
    "fact_sheet": FactSheet
}

# FactSheet schema:
{
    "domain": str,
    "key_insights": List[str],  # 3-5 key insights
    "supporting_evidence": List[Dict],
    "conflicting_evidence": List[Dict],
    "confidence_score": float,
    "signal": str,  # "bullish" | "bearish" | "neutral"
    "recommended_approach": str,  # Strategy suggestion
    "risks": List[str],
    "opportunities": List[str]
}
```

**Tools Used**:
- LLM (for synthesis and reasoning)
- Conflict resolution logic

**Responsibilities**:
1. Receive 5 findings from subagents in the same domain
2. Identify common themes and patterns
3. Resolve conflicts (weighted voting, evidence strength)
4. Extract key insights (3-5 most important)
5. Assess overall confidence
6. Provide strategy recommendations
7. Return fact sheet to Research Leader

---

## Non-Agent Components

### Parallel Backtest Node

**Type**: LangGraph Map-Reduce Node (NOT an agent)

**Role**: Execute backtests for all strategy variants in parallel.

**Implementation**:
```python
async def parallel_backtest_node(state: WorkflowState):
    """
    LangGraph node that executes backtests in parallel.
    Uses asyncio.gather() for parallel execution.
    """
    strategy_variants = state["strategy_variants"]
    
    # Execute all backtests in parallel
    results = await asyncio.gather(*[
        execute_backtest(variant, state["ticker"], state["timeframe"])
        for variant in strategy_variants
    ])
    
    return {"backtest_results": results}
```

**Not an agent because**:
- No LLM reasoning required
- Pure computational task
- No decision-making
- Just executes Backtrader and logs results

---

## Agent Communication Patterns

### Pattern 1: Sequential (Agent → Agent)

```
Research Swarm Agent
    ↓ (synthesized_findings)
Strategy Development Agent
    ↓ (strategy_variants)
Parallel Backtest Node
    ↓ (backtest_results)
Quality Gate Agent
```

### Pattern 2: Parallel (Coordinator → Multiple Subagents)

```
Research Leader Agent
    ├→ Technical Subagent 1
    ├→ Technical Subagent 2
    ├→ ...
    └→ Technical Subagent 5
    
All execute in parallel, results aggregated
```

### Pattern 3: Hierarchical (Coordinator → Subagents → Synthesizers)

```
Research Leader Agent
    ├→ Spawn 15 subagents (parallel)
    ├→ Collect 15 findings
    ├→ Spawn 3 synthesizers (parallel)
    ├→ Collect 3 fact sheets
    └→ Final synthesis
```

### Pattern 4: Conditional Invocation (Agent → Sub-agent if needed)

```
Quality Gate Agent
    ├→ If all failed: Invoke Failure Analysis Agent
    └→ If history ≥ 2: Invoke Trajectory Analyzer Agent
```

---

## Agent Responsibilities Matrix

| Agent | Primary Responsibility | Invokes | Returns To |
|-------|----------------------|---------|------------|
| Research Swarm Agent | Coordinate research | 15 subagents + 3 synthesizers | LangGraph |
| Strategy Development Agent | Generate strategy code | None | LangGraph |
| Quality Gate Agent | Evaluate & route | Failure Analysis, Trajectory Analyzer | LangGraph |
| Failure Analysis Agent | Diagnose failures | None | Quality Gate |
| Trajectory Analyzer Agent | Analyze trends | None | Quality Gate |
| Research Leader Agent | Coordinate research swarm | 15 subagents + 3 synthesizers | Research Swarm Agent |
| Technical Subagents (5) | Analyze technical data | None | Research Leader |
| Fundamental Subagents (5) | Analyze fundamentals | None | Research Leader |
| Sentiment Subagents (5) | Analyze sentiment | None | Research Leader |
| Domain Synthesizers (3) | Synthesize findings | None | Research Leader |

---

## Summary

**Total Agents**: 24
- **Primary Agents**: 5 (Research Swarm, Strategy Dev, Quality Gate, Failure Analysis, Trajectory Analyzer)
- **Coordinator Agents**: 1 (Research Leader)
- **Research Subagents**: 15 (5 technical + 5 fundamental + 5 sentiment)
- **Synthesizer Agents**: 3 (technical + fundamental + sentiment)

**Non-Agent Components**: 1 (Parallel Backtest Node - LangGraph map-reduce)

**Orchestration**: LangGraph StateGraph (manages state, routing, execution)

---

**End of Agent Catalog**
