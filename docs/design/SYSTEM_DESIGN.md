# System Design: LangChain AlgoTrade Development System

## Executive Summary

The LangChain AlgoTrade Development System is a **multi-agent system** that autonomously researches, develops, and optimizes algorithmic trading strategies using LangChain and LangGraph.

**Core Architecture**: LangGraph orchestrates 24 specialized agents that collaborate through a state-driven workflow with intelligent feedback loops.

**Key Innovation**: Hierarchical synthesis prevents cognitive overload, LLM-powered failure analysis enables intelligent iteration, and experiment tracking guides convergence.

---

## System Requirements

### Functional Requirements

1. Conduct systematic research on trading strategies and market patterns using hierarchical agent swarm
2. Develop trading algorithms based on research findings with automatic code generation
3. Backtest strategies using Backtrader framework with parallel execution
4. Validate strategies against quality gates with intelligent failure analysis
5. Store and retrieve knowledge using vector stores (ChromaDB) for persistent learning
6. Track experiments and analyze trajectories for convergence detection
7. Support three-tier feedback loops for intelligent iteration strategy

### Non-Functional Requirements

1. Modular agent-based architecture allowing agent expansion
2. LangGraph-based orchestration (not custom workflow engine)
3. Persistent memory across sessions with lineage tracking
4. Error recovery with LLM-powered failure analysis
5. Comprehensive experiment logging and trajectory analysis
6. Parallel backtest execution for efficiency

### Technical Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| Agent Framework | LangChain + LangGraph | State-of-the-art agent orchestration with graph-based workflows |
| LLM Provider | OpenAI-compatible API | Flexibility to use various models with automatic failover |
| Vector Store | ChromaDB | Lightweight, local, supports semantic search with embeddings |
| Data Source | yfinance | Free, reliable historical market data |
| Backtesting | Backtrader | Mature, feature-rich Python backtesting framework |
| Experiment Tracking | JSONL | Simple, append-only, easy to parse |
| Language | Python 3.11+ | Modern Python with type hints and async support |

---

## Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            USER INPUT                                   │
│  • Ticker: AAPL                                                         │
│  • Research Directive: "Find momentum alpha in tech stocks"             │
│  • Quality Criteria: {sharpe: 1.0, max_drawdown: 0.20, win_rate: 0.50} │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     LANGGRAPH WORKFLOW                                  │
│                    (State-Driven Orchestration)                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  Node: research_swarm                                             │  │
│  │  Agent: Research Swarm Agent (19 agents total)                    │  │
│  │  Output: 3 fact sheets (technical, fundamental, sentiment)        │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                              │                                           │
│                              ▼                                           │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  Node: strategy_dev                                               │  │
│  │  Agent: Strategy Development Agent                                │  │
│  │  Output: N strategy variants                                      │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                              │                                           │
│                              ▼                                           │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  Node: parallel_backtest                                          │  │
│  │  Type: Map-Reduce (NOT an agent)                                  │  │
│  │  Output: N backtest results                                       │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                              │                                           │
│                              ▼                                           │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  Node: quality_gate                                               │  │
│  │  Agent: Quality Gate Agent                                        │  │
│  │  Sub-agents: Failure Analysis, Trajectory Analyzer                │  │
│  │  Output: Decision (SUCCESS | TUNE | REFINE | RESEARCH | ABANDON) │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                              │                                           │
│                              ▼                                           │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  Conditional Routing (Three-Tier Feedback Loops)                  │  │
│  │  ├── SUCCESS → END (return best strategy)                         │  │
│  │  ├── TUNE/FIX/REFINE → strategy_dev (Tier 1)                      │  │
│  │  ├── RESEARCH → research_swarm (Tier 2)                           │  │
│  │  └── ABANDON → END (return failure)                               │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         SHARED TOOLS & MEMORY                           │
│  • Memory System (ChromaDB): Research findings, strategies, lessons    │
│  • Experiment Tracker (JSONL): All iterations logged                   │
│  • Market Data APIs: yfinance, news APIs, sentiment APIs               │
│  • Backtesting Engine: Backtrader with walk-forward analysis           │
│  • Code Tools: Generator, validator, syntax checker                    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Agent Architecture

**Total Agents**: 24
- **Primary Agents**: 5 (Research Swarm, Strategy Dev, Quality Gate, Failure Analysis, Trajectory Analyzer)
- **Coordinator Agents**: 1 (Research Leader)
- **Research Subagents**: 15 (5 technical + 5 fundamental + 5 sentiment)
- **Synthesizer Agents**: 3 (technical + fundamental + sentiment)

**See**: `AGENT_CATALOG.md` for complete agent specifications

---

## LangGraph Workflow

### State Schema

```python
from typing import TypedDict, List, Dict, Optional

class WorkflowState(TypedDict):
    # User input
    ticker: str
    research_directive: str
    quality_criteria: Dict[str, float]
    timeframe: str
    
    # Agent outputs
    research_findings: List[FactSheet]
    strategy_variants: List[StrategyVariant]
    backtest_results: List[BacktestResult]
    
    # Iteration tracking
    strategy_iteration: int
    research_iteration: int
    total_iterations: int
    
    # Decision tracking
    next_action: str
    failure_analysis: Optional[Dict]
    trajectory_analysis: Optional[Dict]
    
    # Results
    best_strategy: Optional[StrategyVariant]
    final_status: str  # "SUCCESS" | "ABANDONED"
    
    # Experiment tracking
    experiment_id: str
    experiment_history: List[ExperimentRecord]
```

### Workflow Definition

```python
from langgraph.graph import StateGraph, END

# Create graph
workflow = StateGraph(WorkflowState)

# Add nodes (agents)
workflow.add_node("research_swarm", research_swarm_agent.run)
workflow.add_node("strategy_dev", strategy_dev_agent.run)
workflow.add_node("parallel_backtest", parallel_backtest_node)
workflow.add_node("quality_gate", quality_gate_agent.run)

# Add linear edges
workflow.set_entry_point("research_swarm")
workflow.add_edge("research_swarm", "strategy_dev")
workflow.add_edge("strategy_dev", "parallel_backtest")
workflow.add_edge("parallel_backtest", "quality_gate")

# Add conditional routing (three-tier feedback loops)
workflow.add_conditional_edges(
    "quality_gate",
    route_after_quality_gate,
    {
        "success": END,
        "tune": "strategy_dev",
        "fix": "strategy_dev",
        "refine": "strategy_dev",
        "research": "research_swarm",
        "abandon": END
    }
)

# Compile
app = workflow.compile()
```

**See**: `LANGGRAPH_IMPLEMENTATION.md` for complete implementation

---

## Core Workflow Phases

### Phase 1: Research Swarm (Hierarchical Synthesis)

**Agent**: Research Swarm Agent (19 agents total)

**Architecture**: 3-tier hierarchical synthesis
- **Tier 1**: 15 Research Subagents (parallel execution)
- **Tier 2**: 3 Domain Synthesizers (parallel execution)
- **Tier 3**: Research Leader Agent (final synthesis)

**Why Hierarchical?**
- Prevents cognitive overload (Leader processes 3 fact sheets vs. 30 findings)
- Scales to 15-20 subagents (3x improvement over flat architecture)
- Reduces context window usage by 50%

**Output**: 3 Fact Sheets (technical, fundamental, sentiment)

**See**: `HIERARCHICAL_SYNTHESIS.md` for detailed architecture

---

### Phase 2: Strategy Development

**Agent**: Strategy Development Agent

**Input**: 3 fact sheets + iteration history + failure analysis

**Output**: N strategy variants (default: 5)

**Variant Generation**:
- **First iteration**: Diverse approaches (momentum, mean reversion, breakout)
- **Subsequent iterations**: Based on failure analysis recommendations
  - TUNE → Parameter variations
  - FIX → Bug fixes
  - REFINE → Design improvements

**Code Validation**: 4-stage pipeline
1. Syntax check (Python AST)
2. Logic check (Backtrader compatibility)
3. Security check (no unsafe operations)
4. Performance check (estimated complexity)

---

### Phase 3: Parallel Backtesting

**Type**: LangGraph Map-Reduce Node (NOT an agent)

**Implementation**:
```python
async def parallel_backtest_node(state: WorkflowState):
    strategy_variants = state["strategy_variants"]
    
    # Execute all backtests in parallel
    results = await asyncio.gather(*[
        execute_backtest(variant, state["ticker"], state["timeframe"])
        for variant in strategy_variants
    ])
    
    return {"backtest_results": results}
```

**Benefits**:
- N variants tested in same time as 1 variant
- 3-10x speedup depending on number of variants
- Resource-aware execution

---

### Phase 4: Quality Gate Validation (Intelligence Stack)

**Agent**: Quality Gate Agent

**Sub-Agents**:
- **Failure Analysis Agent**: Diagnose why strategies failed
- **Trajectory Analyzer Agent**: Analyze improvement trajectory

**Evaluation Process**:
1. Check if any variant passed → SUCCESS
2. If all failed:
   a. Invoke Failure Analysis Agent
   b. Invoke Trajectory Analyzer Agent (if history ≥ 2)
   c. Combine analyses
   d. Make routing decision (three-tier feedback loops)

**Three-Tier Feedback Loops**:
- **Tier 1** (Strategy Refinement): TUNE | FIX | REFINE → strategy_dev
- **Tier 2** (Research Refinement): RESEARCH → research_swarm
- **Tier 3** (Abandonment): ABANDON → END

**See**: `FEEDBACK_LOOPS.md` for detailed routing logic

---

## Intelligence Stack

### 1. Failure Analysis Agent

**Role**: Diagnose why strategies failed and classify failures

**Classifications**:
1. **PARAMETER_ISSUE**: Logic sound, parameters need tuning → TUNE
2. **ALGORITHM_BUG**: Implementation error → FIX
3. **DESIGN_FLAW**: Missing features → REFINE
4. **RESEARCH_GAP**: Insufficient research → RESEARCH
5. **FUNDAMENTAL_IMPOSSIBILITY**: No alpha exists → ABANDON

**Analysis Process**:
- Code analysis (detect bugs)
- Statistical analysis (distance from threshold)
- LLM reasoning (interpret patterns)

**See**: `FAILURE_ANALYSIS_SYSTEM.md` for detailed classification criteria

---

### 2. Trajectory Analyzer Agent

**Role**: Analyze experiment trajectory to detect convergence/divergence

**Trajectory Statuses**:
1. **CONVERGING**: Metrics improving consistently → CONTINUE
2. **DIVERGING**: Metrics getting worse → ABANDON
3. **OSCILLATING**: Metrics fluctuating → PIVOT or ABANDON
4. **STAGNANT**: Metrics not changing → PIVOT

**Analysis Process**:
- Load experiment history from JSONL
- Compute improvement rates per metric
- Detect convergence patterns
- LLM interpretation

**See**: `EXPERIMENT_TRACKING.md` for detailed trajectory analysis

---

## Memory Architecture

### ChromaDB Collections

| Collection | Purpose | Key Fields |
|------------|---------|------------|
| research_findings | Store research outputs | ticker, type, confidence, agent_id |
| strategy_library | Successful strategies | name, code, metrics, performance |
| lessons_learned | Failed attempts | strategy_id, failure_reason, improvement |
| market_regimes | Market conditions | regime_type, indicators, date_range |

**Features**:
- Semantic search with embeddings
- Metadata filtering
- Lineage tracking (parent-child relationships)
- Automatic archiving

**See**: `MEMORY_ARCHITECTURE.md` for detailed schema

---

## Experiment Tracking

### JSONL Format

**One JSON record per line** (append-only):
```jsonl
{"experiment_id":"exp_001","iteration":1,"sharpe_ratio":0.75,"action":"TUNE"}
{"experiment_id":"exp_001","iteration":2,"sharpe_ratio":0.85,"action":"TUNE"}
{"experiment_id":"exp_001","iteration":3,"sharpe_ratio":1.05,"action":"SUCCESS"}
```

**Benefits**:
- Streaming format (no need to load entire file)
- Easy to parse and analyze
- Supports trajectory analysis
- Audit trail for debugging

**See**: `EXPERIMENT_TRACKING.md` for complete schema

---

## Error Handling

### LLM-Powered Failure Analysis

**NOT rule-based error handling**. Instead:
- Failure Analysis Agent uses LLM reasoning
- Analyzes complete context (code, metrics, history)
- Provides specific actionable recommendations
- Classifies failures into 5 categories

### Recovery Strategies

| Error Type | Strategy | Details |
|------------|----------|---------|
| API Rate Limit | Exponential backoff + failover | Use LangChain's `with_fallbacks()` |
| Subagent Failure | Partial results | Proceed if >50% succeed |
| Code Generation Error | Retry with feedback | Include error in next prompt |
| Backtest Error | Skip and log | Move to next variant |
| Context Overflow | Hierarchical synthesis | Use domain synthesizers |

---

## Complete Execution Flow

### Initialization

```python
# User input
user_input = {
    "ticker": "AAPL",
    "research_directive": "Find momentum alpha in tech stocks",
    "quality_criteria": {
        "sharpe_ratio": 1.0,
        "max_drawdown": 0.20,
        "win_rate": 0.50
    },
    "timeframe": "1d"
}

# Run workflow
app = workflow.compile()
result = app.invoke(user_input)
```

### Iteration 1

```
Research Swarm Agent
├─ Spawn 15 subagents (parallel)
├─ Spawn 3 synthesizers (parallel)
└─ Return: 3 fact sheets

Strategy Development Agent
├─ Generate 5 momentum strategy variants
└─ Return: 5 variants

Parallel Backtest Node
├─ Execute 5 backtests (parallel, 45 seconds)
└─ Return: 5 results

Quality Gate Agent
├─ All failed (best Sharpe: 0.85)
├─ Invoke Failure Analysis: "PARAMETER_ISSUE"
└─ Decision: TUNE

LangGraph Routing
└─ Route to strategy_dev (Tier 1)
```

### Iteration 2

```
Strategy Development Agent
├─ Generate 5 parameter variations
└─ Return: 5 tuned variants

Parallel Backtest Node
├─ Execute 5 backtests (parallel)
└─ Return: 5 results

Quality Gate Agent
├─ 1 variant passed! (Sharpe: 1.12)
└─ Decision: SUCCESS

LangGraph Routing
└─ Route to END
```

### Result

```python
{
    "status": "SUCCESS",
    "best_strategy": {
        "name": "Momentum Strategy (12/60 MA)",
        "code": "class MomentumStrategy(bt.Strategy): ...",
        "metrics": {
            "sharpe_ratio": 1.12,
            "max_drawdown": 0.15,
            "total_return": 0.35
        }
    },
    "iterations": 2,
    "total_time": "90 seconds"
}
```

---

## Key Design Decisions

| ID | Decision | Rationale |
|----|----------|-----------|
| D-011 | Hierarchical Synthesis | Prevents cognitive overload, scales to 15-20 subagents |
| D-013 | Algorithm-Owned Regime Awareness | Quality gates stay objective, algorithms handle regimes |
| D-016 | Use LangChain's `with_fallbacks()` | Don't reinvent the wheel |
| D-019 | Three-Tier Feedback Loops | Intelligent routing based on failure type |
| D-020 | LLM-Powered Failure Analysis | Deep reasoning for failure classification |
| D-021 | Experiment Tracking System | Trajectory analysis enables convergence detection |
| D-022 | Queue-and-Worker Pattern | Simple, proven pattern for parallel execution |
| D-023 | LangGraph as Orchestrator | Use LangGraph's built-in features (not custom) |

**See**: `DECISION_LOG.md` for complete decision history

---

## Implementation Phases

| Phase | Focus | Agents | Status |
|-------|-------|--------|--------|
| Phase 1 | Core Infrastructure | 0 | ✅ Complete |
| Phase 2 | Memory System | 0 | 🔄 Ready |
| Phase 3 | Tool Registry | 0 | ⏳ Pending |
| Phase 4 | Tool Meta-System | 0 | ⏳ Pending |
| Phase 5 | Research Swarm | 19 | ⏳ Pending |
| Phase 6 | Strategy Development | 1 | ⏳ Pending |
| Phase 7 | Backtesting | 0 | ⏳ Pending |
| Phase 8 | Quality Gates | 3 | ⏳ Pending |
| Phase 9 | Workflow Pipeline | 0 | ⏳ Pending |
| Phase 10 | Testing & Documentation | 0 | ⏳ Pending |

**Total**: 10 phases, 37.5 days estimated

---

## Summary

**Architecture**: Multi-agent system orchestrated by LangGraph  
**Total Agents**: 24 (5 primary + 1 coordinator + 18 subagents)  
**Orchestration**: LangGraph StateGraph (not custom orchestrator)  
**Key Innovation**: Hierarchical synthesis + LLM-powered feedback loops  
**Parallel Execution**: LangGraph map-reduce for backtests  
**Memory**: ChromaDB for persistent learning  
**Tracking**: JSONL experiment logs for trajectory analysis  

**Next Steps**: Implement Phase 2 (Memory System)

---

**End of System Design**
