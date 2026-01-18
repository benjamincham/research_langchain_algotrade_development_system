# LangGraph Workflow Guide

## Overview

This document provides a practical guide to understanding and working with the LangGraph workflow in the AlgoTrade Development System.

**Purpose**: Help developers understand how LangGraph orchestrates the multi-agent system without getting lost in implementation details.

**For implementation details**: See `LANGGRAPH_IMPLEMENTATION.md`

---

## What is LangGraph?

**LangGraph** is a framework for building stateful, multi-agent applications using LangChain. It provides:

- **StateGraph**: Define workflow as a directed graph
- **Nodes**: Represent agent invocations or operations
- **Edges**: Define flow between nodes
- **Conditional Edges**: Route based on state
- **State Management**: Automatic state passing between nodes
- **Streaming**: Real-time monitoring of workflow execution

**Key Insight**: LangGraph IS the orchestrator. We don't build a custom "Central Orchestrator" - we use LangGraph's built-in features.

---

## Our Workflow Graph

### Visual Representation

```
START
  │
  ▼
┌─────────────────┐
│ research_swarm  │ ← Node (invokes Research Swarm Agent)
└─────────────────┘
  │
  ▼
┌─────────────────┐
│  strategy_dev   │ ← Node (invokes Strategy Development Agent)
└─────────────────┘
  │
  ▼
┌──────────────────┐
│ parallel_backtest│ ← Node (async function with queue-and-worker)
└──────────────────┘
  │
  ▼
┌─────────────────┐
│  quality_gate   │ ← Node (invokes Quality Gate Agent)
└─────────────────┘
  │
  ▼
┌─────────────────────────────────────┐
│   Conditional Routing               │
│   (Three-Tier Feedback Loops)       │
├─────────────────────────────────────┤
│  • SUCCESS → END                    │
│  • TUNE/FIX/REFINE → strategy_dev   │
│  • RESEARCH → research_swarm        │
│  • ABANDON → END                    │
└─────────────────────────────────────┘
```

---

## Nodes Explained

### Node 1: research_swarm

**What it does**: Invokes Research Swarm Agent (19 agents total)

**Input from state**:
- `ticker`
- `research_directive`
- `research_iteration`

**Output to state**:
- `research_findings` (3 fact sheets)
- `research_iteration` (incremented)

**Duration**: ~30-60 seconds (parallel execution of 15 subagents)

---

### Node 2: strategy_dev

**What it does**: Invokes Strategy Development Agent

**Input from state**:
- `research_findings`
- `failure_analysis` (if iteration > 1)
- `strategy_iteration`

**Output to state**:
- `strategy_variants` (N variants, default 5)
- `strategy_iteration` (incremented)

**Duration**: ~20-40 seconds (LLM code generation)

---

### Node 3: parallel_backtest

**What it does**: Executes backtests using queue-and-worker pattern

**Implementation**: Async function (NOT an agent) that uses:
- Task queue for backtest jobs
- Worker pool with resource checking
- Automatic retry for failed tasks

**Input from state**:
- `strategy_variants`
- `ticker`
- `timeframe`

**Output to state**:
- `backtest_results` (N results)
- `best_strategy`
- `best_metrics`

**Duration**: ~30-60 seconds (parallel execution)

**Queue-and-Worker Pattern**:
```python
async def parallel_backtest_node(state: WorkflowState):
    # Create task queue
    task_queue = TaskQueue()
    
    # Enqueue all backtest tasks
    for variant in state["strategy_variants"]:
        task = BacktestTask(
            variant_id=variant["variant_id"],
            strategy_code=variant["code"],
            ticker=state["ticker"]
        )
        task_queue.enqueue(task)
    
    # Create worker pool (resource-aware)
    worker_pool = WorkerPool(
        max_workers=5,
        task_queue=task_queue
    )
    
    # Execute all tasks
    results = await worker_pool.execute_all()
    
    return {"backtest_results": results}
```

**Benefits**:
- ✅ Resource-aware (checks CPU/memory before starting workers)
- ✅ Automatic retry (failed tasks go back to queue)
- ✅ Simple and robust pattern
- ✅ Scales to N variants

---

### Node 4: quality_gate

**What it does**: Invokes Quality Gate Agent (with sub-agents)

**Input from state**:
- `backtest_results`
- `quality_criteria`
- `experiment_history`

**Sub-agents invoked**:
- Failure Analysis Agent (if all variants failed)
- Trajectory Analyzer Agent (if history >= 2)

**Output to state**:
- `next_action` (SUCCESS | TUNE | FIX | REFINE | RESEARCH | ABANDON)
- `failure_analysis`
- `trajectory_analysis`
- `total_iterations` (incremented)

**Duration**: ~10-20 seconds (LLM analysis)

---

## State Flow

### How State is Passed

LangGraph automatically passes state between nodes. Each node:
1. Receives the complete `WorkflowState`
2. Returns a **partial update** (only changed fields)
3. LangGraph merges the update into state
4. Next node receives updated state

**Example**:
```python
# research_swarm node returns
{
    "research_findings": [...],  # New data
    "research_iteration": 1       # Updated counter
}

# LangGraph merges this into state
# strategy_dev node receives complete state with updates
```

---

## Conditional Routing

### How Routing Works

After `quality_gate` node, LangGraph calls `route_after_quality_gate()` function:

```python
def route_after_quality_gate(state: WorkflowState) -> str:
    """Routing function for conditional edges"""
    action = state["next_action"]  # Set by Quality Gate Agent
    return action.lower()  # "success", "tune", "research", etc.
```

LangGraph then routes to the appropriate node based on the return value:

| Return Value | Next Node | Tier |
|--------------|-----------|------|
| `"success"` | END | - |
| `"tune"` | strategy_dev | Tier 1 |
| `"fix"` | strategy_dev | Tier 1 |
| `"refine"` | strategy_dev | Tier 1 |
| `"research"` | research_swarm | Tier 2 |
| `"abandon"` | END | Tier 3 |

---

## Three-Tier Feedback Loops

### Tier 1: Strategy Refinement

**When**: Failure is fixable through parameter tuning, bug fixes, or design improvements

**Route**: `quality_gate` → `strategy_dev` → `parallel_backtest` → `quality_gate`

**Actions**: TUNE | FIX | REFINE

**Max Iterations**: 5 (configurable via `max_strategy_iterations`)

**Example**:
```
Iteration 1: Sharpe 0.75 → TUNE
Iteration 2: Sharpe 0.85 → TUNE
Iteration 3: Sharpe 1.05 → SUCCESS ✅
```

---

### Tier 2: Research Refinement

**When**: Failure indicates research gap or wrong hypothesis

**Route**: `quality_gate` → `research_swarm` → `strategy_dev` → `parallel_backtest` → `quality_gate`

**Actions**: RESEARCH

**Max Iterations**: 3 (configurable via `max_research_iterations`)

**Example**:
```
Research 1: Momentum strategies → All fail (negative Sharpe)
Research 2: Mean reversion strategies → Some pass ✅
```

---

### Tier 3: Abandonment

**When**: Fundamental impossibility or iteration limits exceeded

**Route**: `quality_gate` → END

**Actions**: ABANDON

**Triggers**:
- Failure Analysis classifies as "FUNDAMENTAL_IMPOSSIBILITY"
- Total iterations >= `max_total_iterations` (default: 15)
- Trajectory Analysis detects "DIVERGING" pattern

---

## Parallel Execution with Queue-and-Worker

### How It Works

Instead of using LangGraph's map-reduce (which doesn't exist), we use a **queue-and-worker pattern** within the `parallel_backtest` node:

**Components**:
1. **TaskQueue**: Thread-safe queue for backtest tasks
2. **WorkerPool**: Pool of workers that pick up tasks
3. **Resource Checker**: Ensures CPU/memory available before starting workers

**Flow**:
```
1. Enqueue all backtest tasks (5 variants → 5 tasks)
2. Workers pick up tasks from queue
3. If resources available → execute backtest
4. If task fails → requeue (with retry limit)
5. When all tasks complete → return results
```

**Benefits**:
- ✅ Simple and proven pattern
- ✅ Resource-aware (doesn't overwhelm system)
- ✅ Automatic retry for transient failures
- ✅ Scales to N variants

---

## Monitoring and Debugging

### Streaming Events

```python
# Stream events for real-time monitoring
async for event in app.astream(user_input):
    print(f"Node: {event['node']}")
    print(f"State: {event['state']}")
```

### Visualizing the Graph

```python
from IPython.display import Image, display

# Visualize workflow graph
display(Image(app.get_graph().draw_mermaid_png()))
```

### Debugging State

```python
# Add breakpoints in node functions
async def research_swarm_node(state: WorkflowState):
    print(f"DEBUG: Research iteration {state['research_iteration']}")
    print(f"DEBUG: Ticker {state['ticker']}")
    
    # ... rest of implementation
```

---

## Configuration

### Iteration Limits

Control how many times each tier can iterate:

```python
user_input = {
    # ... other fields ...
    "max_strategy_iterations": 5,   # Tier 1 limit
    "max_research_iterations": 3,   # Tier 2 limit
    "max_total_iterations": 15      # Overall limit
}
```

### Quality Criteria

Define what "passing" means:

```python
user_input = {
    # ... other fields ...
    "quality_criteria": {
        "sharpe_ratio": 1.0,        # Minimum Sharpe ratio
        "max_drawdown": 0.20,       # Maximum drawdown (20%)
        "win_rate": 0.50,           # Minimum win rate (50%)
        "total_return": 0.15        # Minimum total return (15%)
    }
}
```

### Worker Pool Configuration

Configure parallel execution:

```python
user_input = {
    # ... other fields ...
    "max_parallel_workers": 5,      # Max concurrent backtests
    "cpu_threshold": 0.8,           # Don't start if CPU > 80%
    "memory_threshold": 0.8         # Don't start if memory > 80%
}
```

---

## Common Patterns

### Pattern 1: Linear Flow

```
research_swarm → strategy_dev → parallel_backtest → quality_gate
```

**When**: First iteration, no feedback loops yet

---

### Pattern 2: Tier 1 Loop (Parameter Tuning)

```
quality_gate → strategy_dev → parallel_backtest → quality_gate
```

**When**: Strategies close to passing, need parameter adjustments

**Iterations**: 2-5 typically

---

### Pattern 3: Tier 2 Loop (Research Pivot)

```
quality_gate → research_swarm → strategy_dev → parallel_backtest → quality_gate
```

**When**: Strategies fundamentally flawed, need new research direction

**Iterations**: 1-3 typically

---

### Pattern 4: Success

```
quality_gate → END
```

**When**: At least one variant passed all quality gates

---

### Pattern 5: Abandonment

```
quality_gate → END
```

**When**: No viable alpha found after exhausting iterations

---

## Best Practices

### 1. Start with Loose Criteria

Don't set quality criteria too high initially:

```python
# ❌ Too strict for first run
"quality_criteria": {"sharpe_ratio": 2.0, "max_drawdown": 0.10}

# ✅ Reasonable for exploration
"quality_criteria": {"sharpe_ratio": 1.0, "max_drawdown": 0.20}
```

### 2. Monitor Iteration Counts

Track which tier is consuming iterations:

```python
print(f"Strategy iterations: {result['strategy_iteration']}")
print(f"Research iterations: {result['research_iteration']}")
print(f"Total iterations: {result['total_iterations']}")
```

### 3. Review Failure Analysis

Always check why strategies failed:

```python
if result["final_status"] == "ABANDONED":
    print(f"Classification: {result['failure_analysis']['classification']}")
    print(f"Reasoning: {result['failure_analysis']['reasoning']}")
```

### 4. Use Experiment Tracking

Review experiment history to understand trajectory:

```python
for record in result["experiment_history"]:
    print(f"Iteration {record['iteration']}: Sharpe {record['best_sharpe']}")
```

### 5. Configure Worker Pool

Adjust based on your hardware:

```python
# For powerful machine (16+ cores, 32+ GB RAM)
"max_parallel_workers": 10

# For modest machine (4-8 cores, 16 GB RAM)
"max_parallel_workers": 3
```

---

## Troubleshooting

### Issue: Workflow stuck in Tier 1 loop

**Symptom**: `strategy_iteration` keeps increasing, never passes

**Solution**: Check if quality criteria are too strict, or if Failure Analysis is misclassifying

---

### Issue: Workflow abandons too early

**Symptom**: Only 2-3 iterations before ABANDON

**Solution**: Increase `max_strategy_iterations` or review Trajectory Analyzer logic

---

### Issue: Research Swarm takes too long

**Symptom**: `research_swarm` node takes > 2 minutes

**Solution**: Reduce number of subagents or use faster LLM model

---

### Issue: Backtests fail with errors

**Symptom**: `backtest_results` contains exceptions or failed tasks

**Solution**: 
- Check strategy code validation
- Ensure Backtrader compatibility
- Review task queue logs for retry attempts

---

### Issue: Workers not starting

**Symptom**: Tasks queued but no workers picking them up

**Solution**:
- Check resource thresholds (CPU/memory might be too high)
- Verify worker pool is initialized correctly
- Check logs for resource checker output

---

## Summary

**LangGraph provides**:
- ✅ State management (automatic passing between nodes)
- ✅ Workflow orchestration (graph-based execution)
- ✅ Conditional routing (three-tier feedback loops)
- ✅ Streaming (real-time monitoring)

**We provide**:
- ✅ Agent implementations (Research Swarm, Strategy Dev, Quality Gate)
- ✅ State schema (WorkflowState TypedDict)
- ✅ Routing logic (route_after_quality_gate function)
- ✅ Parallel execution (queue-and-worker pattern in parallel_backtest node)

**Key Insights**:
1. **LangGraph IS the orchestrator** - Don't build custom orchestrators
2. **Queue-and-worker for parallel execution** - Simple, robust, resource-aware
3. **Three-tier feedback loops** - Intelligent routing based on failure type
4. **LLM-powered intelligence** - Failure Analysis + Trajectory Analyzer guide decisions

---

**End of LangGraph Workflow Guide**
