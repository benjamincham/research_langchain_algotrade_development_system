# Agentic System Design Review

## Problem Statement

The current design documentation is **confused and unclear** about the agentic architecture. There are three separate concepts that aren't properly integrated:

1. **High-Level Workflow** - Shows phases (Research → Strategy → Backtest → Quality Gate)
2. **Central Orchestrator Architecture** - Shows state management, task queue, worker pool
3. **Intelligence Stack** - Shows failure analysis and experiment tracking

**The confusion**: These are described separately without clearly explaining:
- What **agents** exist in the system?
- What does each **agent** do?
- How do **agents communicate**?
- How does **LangGraph orchestrate** the agents?
- How do the "Intelligence Stack" and "Central Orchestrator" fit into the agent architecture?

## Root Cause Analysis

### Issue 1: Missing Agent Definitions

The design talks about "phases" but doesn't clearly define the **agents** that execute those phases.

**Questions not answered**:
- Is there a "Research Swarm Agent"? Or multiple research agents?
- Is there a "Strategy Development Agent"?
- Is there a "Quality Gate Agent"?
- How many agents are there total?
- What's the agent hierarchy?

### Issue 2: Mixing Orchestration with Agents

The "Central Orchestrator" is described as managing state, queues, and workers - but it's unclear:
- Is the Central Orchestrator itself an agent?
- Or is it a LangGraph workflow that coordinates agents?
- How do agents interact with the orchestrator?

### Issue 3: Unclear Role of "Intelligence Stack"

The "Intelligence Stack" (Failure Analysis + Experiment Tracking) is described separately:
- Are these agents?
- Are these tools that agents use?
- Are these services that the orchestrator uses?

### Issue 4: No LangGraph Integration

The design doesn't explain:
- How LangGraph state management works
- How agents pass messages
- How conditional routing works
- How parallel execution works in LangGraph

---

## What a Proper Agentic System Design Should Have

### 1. Clear Agent Catalog

**Example**:
```
Agents in the System:
1. Supervisor Agent (coordinates workflow)
2. Research Leader Agent (manages research swarm)
3. Technical Research Agent (analyzes technical indicators)
4. Fundamental Research Agent (analyzes fundamentals)
5. Sentiment Research Agent (analyzes sentiment)
6. Domain Synthesizer Agents (3 agents: technical, fundamental, sentiment)
7. Strategy Developer Agent (generates strategy code)
8. Backtest Executor Agent (runs backtests)
9. Quality Gate Agent (evaluates strategies)
10. Failure Analysis Agent (diagnoses failures)
11. Trajectory Analyzer Agent (analyzes experiment trends)
```

### 2. Agent Responsibilities Matrix

**Example**:
| Agent | Input | Output | Tools Used |
|-------|-------|--------|------------|
| Supervisor | User directive | Workflow result | State management, routing |
| Research Leader | Research directive | Research findings | Subagent coordination |
| Technical Research Agent | Ticker, timeframe | Technical findings | yfinance, TA-Lib |
| Strategy Developer | Research findings | Strategy code | Code generation, validation |
| ... | ... | ... | ... |

### 3. LangGraph Workflow Definition

**Example**:
```python
from langgraph.graph import StateGraph, END

# Define state
class WorkflowState(TypedDict):
    research_directive: str
    research_findings: List[Dict]
    strategy_variants: List[Dict]
    backtest_results: List[Dict]
    quality_gate_decision: str
    # ...

# Create graph
workflow = StateGraph(WorkflowState)

# Add nodes (agents)
workflow.add_node("supervisor", supervisor_agent)
workflow.add_node("research_swarm", research_swarm_agent)
workflow.add_node("strategy_dev", strategy_dev_agent)
workflow.add_node("parallel_backtest", parallel_backtest_node)
workflow.add_node("quality_gate", quality_gate_agent)

# Add edges (routing)
workflow.add_edge("supervisor", "research_swarm")
workflow.add_edge("research_swarm", "strategy_dev")
workflow.add_edge("strategy_dev", "parallel_backtest")
workflow.add_edge("parallel_backtest", "quality_gate")

# Add conditional routing
workflow.add_conditional_edges(
    "quality_gate",
    route_after_quality_gate,  # Function that decides next step
    {
        "success": END,
        "refine_strategy": "strategy_dev",
        "refine_research": "research_swarm",
        "abandon": END
    }
)
```

### 4. Agent Communication Patterns

**Example**:
```
Research Leader Agent
├── Spawns 15 Research Subagents (parallel)
├── Collects findings from all subagents
├── Passes findings to 3 Domain Synthesizers (parallel)
├── Collects synthesized findings
└── Returns to Supervisor

Supervisor Agent
├── Receives synthesized findings
├── Passes to Strategy Developer Agent
└── Waits for strategy variants
```

### 5. Integration of "Intelligence" Components

**Example**:
```
Failure Analysis Agent
├── Type: Agent (not a service)
├── Invoked by: Quality Gate Agent
├── Input: Strategy code, metrics, history
├── Output: Failure classification, recommendations
├── Tools: LLM reasoning, code analysis

Experiment Tracker
├── Type: Tool (not an agent)
├── Used by: All agents
├── Purpose: Log experiments, compute trajectories
├── Storage: JSONL files
```

---

## Proposed Redesign: Clear Agentic Architecture

### Layer 1: LangGraph Orchestration (The "Supervisor")

```
┌─────────────────────────────────────────────────────────────────┐
│                    LANGGRAPH WORKFLOW                           │
│                  (Supervisor Orchestration)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  StateGraph manages:                                            │
│  • Workflow state (shared across all agents)                    │
│  • Agent invocation (which agent to call next)                  │
│  • Conditional routing (based on agent outputs)                 │
│  • Parallel execution (map-reduce for subagents)                │
│                                                                 │
│  Nodes (Agents):                                                │
│  1. research_swarm_node                                         │
│  2. strategy_development_node                                   │
│  3. parallel_backtest_node                                      │
│  4. quality_gate_node                                           │
│  5. failure_analysis_node                                       │
│  6. trajectory_analysis_node                                    │
│                                                                 │
│  Edges (Routing):                                               │
│  • Linear: research → strategy → backtest → quality_gate       │
│  • Conditional: quality_gate → {success, refine, research}      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key Point**: LangGraph IS the "Central Orchestrator". It's not a separate component.

### Layer 2: Agents (The "Workers")

```
┌─────────────────────────────────────────────────────────────────┐
│                    RESEARCH SWARM AGENT                         │
├─────────────────────────────────────────────────────────────────┤
│  Role: Coordinate research subagents and synthesizers           │
│                                                                 │
│  Sub-architecture (hierarchical):                               │
│  Research Leader Agent                                          │
│  ├── Technical Research Subagents (5 agents, parallel)         │
│  ├── Fundamental Research Subagents (5 agents, parallel)       │
│  ├── Sentiment Research Subagents (5 agents, parallel)         │
│  ├── Technical Domain Synthesizer Agent                        │
│  ├── Fundamental Domain Synthesizer Agent                      │
│  └── Sentiment Domain Synthesizer Agent                        │
│                                                                 │
│  Input: research_directive, ticker, timeframe                   │
│  Output: synthesized_findings (3 fact sheets)                   │
│  Tools: yfinance, TA-Lib, news APIs, memory (ChromaDB)         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                STRATEGY DEVELOPMENT AGENT                       │
├─────────────────────────────────────────────────────────────────┤
│  Role: Generate strategy code variants                          │
│                                                                 │
│  Input: synthesized_findings, iteration_history                 │
│  Output: strategy_variants (N variants with code + params)      │
│  Tools: Code generation, validation, memory (strategy library)  │
│                                                                 │
│  Generates N variants based on:                                 │
│  • Research findings                                            │
│  • Previous iteration failures                                  │
│  • Failure analysis recommendations                             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                PARALLEL BACKTEST NODE                           │
├─────────────────────────────────────────────────────────────────┤
│  Role: Execute backtests for all variants in parallel           │
│                                                                 │
│  NOT an agent - this is a LangGraph map-reduce node             │
│                                                                 │
│  Implementation:                                                │
│  • LangGraph's map() function spawns N parallel tasks           │
│  • Each task runs: backtest_executor_function(variant)          │
│  • Results aggregated and returned                              │
│                                                                 │
│  Input: strategy_variants (list of N variants)                  │
│  Output: backtest_results (list of N results)                   │
│  Tools: Backtrader, experiment tracker                          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    QUALITY GATE AGENT                           │
├─────────────────────────────────────────────────────────────────┤
│  Role: Evaluate all backtest results and decide next action     │
│                                                                 │
│  Sub-agents (invoked if needed):                                │
│  • Failure Analysis Agent                                       │
│  • Trajectory Analyzer Agent                                    │
│                                                                 │
│  Input: backtest_results, quality_criteria, iteration_history   │
│  Output: decision (SUCCESS | TUNE | FIX | REFINE | RESEARCH | ABANDON) │
│  Tools: Fuzzy logic scorer, experiment tracker                  │
│                                                                 │
│  Decision Logic:                                                │
│  1. Check if any variant passed → SUCCESS                       │
│  2. If all failed → invoke Failure Analysis Agent               │
│  3. Invoke Trajectory Analyzer Agent                            │
│  4. Combine analyses → decide next action                       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                FAILURE ANALYSIS AGENT                           │
├─────────────────────────────────────────────────────────────────┤
│  Role: Diagnose why strategies failed                           │
│                                                                 │
│  Input: strategy_code, metrics, quality_gate_results, history   │
│  Output: failure_classification, root_cause, recommendations    │
│  Tools: LLM reasoning, code analysis, statistical analysis      │
│                                                                 │
│  Classifications:                                               │
│  • PARAMETER_ISSUE → recommend TUNE                             │
│  • ALGORITHM_BUG → recommend FIX                                │
│  • DESIGN_FLAW → recommend REFINE                               │
│  • RESEARCH_GAP → recommend RESEARCH                            │
│  • FUNDAMENTAL_IMPOSSIBILITY → recommend ABANDON                │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                TRAJECTORY ANALYZER AGENT                        │
├─────────────────────────────────────────────────────────────────┤
│  Role: Analyze experiment trajectory and convergence            │
│                                                                 │
│  Input: experiment_history (all iterations)                     │
│  Output: trajectory_status, convergence_assessment, recommendation │
│  Tools: Statistical analysis, LLM interpretation                │
│                                                                 │
│  Assessments:                                                   │
│  • CONVERGING → recommend CONTINUE                              │
│  • DIVERGING → recommend PIVOT                                  │
│  • OSCILLATING → recommend ABANDON                              │
│  • STAGNANT → recommend PIVOT                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Layer 3: Tools (Shared Resources)

```
┌─────────────────────────────────────────────────────────────────┐
│                        SHARED TOOLS                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Memory System (ChromaDB)                                    │
│     • Research findings storage                                 │
│     • Strategy library                                          │
│     • Lessons learned                                           │
│     • Lineage tracking                                          │
│                                                                 │
│  2. Experiment Tracker (JSONL + Analysis)                       │
│     • ExperimentLogger (writes to JSONL)                        │
│     • TrajectoryAnalyzer (computes statistics)                  │
│     • Used by ALL agents to log their activities                │
│                                                                 │
│  3. Market Data APIs                                            │
│     • yfinance                                                  │
│     • News APIs                                                 │
│     • Alternative data sources                                  │
│                                                                 │
│  4. Backtesting Engine                                          │
│     • Backtrader                                                │
│     • Walk-forward analysis                                     │
│     • Monte Carlo simulation                                    │
│                                                                 │
│  5. Code Tools                                                  │
│     • Code generator                                            │
│     • Code validator                                            │
│     • Syntax checker                                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## How They Come Together: Complete Flow

### Initialization

```python
# 1. User provides input
user_input = {
    "ticker": "AAPL",
    "research_directive": "Find momentum alpha in tech stocks",
    "quality_criteria": {
        "sharpe_ratio": 1.0,
        "max_drawdown": 0.20,
        "win_rate": 0.50
    }
}

# 2. Initialize LangGraph workflow
from langgraph.graph import StateGraph

workflow = StateGraph(WorkflowState)

# 3. Add agent nodes
workflow.add_node("research_swarm", research_swarm_agent.run)
workflow.add_node("strategy_dev", strategy_dev_agent.run)
workflow.add_node("parallel_backtest", parallel_backtest_node)
workflow.add_node("quality_gate", quality_gate_agent.run)

# 4. Define routing
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

# 5. Compile and run
app = workflow.compile()
result = app.invoke(user_input)
```

### Execution Flow

```
Step 1: LangGraph invokes research_swarm_agent
├── Research Leader Agent spawns 15 subagents (parallel)
├── Subagents research AAPL (technical, fundamental, sentiment)
├── Domain Synthesizers create 3 fact sheets
├── Research Leader returns synthesized findings
└── LangGraph updates state.research_findings

Step 2: LangGraph invokes strategy_dev_agent
├── Strategy Developer reads research findings from state
├── Generates 5 strategy variants (different parameters)
├── Validates all variants (syntax, logic)
├── Returns strategy_variants
└── LangGraph updates state.strategy_variants

Step 3: LangGraph invokes parallel_backtest_node
├── LangGraph's map() spawns 5 parallel tasks
├── Each task runs backtest_executor(variant)
├── Backtest Executor uses Backtrader
├── Logs to Experiment Tracker
├── Returns metrics for each variant
├── LangGraph aggregates results
└── LangGraph updates state.backtest_results

Step 4: LangGraph invokes quality_gate_agent
├── Quality Gate Agent evaluates all 5 results
├── Checks if any passed quality criteria
├── If all failed:
│   ├── Invokes Failure Analysis Agent
│   ├── Invokes Trajectory Analyzer Agent
│   └── Combines analyses
├── Makes decision: TUNE | FIX | REFINE | RESEARCH | ABANDON
└── LangGraph updates state.next_action

Step 5: LangGraph routes based on decision
├── If SUCCESS → END (return best strategy)
├── If TUNE/FIX/REFINE → Go to strategy_dev (Tier 1)
├── If RESEARCH → Go to research_swarm (Tier 2)
└── If ABANDON → END (return failure)

Step 6: Iteration continues until success or abandon
```

---

## Key Clarifications

### 1. LangGraph IS the Central Orchestrator

**NOT**:
```
Central Orchestrator (custom class)
├── Manages state
├── Manages queue
├── Invokes agents
```

**YES**:
```
LangGraph StateGraph
├── Manages state (built-in)
├── Manages execution (built-in)
├── Invokes agents (nodes)
├── Routes conditionally (edges)
├── Handles parallel execution (map-reduce)
```

### 2. "Intelligence Stack" = Agents + Tools

**NOT**: A separate layer

**YES**:
- **Failure Analysis Agent**: An agent invoked by Quality Gate Agent
- **Trajectory Analyzer Agent**: An agent invoked by Quality Gate Agent
- **Experiment Tracker**: A tool used by all agents

### 3. Parallel Execution in LangGraph

**NOT**: Custom TaskQueue + WorkerPool

**YES**: LangGraph's built-in `map()` function

```python
from langgraph.graph import StateGraph

def parallel_backtest_node(state: WorkflowState):
    """
    LangGraph node that executes backtests in parallel.
    """
    strategy_variants = state["strategy_variants"]
    
    # LangGraph's map() handles parallelism
    results = await asyncio.gather(*[
        backtest_executor(variant)
        for variant in strategy_variants
    ])
    
    return {"backtest_results": results}
```

### 4. State Management

**NOT**: Custom StateManager class

**YES**: LangGraph's TypedDict state

```python
from typing import TypedDict, List, Dict

class WorkflowState(TypedDict):
    # User input
    ticker: str
    research_directive: str
    quality_criteria: Dict[str, float]
    
    # Agent outputs
    research_findings: List[Dict]
    strategy_variants: List[Dict]
    backtest_results: List[Dict]
    
    # Iteration tracking
    strategy_iteration: int
    research_iteration: int
    
    # Decision
    next_action: str
    failure_analysis: Dict
    trajectory_analysis: Dict
```

---

## Corrected Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         USER INPUT                                      │
│  • Ticker: AAPL                                                         │
│  • Research Directive: "Find momentum alpha"                            │
│  • Quality Criteria: {sharpe: 1.0, drawdown: 0.20}                      │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    LANGGRAPH WORKFLOW (Orchestrator)                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  Node: research_swarm                                             │  │
│  │  Agent: Research Swarm Agent                                      │  │
│  │  ├── 15 Research Subagents (parallel)                             │  │
│  │  ├── 3 Domain Synthesizers (parallel)                             │  │
│  │  └── Output: research_findings                                    │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                              │                                           │
│                              ▼                                           │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  Node: strategy_dev                                               │  │
│  │  Agent: Strategy Development Agent                                │  │
│  │  └── Output: strategy_variants (N variants)                       │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                              │                                           │
│                              ▼                                           │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  Node: parallel_backtest                                          │  │
│  │  Function: parallel_backtest_node (LangGraph map-reduce)          │  │
│  │  ├── Spawn N parallel backtest tasks                              │  │
│  │  └── Output: backtest_results (N results)                         │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                              │                                           │
│                              ▼                                           │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  Node: quality_gate                                               │  │
│  │  Agent: Quality Gate Agent                                        │  │
│  │  ├── Invokes: Failure Analysis Agent (if needed)                  │  │
│  │  ├── Invokes: Trajectory Analyzer Agent (if needed)               │  │
│  │  └── Output: next_action (SUCCESS | TUNE | RESEARCH | ABANDON)   │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                              │                                           │
│                              ▼                                           │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  Conditional Routing                                              │  │
│  │  ├── SUCCESS → END                                                │  │
│  │  ├── TUNE/FIX/REFINE → strategy_dev (Tier 1)                      │  │
│  │  ├── RESEARCH → research_swarm (Tier 2)                           │  │
│  │  └── ABANDON → END                                                │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         SHARED TOOLS                                    │
│  • Memory System (ChromaDB)                                             │
│  • Experiment Tracker (JSONL)                                           │
│  • Market Data APIs (yfinance)                                          │
│  • Backtesting Engine (Backtrader)                                      │
│  • Code Tools (generator, validator)                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Action Items

1. **Rewrite SYSTEM_DESIGN.md** with clear agent definitions
2. **Rewrite CENTRAL_ORCHESTRATOR.md** as LangGraph workflow design
3. **Create AGENT_CATALOG.md** listing all agents and their responsibilities
4. **Create LANGGRAPH_IMPLEMENTATION.md** with complete LangGraph code
5. **Update README.md** to reflect corrected architecture
6. **Remove confusion** between "Central Orchestrator" and LangGraph

---

**End of Review - Awaiting User Approval to Proceed with Redesign**
