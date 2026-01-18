# Agentic System Architecture Diagram

## LangChain Algorithmic Trading Development System

This diagram shows the complete agentic workflow orchestrated by LangGraph.

```mermaid
graph TB
    Start([User: Develop Strategy<br/>for TICKER]) --> ResearchSwarm{Research Swarm<br/>Coordinator}

    %% Research Swarm: Tier 1 - Parallel Subagents (15 agents)
    ResearchSwarm --> TechParallel{Technical<br/>Subagents}
    ResearchSwarm --> FundParallel{Fundamental<br/>Subagents}
    ResearchSwarm --> SentParallel{Sentiment<br/>Subagents}

    %% Technical Subagents (5)
    TechParallel --> TechAgent1["Price Action<br/>Analyst"]
    TechParallel --> TechAgent2["Volume Profile<br/>Analyst"]
    TechParallel --> TechAgent3["Momentum<br/>Analyst"]
    TechParallel --> TechAgent4["Volatility<br/>Analyst"]
    TechParallel --> TechAgent5["Pattern<br/>Recognition"]

    %% Fundamental Subagents (5)
    FundParallel --> FundAgent1["Earnings<br/>Analyst"]
    FundParallel --> FundAgent2["Balance Sheet<br/>Analyst"]
    FundParallel --> FundAgent3["Cash Flow<br/>Analyst"]
    FundParallel --> FundAgent4["Valuation<br/>Analyst"]
    FundParallel --> FundAgent5["Growth<br/>Analyst"]

    %% Sentiment Subagents (5)
    SentParallel --> SentAgent1["News<br/>Analyst"]
    SentParallel --> SentAgent2["Social Media<br/>Analyst"]
    SentParallel --> SentAgent3["Analyst Ratings<br/>Monitor"]
    SentParallel --> SentAgent4["Insider Trading<br/>Monitor"]
    SentParallel --> SentAgent5["Options Flow<br/>Analyst"]

    %% Tier 2 - Domain Synthesizers (3 agents)
    TechAgent1 --> TechSync["Technical<br/>Sync"]
    TechAgent2 --> TechSync
    TechAgent3 --> TechSync
    TechAgent4 --> TechSync
    TechAgent5 --> TechSync
    TechSync --> TechSynthesizer["Technical<br/>Synthesizer<br/>(Fact Sheet)"]

    FundAgent1 --> FundSync["Fundamental<br/>Sync"]
    FundAgent2 --> FundSync
    FundAgent3 --> FundSync
    FundAgent4 --> FundSync
    FundAgent5 --> FundSync
    FundSync --> FundSynthesizer["Fundamental<br/>Synthesizer<br/>(Fact Sheet)"]

    SentAgent1 --> SentSync["Sentiment<br/>Sync"]
    SentAgent2 --> SentSync
    SentAgent3 --> SentSync
    SentAgent4 --> SentSync
    SentAgent5 --> SentSync
    SentSync --> SentSynthesizer["Sentiment<br/>Synthesizer<br/>(Fact Sheet)"]

    %% Tier 3 - Research Leader (1 agent)
    TechSynthesizer --> ResearchLeader["Research Leader<br/>(Cross-Domain<br/>Synthesis)"]
    FundSynthesizer --> ResearchLeader
    SentSynthesizer --> ResearchLeader

    ResearchLeader --> ResearchReport["Research Report<br/>(Alpha Hypothesis)"]

    %% Strategy Development Phase
    ResearchReport --> StrategyDev["Strategy Development<br/>Agent<br/>(Code Generation)"]
    StrategyDev --> StrategyVariants["Strategy Variants<br/>(3-5 variants)"]

    %% Parallel Backtesting Phase
    StrategyVariants --> BacktestQueue{Backtest<br/>Task Queue}
    BacktestQueue --> Worker1["Worker 1<br/>(Backtest)"]
    BacktestQueue --> Worker2["Worker 2<br/>(Backtest)"]
    BacktestQueue --> Worker3["Worker 3<br/>(Backtest)"]
    BacktestQueue --> Worker4["Worker 4<br/>(Backtest)"]
    BacktestQueue --> Worker5["Worker 5<br/>(Backtest)"]

    Worker1 --> BacktestSync["Backtest<br/>Results Sync"]
    Worker2 --> BacktestSync
    Worker3 --> BacktestSync
    Worker4 --> BacktestSync
    Worker5 --> BacktestSync

    %% Quality Gate Phase
    BacktestSync --> QualityGate["Quality Gate<br/>Agent<br/>(Evaluation)"]

    QualityGate -->|All Failed| FailureAnalysis["Failure Analysis<br/>Agent<br/>(Diagnosis)"]
    QualityGate -->|Any Passed| Success([SUCCESS<br/>Deploy Strategy])

    FailureAnalysis --> TrajectoryCheck{History<br/>>= 2?}
    TrajectoryCheck -->|Yes| TrajectoryAnalyzer["Trajectory Analyzer<br/>Agent<br/>(Convergence)"]
    TrajectoryCheck -->|No| RoutingDecision

    TrajectoryAnalyzer --> RoutingDecision{Routing<br/>Decision}

    %% Three-Tier Feedback Loops
    RoutingDecision -->|Tier 1:<br/>TUNE_PARAMETERS| StrategyDev
    RoutingDecision -->|Tier 1:<br/>FIX_BUG| StrategyDev
    RoutingDecision -->|Tier 1:<br/>REFINE_ALGORITHM| StrategyDev
    RoutingDecision -->|Tier 2:<br/>REFINE_RESEARCH| ResearchSwarm
    RoutingDecision -->|Tier 3:<br/>ABANDON| Abandon([ABANDONED<br/>No Alpha Found])

    %% Styling
    style Start fill:#55efc4,color:#333,stroke:#333,stroke-width:3px
    style Success fill:#55efc4,color:#333,stroke:#333,stroke-width:3px
    style Abandon fill:#ff7675,color:#333,stroke:#333,stroke-width:3px

    style ResearchSwarm fill:#ffeaa7,color:#333,stroke:#333,stroke-width:2px
    style TechParallel fill:#ffeaa7,color:#333
    style FundParallel fill:#ffeaa7,color:#333
    style SentParallel fill:#ffeaa7,color:#333
    style BacktestQueue fill:#ffeaa7,color:#333
    style RoutingDecision fill:#ffeaa7,color:#333
    style TrajectoryCheck fill:#e0e0e0,color:#333

    style TechAgent1 fill:#e1f5ff,color:#333
    style TechAgent2 fill:#e1f5ff,color:#333
    style TechAgent3 fill:#e1f5ff,color:#333
    style TechAgent4 fill:#e1f5ff,color:#333
    style TechAgent5 fill:#e1f5ff,color:#333

    style FundAgent1 fill:#e1ffe1,color:#333
    style FundAgent2 fill:#e1ffe1,color:#333
    style FundAgent3 fill:#e1ffe1,color:#333
    style FundAgent4 fill:#e1ffe1,color:#333
    style FundAgent5 fill:#e1ffe1,color:#333

    style SentAgent1 fill:#fff0f5,color:#333
    style SentAgent2 fill:#fff0f5,color:#333
    style SentAgent3 fill:#fff0f5,color:#333
    style SentAgent4 fill:#fff0f5,color:#333
    style SentAgent5 fill:#fff0f5,color:#333

    style TechSync fill:#e0e0e0,color:#333
    style FundSync fill:#e0e0e0,color:#333
    style SentSync fill:#e0e0e0,color:#333
    style BacktestSync fill:#e0e0e0,color:#333

    style TechSynthesizer fill:#d4edda,color:#333
    style FundSynthesizer fill:#d4edda,color:#333
    style SentSynthesizer fill:#d4edda,color:#333

    style ResearchLeader fill:#fff4e1,color:#333,stroke:#333,stroke-width:2px
    style ResearchReport fill:#e6f3ff,color:#333

    style StrategyDev fill:#e8daff,color:#333,stroke:#333,stroke-width:2px
    style StrategyVariants fill:#e6f3ff,color:#333

    style Worker1 fill:#ffe4e1,color:#333
    style Worker2 fill:#ffe4e1,color:#333
    style Worker3 fill:#ffe4e1,color:#333
    style Worker4 fill:#ffe4e1,color:#333
    style Worker5 fill:#ffe4e1,color:#333

    style QualityGate fill:#d1ecf1,color:#333,stroke:#333,stroke-width:2px
    style FailureAnalysis fill:#fff3cd,color:#333
    style TrajectoryAnalyzer fill:#fff3cd,color:#333
```

## Legend

### Node Types

| Color | Type | Examples |
|-------|------|----------|
| 🟢 Green | Entry/Exit Points | Start, Success, Abandon |
| 🟡 Yellow | Coordinators/Dispatchers | Research Swarm, Routing Decision |
| 🔵 Blue | Technical Agents | Price Action, Volume Profile, Momentum |
| 🟢 Light Green | Fundamental Agents | Earnings, Balance Sheet, Cash Flow |
| 🌸 Pink | Sentiment Agents | News, Social Media, Analyst Ratings |
| 🟤 Gray | Sync Points | Technical Sync, Backtest Sync |
| 🟢 Light Green | Synthesizers | Technical/Fundamental/Sentiment Synthesizers |
| 🟡 Light Yellow | Primary Agents | Research Leader, Strategy Dev, Quality Gate |
| 🟠 Orange | Workers | Backtest Workers 1-5 |
| 🔵 Light Blue | Intermediate Outputs | Research Report, Strategy Variants |
| 🟡 Light Yellow | Sub-Agents | Failure Analysis, Trajectory Analyzer |

### Workflow Phases

1. **Research Swarm Phase** (19 agents)
   - **Tier 1**: 15 subagents run in parallel (5 technical + 5 fundamental + 5 sentiment)
   - **Tier 2**: 3 domain synthesizers produce Fact Sheets
   - **Tier 3**: 1 research leader performs cross-domain synthesis

2. **Strategy Development Phase** (1 agent)
   - Generates 3-5 strategy variants based on research findings
   - Each variant has different parameters or approaches

3. **Parallel Backtesting Phase** (Queue-and-Worker)
   - Task queue holds all backtest tasks
   - 5 workers execute backtests in parallel
   - Results synchronized after all complete

4. **Quality Gate Phase** (3 agents)
   - Quality Gate Agent evaluates all variants
   - If all fail → Failure Analysis Agent diagnoses root cause
   - If history >= 2 → Trajectory Analyzer Agent checks convergence
   - Routing Decision determines next action

### Three-Tier Feedback Loops

| Tier | Trigger | Action | Destination |
|------|---------|--------|-------------|
| **Tier 1** | Fixable issues | TUNE_PARAMETERS, FIX_BUG, REFINE_ALGORITHM | Strategy Development |
| **Tier 2** | Wrong hypothesis | REFINE_RESEARCH | Research Swarm |
| **Tier 3** | No alpha exists | ABANDON | End workflow |

### Agent Count Summary

| Category | Count | Details |
|----------|-------|---------|
| **Research Subagents** | 15 | 5 technical + 5 fundamental + 5 sentiment |
| **Domain Synthesizers** | 3 | Technical, Fundamental, Sentiment |
| **Research Leader** | 1 | Cross-domain synthesis |
| **Strategy Development** | 1 | Code generation |
| **Quality Gate** | 1 | Evaluation |
| **Failure Analysis** | 1 | Diagnosis |
| **Trajectory Analyzer** | 1 | Convergence analysis |
| **Total Agents** | 23 | (Workers are not agents) |

### Parallel Execution Points

1. **Research Swarm Tier 1**: 15 subagents run in parallel
2. **Backtesting**: 5 workers execute backtests in parallel
3. **Domain Synthesizers**: 3 synthesizers can run in parallel (optional optimization)

### Key Design Decisions

- **LangGraph orchestrates the workflow** (not custom orchestrator)
- **Hierarchical synthesis** reduces cognitive load on Research Leader
- **Queue-and-worker pattern** enables scalable parallel backtesting
- **Three-tier feedback loops** prevent endless iteration
- **LLM-powered failure analysis** provides intelligent routing decisions
- **Experiment tracking** enables trajectory analysis and convergence detection

---

**Document**: Agentic System Architecture Diagram  
**Created**: 2026-01-18  
**Status**: Complete
