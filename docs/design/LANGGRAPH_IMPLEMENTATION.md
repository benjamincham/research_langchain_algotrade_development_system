# LangGraph Implementation Guide

## Overview

This document provides the complete LangGraph implementation for the AlgoTrade Development System. It shows how to implement the multi-agent workflow using LangGraph's StateGraph, nodes, edges, and conditional routing.

**Prerequisites**: Read `AGENT_CATALOG.md` and `SYSTEM_DESIGN.md` first.

---

## State Schema

### WorkflowState (TypedDict)

```python
from typing import TypedDict, List, Dict, Optional, Literal
from datetime import datetime

class FactSheet(TypedDict):
    """Output from Domain Synthesizer"""
    domain: Literal["technical", "fundamental", "sentiment"]
    key_insights: List[str]
    confidence: float
    evidence: List[Dict]
    agent_ids: List[str]
    timestamp: str

class StrategyVariant(TypedDict):
    """Generated strategy code"""
    variant_id: str
    name: str
    code: str
    parameters: Dict
    description: str
    expected_behavior: str

class BacktestResult(TypedDict):
    """Backtest output"""
    variant_id: str
    metrics: Dict[str, float]  # sharpe_ratio, max_drawdown, total_return, etc.
    passed_gates: bool
    gate_scores: Dict[str, float]

class FailureAnalysis(TypedDict):
    """Output from Failure Analysis Agent"""
    classification: Literal["PARAMETER_ISSUE", "ALGORITHM_BUG", "DESIGN_FLAW", "RESEARCH_GAP", "FUNDAMENTAL_IMPOSSIBILITY"]
    root_cause: str
    specific_actions: List[str]
    confidence: float
    reasoning: str

class TrajectoryAnalysis(TypedDict):
    """Output from Trajectory Analyzer Agent"""
    status: Literal["CONVERGING", "DIVERGING", "OSCILLATING", "STAGNANT"]
    improvement_rate: float
    convergence_probability: float
    recommendation: str
    reasoning: str

class ExperimentRecord(TypedDict):
    """Single experiment iteration"""
    experiment_id: str
    iteration: int
    timestamp: str
    strategy_variants: List[StrategyVariant]
    backtest_results: List[BacktestResult]
    best_sharpe: float
    best_variant_id: str
    action: str
    failure_analysis: Optional[FailureAnalysis]
    trajectory_analysis: Optional[TrajectoryAnalysis]

class WorkflowState(TypedDict):
    """Complete workflow state"""
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
    max_strategy_iterations: int
    max_research_iterations: int
    max_total_iterations: int
    
    # Decision tracking
    next_action: Literal["SUCCESS", "TUNE", "FIX", "REFINE", "RESEARCH", "ABANDON"]
    failure_analysis: Optional[FailureAnalysis]
    trajectory_analysis: Optional[TrajectoryAnalysis]
    
    # Results
    best_strategy: Optional[StrategyVariant]
    best_metrics: Optional[Dict[str, float]]
    final_status: Literal["SUCCESS", "ABANDONED", "IN_PROGRESS"]
    
    # Experiment tracking
    experiment_id: str
    experiment_history: List[ExperimentRecord]
```

---

## Agent Implementations

### 1. Research Swarm Agent

```python
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import asyncio

class ResearchSwarmAgent:
    """Orchestrates 19 agents in hierarchical synthesis"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.subagents = self._create_subagents()
        self.synthesizers = self._create_synthesizers()
        self.leader = ResearchLeaderAgent(llm)
    
    def _create_subagents(self) -> List[ResearchSubagent]:
        """Create 15 research subagents"""
        subagents = []
        
        # 5 Technical Analysis Subagents
        for name in ["Price Action", "Volume Analysis", "Momentum Indicators", "Volatility Analysis", "Pattern Recognition"]:
            subagents.append(ResearchSubagent(
                name=name,
                domain="technical",
                llm=self.llm
            ))
        
        # 5 Fundamental Analysis Subagents
        for name in ["Financial Statements", "Valuation Metrics", "Growth Analysis", "Industry Comparison", "Risk Assessment"]:
            subagents.append(ResearchSubagent(
                name=name,
                domain="fundamental",
                llm=self.llm
            ))
        
        # 5 Sentiment Analysis Subagents
        for name in ["News Sentiment", "Social Media", "Analyst Ratings", "Insider Trading", "Market Sentiment"]:
            subagents.append(ResearchSubagent(
                name=name,
                domain="sentiment",
                llm=self.llm
            ))
        
        return subagents
    
    def _create_synthesizers(self) -> List[DomainSynthesizer]:
        """Create 3 domain synthesizers"""
        return [
            DomainSynthesizer(domain="technical", llm=self.llm),
            DomainSynthesizer(domain="fundamental", llm=self.llm),
            DomainSynthesizer(domain="sentiment", llm=self.llm)
        ]
    
    async def run(self, state: WorkflowState) -> Dict:
        """Execute hierarchical research"""
        ticker = state["ticker"]
        directive = state["research_directive"]
        
        # Tier 1: Execute all 15 subagents in parallel
        subagent_tasks = [
            agent.research(ticker, directive)
            for agent in self.subagents
        ]
        findings = await asyncio.gather(*subagent_tasks)
        
        # Group findings by domain
        technical_findings = [f for f in findings if f["domain"] == "technical"]
        fundamental_findings = [f for f in findings if f["domain"] == "fundamental"]
        sentiment_findings = [f for f in findings if f["domain"] == "sentiment"]
        
        # Tier 2: Execute 3 synthesizers in parallel
        synthesizer_tasks = [
            self.synthesizers[0].synthesize(technical_findings),
            self.synthesizers[1].synthesize(fundamental_findings),
            self.synthesizers[2].synthesize(sentiment_findings)
        ]
        fact_sheets = await asyncio.gather(*synthesizer_tasks)
        
        # Tier 3: Leader synthesizes fact sheets
        final_synthesis = await self.leader.synthesize(fact_sheets, ticker, directive)
        
        return {
            "research_findings": fact_sheets,
            "research_iteration": state["research_iteration"] + 1
        }


class ResearchSubagent:
    """Individual research subagent"""
    
    def __init__(self, name: str, domain: str, llm: ChatOpenAI):
        self.name = name
        self.domain = domain
        self.llm = llm
    
    async def research(self, ticker: str, directive: str) -> Dict:
        """Execute research task"""
        prompt = f"""You are a {self.name} specialist analyzing {ticker}.

Research Directive: {directive}

Your task: Conduct {self.domain} analysis and provide:
1. Key findings (3-5 bullet points)
2. Confidence score (0.0 to 1.0)
3. Supporting evidence
4. Actionable insights

Focus on finding alpha opportunities relevant to the directive.
"""
        
        response = await self.llm.ainvoke([
            SystemMessage(content="You are an expert financial analyst."),
            HumanMessage(content=prompt)
        ])
        
        # Parse response and structure findings
        return {
            "agent_name": self.name,
            "domain": self.domain,
            "findings": response.content,
            "timestamp": datetime.now().isoformat()
        }


class DomainSynthesizer:
    """Synthesizes findings within a domain"""
    
    def __init__(self, domain: str, llm: ChatOpenAI):
        self.domain = domain
        self.llm = llm
    
    async def synthesize(self, findings: List[Dict]) -> FactSheet:
        """Synthesize domain findings into fact sheet"""
        findings_text = "\n\n".join([
            f"Agent: {f['agent_name']}\n{f['findings']}"
            for f in findings
        ])
        
        prompt = f"""Synthesize the following {self.domain} research findings into a concise fact sheet.

{findings_text}

Create a fact sheet with:
1. Key Insights (3-5 most important findings)
2. Overall Confidence (weighted average)
3. Evidence (supporting data points)
4. Consensus vs. Conflicts (where agents agree/disagree)

Format as JSON.
"""
        
        response = await self.llm.ainvoke([
            SystemMessage(content="You are a domain synthesis expert."),
            HumanMessage(content=prompt)
        ])
        
        # Parse and return fact sheet
        return {
            "domain": self.domain,
            "key_insights": [...],  # Parsed from response
            "confidence": 0.85,
            "evidence": [...],
            "agent_ids": [f["agent_name"] for f in findings],
            "timestamp": datetime.now().isoformat()
        }
```

---

### 2. Strategy Development Agent

```python
class StrategyDevelopmentAgent:
    """Generates strategy code based on research"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    async def run(self, state: WorkflowState) -> Dict:
        """Generate strategy variants"""
        research_findings = state["research_findings"]
        failure_analysis = state.get("failure_analysis")
        iteration = state["strategy_iteration"]
        
        if iteration == 1:
            # First iteration: Generate diverse strategies
            variants = await self._generate_initial_strategies(research_findings, state)
        else:
            # Subsequent iterations: Refine based on failure analysis
            variants = await self._refine_strategies(
                research_findings,
                failure_analysis,
                state["strategy_variants"],
                state["backtest_results"]
            )
        
        return {
            "strategy_variants": variants,
            "strategy_iteration": iteration + 1
        }
    
    async def _generate_initial_strategies(
        self,
        research_findings: List[FactSheet],
        state: WorkflowState
    ) -> List[StrategyVariant]:
        """Generate 5 diverse strategy variants"""
        
        # Combine fact sheets
        research_summary = self._summarize_research(research_findings)
        
        prompt = f"""Based on the following research, generate 5 diverse trading strategy variants for {state['ticker']}.

{research_summary}

Requirements:
- Each strategy should use Backtrader framework
- Include complete Python code
- Vary approaches (momentum, mean reversion, breakout, etc.)
- Include parameter variations
- Add clear comments

Return as JSON array of strategies.
"""
        
        response = await self.llm.ainvoke([
            SystemMessage(content="You are an expert algorithmic trading developer."),
            HumanMessage(content=prompt)
        ])
        
        # Parse and validate strategies
        variants = self._parse_strategies(response.content)
        validated_variants = [self._validate_code(v) for v in variants]
        
        return validated_variants
    
    async def _refine_strategies(
        self,
        research_findings: List[FactSheet],
        failure_analysis: FailureAnalysis,
        previous_variants: List[StrategyVariant],
        previous_results: List[BacktestResult]
    ) -> List[StrategyVariant]:
        """Refine strategies based on failure analysis"""
        
        classification = failure_analysis["classification"]
        specific_actions = failure_analysis["specific_actions"]
        
        if classification == "PARAMETER_ISSUE":
            # Generate parameter variations
            return await self._tune_parameters(previous_variants, specific_actions)
        
        elif classification == "ALGORITHM_BUG":
            # Fix bugs
            return await self._fix_bugs(previous_variants, specific_actions)
        
        elif classification == "DESIGN_FLAW":
            # Refine design
            return await self._refine_design(previous_variants, specific_actions, research_findings)
        
        else:
            # Should not reach here (RESEARCH or ABANDON routes elsewhere)
            return previous_variants
    
    def _validate_code(self, variant: StrategyVariant) -> StrategyVariant:
        """4-stage validation pipeline"""
        code = variant["code"]
        
        # Stage 1: Syntax check
        try:
            ast.parse(code)
        except SyntaxError as e:
            raise ValueError(f"Syntax error: {e}")
        
        # Stage 2: Logic check (Backtrader compatibility)
        if "bt.Strategy" not in code:
            raise ValueError("Must inherit from bt.Strategy")
        
        # Stage 3: Security check
        unsafe_patterns = ["eval(", "exec(", "os.system(", "__import__"]
        if any(pattern in code for pattern in unsafe_patterns):
            raise ValueError("Unsafe code detected")
        
        # Stage 4: Performance check (estimated complexity)
        # ... complexity analysis ...
        
        return variant
```

---

### 3. Parallel Backtest Node

```python
async def parallel_backtest_node(state: WorkflowState) -> Dict:
    """Execute backtests in parallel (NOT an agent)"""
    
    strategy_variants = state["strategy_variants"]
    ticker = state["ticker"]
    timeframe = state["timeframe"]
    
    # Execute all backtests in parallel
    backtest_tasks = [
        execute_backtest(variant, ticker, timeframe)
        for variant in strategy_variants
    ]
    
    results = await asyncio.gather(*backtest_tasks, return_exceptions=True)
    
    # Filter out exceptions
    valid_results = [r for r in results if not isinstance(r, Exception)]
    
    # Find best variant
    best_result = max(valid_results, key=lambda r: r["metrics"]["sharpe_ratio"])
    
    return {
        "backtest_results": valid_results,
        "best_strategy": next(v for v in strategy_variants if v["variant_id"] == best_result["variant_id"]),
        "best_metrics": best_result["metrics"]
    }


async def execute_backtest(
    variant: StrategyVariant,
    ticker: str,
    timeframe: str
) -> BacktestResult:
    """Execute single backtest"""
    
    # Load strategy code
    strategy_class = load_strategy_from_code(variant["code"])
    
    # Run Backtrader
    cerebro = bt.Cerebro()
    cerebro.addstrategy(strategy_class, **variant["parameters"])
    
    # Load data
    data = bt.feeds.PandasData(dataname=get_market_data(ticker, timeframe))
    cerebro.adddata(data)
    
    # Run backtest
    cerebro.run()
    
    # Extract metrics
    metrics = extract_metrics(cerebro)
    
    # Check quality gates
    passed_gates = check_quality_gates(metrics, state["quality_criteria"])
    
    return {
        "variant_id": variant["variant_id"],
        "metrics": metrics,
        "passed_gates": passed_gates,
        "gate_scores": compute_gate_scores(metrics)
    }
```

---

### 4. Quality Gate Agent

```python
class QualityGateAgent:
    """Evaluates strategies and routes workflow"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.failure_analyzer = FailureAnalysisAgent(llm)
        self.trajectory_analyzer = TrajectoryAnalyzerAgent(llm)
    
    async def run(self, state: WorkflowState) -> Dict:
        """Evaluate results and make routing decision"""
        
        backtest_results = state["backtest_results"]
        
        # Check if any variant passed
        passed_variants = [r for r in backtest_results if r["passed_gates"]]
        
        if passed_variants:
            # SUCCESS: At least one variant passed
            best_variant = max(passed_variants, key=lambda r: r["metrics"]["sharpe_ratio"])
            return {
                "next_action": "SUCCESS",
                "final_status": "SUCCESS",
                "best_strategy": state["best_strategy"],
                "best_metrics": best_variant["metrics"]
            }
        
        # All failed: Invoke intelligence stack
        failure_analysis = await self.failure_analyzer.analyze(
            strategy_code=state["strategy_variants"][0]["code"],
            research_findings=state["research_findings"],
            backtest_results=backtest_results,
            iteration_history=state["experiment_history"],
            current_iteration=state["strategy_iteration"]
        )
        
        # Invoke trajectory analyzer if history >= 2
        trajectory_analysis = None
        if len(state["experiment_history"]) >= 2:
            trajectory_analysis = await self.trajectory_analyzer.analyze(
                experiment_history=state["experiment_history"],
                current_metrics=state["best_metrics"]
            )
        
        # Make routing decision
        next_action = self._route_decision(
            failure_analysis,
            trajectory_analysis,
            state
        )
        
        # Log experiment
        self._log_experiment(state, failure_analysis, trajectory_analysis, next_action)
        
        return {
            "next_action": next_action,
            "failure_analysis": failure_analysis,
            "trajectory_analysis": trajectory_analysis,
            "total_iterations": state["total_iterations"] + 1
        }
    
    def _route_decision(
        self,
        failure_analysis: FailureAnalysis,
        trajectory_analysis: Optional[TrajectoryAnalysis],
        state: WorkflowState
    ) -> str:
        """Three-tier feedback loop routing"""
        
        classification = failure_analysis["classification"]
        
        # Tier 3: Abandonment
        if classification == "FUNDAMENTAL_IMPOSSIBILITY":
            return "ABANDON"
        
        if state["total_iterations"] >= state["max_total_iterations"]:
            return "ABANDON"
        
        # Check trajectory if available
        if trajectory_analysis and trajectory_analysis["status"] == "DIVERGING":
            return "ABANDON"
        
        # Tier 2: Research refinement
        if classification == "RESEARCH_GAP":
            if state["research_iteration"] < state["max_research_iterations"]:
                return "RESEARCH"
            else:
                return "ABANDON"
        
        # Tier 1: Strategy refinement
        if state["strategy_iteration"] < state["max_strategy_iterations"]:
            if classification == "PARAMETER_ISSUE":
                return "TUNE"
            elif classification == "ALGORITHM_BUG":
                return "FIX"
            elif classification == "DESIGN_FLAW":
                return "REFINE"
        
        # Exhausted iterations
        return "ABANDON"
```

---

## LangGraph Workflow Definition

### Complete Workflow

```python
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

def create_workflow(llm: ChatOpenAI) -> StateGraph:
    """Create complete LangGraph workflow"""
    
    # Initialize agents
    research_swarm = ResearchSwarmAgent(llm)
    strategy_dev = StrategyDevelopmentAgent(llm)
    quality_gate = QualityGateAgent(llm)
    
    # Create graph
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("research_swarm", research_swarm.run)
    workflow.add_node("strategy_dev", strategy_dev.run)
    workflow.add_node("parallel_backtest", parallel_backtest_node)
    workflow.add_node("quality_gate", quality_gate.run)
    
    # Set entry point
    workflow.set_entry_point("research_swarm")
    
    # Add linear edges
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
    
    return workflow


def route_after_quality_gate(state: WorkflowState) -> str:
    """Routing function for conditional edges"""
    action = state["next_action"]
    return action.lower()
```

---

## Usage Example

### Complete Execution

```python
from langchain_openai import ChatOpenAI
from src.core.llm_client import create_llm_with_fallbacks

# Initialize LLM with failover
llm = create_llm_with_fallbacks()

# Create workflow
workflow = create_workflow(llm)
app = workflow.compile()

# User input
user_input = {
    "ticker": "AAPL",
    "research_directive": "Find momentum alpha in tech stocks during earnings season",
    "quality_criteria": {
        "sharpe_ratio": 1.0,
        "max_drawdown": 0.20,
        "win_rate": 0.50,
        "total_return": 0.15
    },
    "timeframe": "1d",
    
    # Initialize iteration counters
    "strategy_iteration": 0,
    "research_iteration": 0,
    "total_iterations": 0,
    
    # Set limits
    "max_strategy_iterations": 5,
    "max_research_iterations": 3,
    "max_total_iterations": 15,
    
    # Initialize tracking
    "experiment_id": f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    "experiment_history": [],
    "final_status": "IN_PROGRESS"
}

# Execute workflow
result = app.invoke(user_input)

# Check result
if result["final_status"] == "SUCCESS":
    print(f"✅ Success! Strategy: {result['best_strategy']['name']}")
    print(f"Metrics: {result['best_metrics']}")
    print(f"Iterations: {result['total_iterations']}")
else:
    print(f"❌ Abandoned after {result['total_iterations']} iterations")
    print(f"Reason: {result['failure_analysis']['reasoning']}")
```

---

## Streaming and Monitoring

### Stream Events

```python
# Stream events for real-time monitoring
async for event in app.astream(user_input):
    node_name = event["node"]
    state = event["state"]
    
    if node_name == "research_swarm":
        print(f"🔍 Research iteration {state['research_iteration']}")
    
    elif node_name == "strategy_dev":
        print(f"💻 Generated {len(state['strategy_variants'])} variants")
    
    elif node_name == "parallel_backtest":
        print(f"📊 Backtesting {len(state['strategy_variants'])} variants...")
    
    elif node_name == "quality_gate":
        print(f"✅ Quality Gate: {state['next_action']}")
        if state.get("failure_analysis"):
            print(f"   Classification: {state['failure_analysis']['classification']}")
```

---

## Testing

### Unit Test Example

```python
import pytest

@pytest.mark.asyncio
async def test_research_swarm_agent():
    """Test research swarm hierarchical synthesis"""
    
    llm = ChatOpenAI(model="gpt-4o-mini")
    agent = ResearchSwarmAgent(llm)
    
    state = {
        "ticker": "AAPL",
        "research_directive": "Find momentum alpha",
        "research_iteration": 0
    }
    
    result = await agent.run(state)
    
    # Assertions
    assert len(result["research_findings"]) == 3  # 3 fact sheets
    assert all(fs["domain"] in ["technical", "fundamental", "sentiment"] for fs in result["research_findings"])
    assert result["research_iteration"] == 1
```

---

## Summary

**LangGraph Components Used**:
- ✅ StateGraph for workflow orchestration
- ✅ TypedDict for state schema
- ✅ Nodes for agent invocation
- ✅ Edges for linear flow
- ✅ Conditional edges for routing
- ✅ Map-reduce for parallel execution (backtests)
- ✅ Streaming for monitoring

**Key Patterns**:
- Hierarchical agent architecture (Research Swarm)
- Async/await for parallel execution
- LLM-powered decision making (Failure Analysis, Trajectory Analyzer)
- State-driven workflow (no custom orchestrator)

**Next Steps**: Implement Phase 2 (Memory System) to persist findings and strategies.

---

**End of LangGraph Implementation Guide**
