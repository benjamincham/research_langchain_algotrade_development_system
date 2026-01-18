"""
Central Orchestrator - LangGraph Workflow Implementation

This module implements the complete LangGraph workflow for the algorithmic trading
strategy development system using Workflow-Based Communication (Decision D-025).

Architecture:
- 4 Nodes: research_swarm, strategy_dev, parallel_backtest, quality_gate
- Conditional Edges: Three-tier feedback loops
- State: Centralized WorkflowState TypedDict
- Communication: LastValue channels (LangGraph default)

Reference: docs/design/AGENT_COMMUNICATION_APPROACHES.md
"""

import asyncio
from typing import TypedDict, List, Dict, Any, Literal, Annotated
from datetime import datetime
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langfuse.callback import CallbackHandler

# Import agents (to be implemented in Phase 5)
from src.agents.research_swarm.research_swarm_agent import ResearchSwarmAgent
from src.agents.strategy_dev.strategy_development_agent import StrategyDevelopmentAgent
from src.agents.quality_gate.quality_gate_agent import QualityGateAgent
from src.agents.quality_gate.schemas import (
    ResearchFinding,
    StrategyVariant,
    BacktestMetrics,
    QualityGateResult,
    IterationHistory,
)

# Import tools
from src.memory.memory_manager import MemoryManager
from src.core.llm_client import create_llm_with_fallbacks
from src.core.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# STATE DEFINITION
# =============================================================================

class WorkflowState(TypedDict):
    """
    Centralized state for the entire workflow.
    
    This TypedDict defines all state that flows through the LangGraph workflow.
    Each node reads from and writes to this state. LangGraph manages state
    updates automatically using the LastValue channel (default).
    
    Design Decision D-025: Workflow-Based Communication
    - State is centralized and deterministic
    - All nodes communicate through state updates
    - No pub-sub or event-driven communication
    """
    
    # -------------------------------------------------------------------------
    # Input (Set by user at workflow start)
    # -------------------------------------------------------------------------
    ticker: str  # Stock ticker to research (e.g., "AAPL")
    user_objective: str  # User's goal (e.g., "Find momentum strategies")
    max_iterations: int  # Maximum total iterations (default: 15)
    max_strategy_iterations: int  # Max iterations for strategy refinement (default: 5)
    max_research_iterations: int  # Max iterations for research refinement (default: 3)
    
    # -------------------------------------------------------------------------
    # Research Swarm Phase (Written by research_swarm node)
    # -------------------------------------------------------------------------
    research_findings: List[ResearchFinding]  # All findings from 15 subagents
    technical_fact_sheet: Dict[str, Any]  # Technical domain synthesis
    fundamental_fact_sheet: Dict[str, Any]  # Fundamental domain synthesis
    sentiment_fact_sheet: Dict[str, Any]  # Sentiment domain synthesis
    research_synthesis: str  # Final cross-domain synthesis from Research Leader
    research_confidence: float  # Confidence in research quality (0.0-1.0)
    
    # -------------------------------------------------------------------------
    # Strategy Development Phase (Written by strategy_dev node)
    # -------------------------------------------------------------------------
    strategy_variants: List[StrategyVariant]  # 3-5 strategy variants generated
    strategy_rationale: str  # Why these strategies were chosen
    expected_performance: Dict[str, float]  # Expected metrics (Sharpe, etc.)
    
    # -------------------------------------------------------------------------
    # Backtesting Phase (Written by parallel_backtest node)
    # -------------------------------------------------------------------------
    backtest_results: List[BacktestMetrics]  # Results for each variant
    best_variant_index: int  # Index of best performing variant
    backtest_summary: str  # Summary of backtest results
    
    # -------------------------------------------------------------------------
    # Quality Gate Phase (Written by quality_gate node)
    # -------------------------------------------------------------------------
    quality_gate_results: List[QualityGateResult]  # Gate results for each variant
    passed_quality_gates: bool  # True if any variant passed all gates
    failure_analysis: Dict[str, Any]  # Output from Failure Analysis Agent
    trajectory_analysis: Dict[str, Any]  # Output from Trajectory Analyzer Agent
    next_action: Literal[
        "SUCCESS",  # Strategy passed, end workflow
        "TUNE_PARAMETERS",  # Tier 1: Refine parameters
        "FIX_BUG",  # Tier 1: Fix algorithm bug
        "REFINE_ALGORITHM",  # Tier 1: Improve algorithm design
        "REFINE_RESEARCH",  # Tier 2: Go back to research
        "ABANDON",  # Tier 3: Give up, alpha doesn't exist
    ]
    decision_reasoning: str  # Why this action was chosen
    
    # -------------------------------------------------------------------------
    # Iteration Tracking (Updated by all nodes)
    # -------------------------------------------------------------------------
    current_iteration: int  # Total iterations so far
    strategy_iteration: int  # Strategy refinement iterations
    research_iteration: int  # Research refinement iterations
    iteration_history: List[IterationHistory]  # Complete history of all iterations
    
    # -------------------------------------------------------------------------
    # Final Output (Set when workflow ends)
    # -------------------------------------------------------------------------
    final_strategy: StrategyVariant | None  # Best strategy if SUCCESS
    final_status: Literal["SUCCESS", "ABANDONED", "MAX_ITERATIONS"]
    final_message: str  # Human-readable summary
    
    # -------------------------------------------------------------------------
    # Metadata (For monitoring and debugging)
    # -------------------------------------------------------------------------
    workflow_id: str  # Unique ID for this workflow run
    start_time: datetime  # When workflow started
    end_time: datetime | None  # When workflow ended
    total_llm_calls: int  # Total LLM invocations
    total_cost: float  # Total cost in USD
    error_log: List[str]  # Any errors encountered


# =============================================================================
# NODE FUNCTIONS
# =============================================================================

async def research_swarm_node(state: WorkflowState) -> Dict[str, Any]:
    """
    Node 1: Research Swarm Phase
    
    Coordinates 19 research agents (15 subagents + 3 synthesizers + 1 leader)
    to gather comprehensive market intelligence.
    
    Architecture (Hierarchical Synthesis - Decision D-011):
    - Tier 1: 15 subagents (5 technical + 5 fundamental + 5 sentiment) run in parallel
    - Tier 2: 3 domain synthesizers produce Fact Sheets
    - Tier 3: 1 Research Leader produces final cross-domain synthesis
    
    Args:
        state: Current workflow state
        
    Returns:
        Dict with updated state fields:
        - research_findings
        - technical_fact_sheet
        - fundamental_fact_sheet
        - sentiment_fact_sheet
        - research_synthesis
        - research_confidence
    """
    logger.info(f"[Research Swarm] Starting research for {state['ticker']}")
    
    # Initialize Research Swarm Agent
    research_agent = ResearchSwarmAgent(
        ticker=state["ticker"],
        objective=state["user_objective"],
        llm=create_llm_with_fallbacks(),
        memory=MemoryManager(),
    )
    
    # Execute hierarchical research
    result = await research_agent.execute_research()
    
    logger.info(f"[Research Swarm] Completed. Found {len(result['findings'])} insights")
    
    return {
        "research_findings": result["findings"],
        "technical_fact_sheet": result["technical_fact_sheet"],
        "fundamental_fact_sheet": result["fundamental_fact_sheet"],
        "sentiment_fact_sheet": result["sentiment_fact_sheet"],
        "research_synthesis": result["synthesis"],
        "research_confidence": result["confidence"],
        "research_iteration": state["research_iteration"] + 1,
    }


async def strategy_dev_node(state: WorkflowState) -> Dict[str, Any]:
    """
    Node 2: Strategy Development Phase
    
    Generates 3-5 trading strategy variants based on research findings.
    
    Inputs from state:
    - research_synthesis: Cross-domain insights
    - technical/fundamental/sentiment_fact_sheet: Domain-specific insights
    - iteration_history: Past attempts (for refinement)
    - failure_analysis: Why previous strategies failed (if applicable)
    
    Outputs:
    - strategy_variants: List of 3-5 StrategyVariant objects
    - strategy_rationale: Why these strategies were chosen
    - expected_performance: Expected Sharpe, drawdown, etc.
    
    Args:
        state: Current workflow state
        
    Returns:
        Dict with updated state fields
    """
    logger.info(f"[Strategy Dev] Generating strategies for {state['ticker']}")
    
    # Initialize Strategy Development Agent
    strategy_agent = StrategyDevelopmentAgent(
        ticker=state["ticker"],
        research_synthesis=state["research_synthesis"],
        fact_sheets={
            "technical": state["technical_fact_sheet"],
            "fundamental": state["fundamental_fact_sheet"],
            "sentiment": state["sentiment_fact_sheet"],
        },
        llm=create_llm_with_fallbacks(),
        memory=MemoryManager(),
    )
    
    # If this is a refinement iteration, provide failure analysis
    refinement_context = None
    if state["strategy_iteration"] > 0 and state.get("failure_analysis"):
        refinement_context = {
            "failure_analysis": state["failure_analysis"],
            "trajectory_analysis": state.get("trajectory_analysis"),
            "previous_variants": state.get("strategy_variants", []),
            "iteration_history": state.get("iteration_history", []),
        }
    
    # Generate strategy variants
    result = await strategy_agent.generate_strategies(
        refinement_context=refinement_context
    )
    
    logger.info(f"[Strategy Dev] Generated {len(result['variants'])} variants")
    
    return {
        "strategy_variants": result["variants"],
        "strategy_rationale": result["rationale"],
        "expected_performance": result["expected_performance"],
        "strategy_iteration": state["strategy_iteration"] + 1,
        "current_iteration": state["current_iteration"] + 1,
    }


async def parallel_backtest_node(state: WorkflowState) -> Dict[str, Any]:
    """
    Node 3: Parallel Backtesting Phase
    
    Backtests all strategy variants in parallel using queue-and-worker pattern.
    
    Architecture (Decision D-022: Queue-and-Worker Pattern):
    - Task queue holds all backtest tasks
    - 5 workers execute tasks in parallel
    - If task fails, requeue (up to 3 retries)
    - If resources unavailable, wait and retry
    
    Inputs from state:
    - strategy_variants: List of strategies to backtest
    
    Outputs:
    - backtest_results: List of BacktestMetrics (one per variant)
    - best_variant_index: Index of best performing variant
    - backtest_summary: Human-readable summary
    
    Args:
        state: Current workflow state
        
    Returns:
        Dict with updated state fields
    """
    logger.info(f"[Parallel Backtest] Testing {len(state['strategy_variants'])} variants")
    
    from src.backtesting.backtest_engine import BacktestEngine
    from src.workflow.worker_pool import WorkerPool
    
    # Initialize backtest engine and worker pool
    engine = BacktestEngine()
    worker_pool = WorkerPool(max_workers=5)
    
    # Create backtest tasks
    tasks = [
        {
            "variant_index": i,
            "strategy_code": variant.code,
            "parameters": variant.parameters,
            "ticker": state["ticker"],
        }
        for i, variant in enumerate(state["strategy_variants"])
    ]
    
    # Execute in parallel with queue-and-worker
    results = await worker_pool.execute_parallel(tasks, engine.run_backtest)
    
    # Find best variant (highest Sharpe ratio)
    best_index = max(range(len(results)), key=lambda i: results[i].sharpe_ratio)
    
    # Create summary
    summary = f"Tested {len(results)} variants. Best: Variant {best_index} (Sharpe: {results[best_index].sharpe_ratio:.2f})"
    
    logger.info(f"[Parallel Backtest] Completed. {summary}")
    
    return {
        "backtest_results": results,
        "best_variant_index": best_index,
        "backtest_summary": summary,
    }


async def quality_gate_node(state: WorkflowState) -> Dict[str, Any]:
    """
    Node 4: Quality Gate Phase
    
    Evaluates all strategy variants against quality gates and determines next action.
    
    Sub-agents:
    1. Quality Gate Agent: Evaluates metrics against thresholds
    2. Failure Analysis Agent: Diagnoses WHY strategies failed (if any failed)
    3. Trajectory Analyzer Agent: Analyzes improvement trajectory (if history >= 2)
    
    Three-Tier Feedback Loop Logic (Decision D-019):
    - Tier 1: TUNE/FIX/REFINE → strategy_dev (fixable issues)
    - Tier 2: REFINE_RESEARCH → research_swarm (wrong research direction)
    - Tier 3: ABANDON → END (alpha doesn't exist)
    
    Args:
        state: Current workflow state
        
    Returns:
        Dict with updated state fields:
        - quality_gate_results
        - passed_quality_gates
        - failure_analysis
        - trajectory_analysis
        - next_action
        - decision_reasoning
    """
    logger.info(f"[Quality Gate] Evaluating {len(state['backtest_results'])} results")
    
    # Initialize Quality Gate Agent
    quality_gate_agent = QualityGateAgent(
        llm=create_llm_with_fallbacks(),
        memory=MemoryManager(),
    )
    
    # Evaluate all variants
    gate_results = await quality_gate_agent.evaluate_variants(
        strategy_variants=state["strategy_variants"],
        backtest_results=state["backtest_results"],
        research_findings=state["research_findings"],
    )
    
    # Check if any variant passed all gates
    passed = any(result.passed_all_gates for result in gate_results)
    
    if passed:
        # SUCCESS: At least one strategy passed
        logger.info("[Quality Gate] SUCCESS - Strategy passed all gates")
        return {
            "quality_gate_results": gate_results,
            "passed_quality_gates": True,
            "next_action": "SUCCESS",
            "decision_reasoning": "At least one strategy variant passed all quality gates",
            "final_strategy": state["strategy_variants"][state["best_variant_index"]],
            "final_status": "SUCCESS",
            "final_message": f"Successfully developed strategy for {state['ticker']}",
        }
    
    # FAILURE: No variant passed, need to analyze and decide next action
    logger.info("[Quality Gate] FAILED - Analyzing failure...")
    
    # Invoke Failure Analysis Agent
    from src.agents.quality_gate.failure_analysis_agent import FailureAnalysisAgent
    from src.agents.quality_gate.schemas import create_failure_analysis_input_from_state
    
    failure_agent = FailureAnalysisAgent(llm=create_llm_with_fallbacks())
    failure_input = create_failure_analysis_input_from_state(state)
    failure_analysis = await failure_agent.analyze(failure_input)
    
    # If we have iteration history (>= 2), invoke Trajectory Analyzer Agent
    trajectory_analysis = None
    if len(state.get("iteration_history", [])) >= 2:
        from src.agents.quality_gate.trajectory_analyzer_agent import TrajectoryAnalyzerAgent
        from src.agents.quality_gate.schemas import create_trajectory_analysis_input_from_state
        
        trajectory_agent = TrajectoryAnalyzerAgent(llm=create_llm_with_fallbacks())
        trajectory_input = create_trajectory_analysis_input_from_state(state)
        trajectory_analysis = await trajectory_agent.analyze(trajectory_input)
    
    # Determine next action based on analyses
    next_action = determine_next_action(
        failure_analysis=failure_analysis,
        trajectory_analysis=trajectory_analysis,
        state=state,
    )
    
    logger.info(f"[Quality Gate] Next action: {next_action}")
    
    return {
        "quality_gate_results": gate_results,
        "passed_quality_gates": False,
        "failure_analysis": failure_analysis.model_dump(),
        "trajectory_analysis": trajectory_analysis.model_dump() if trajectory_analysis else None,
        "next_action": next_action,
        "decision_reasoning": failure_analysis.reasoning,
    }


def determine_next_action(
    failure_analysis: Any,
    trajectory_analysis: Any | None,
    state: WorkflowState,
) -> Literal["TUNE_PARAMETERS", "FIX_BUG", "REFINE_ALGORITHM", "REFINE_RESEARCH", "ABANDON"]:
    """
    Determine next action based on failure analysis and trajectory analysis.
    
    Three-Tier Feedback Loop Logic:
    
    Tier 1 (Strategy Refinement):
    - TUNE_PARAMETERS: Metrics close to passing, just need parameter adjustment
    - FIX_BUG: Algorithm has a bug (e.g., incorrect RSI calculation)
    - REFINE_ALGORITHM: Algorithm design needs improvement (e.g., add regime awareness)
    
    Tier 2 (Research Refinement):
    - REFINE_RESEARCH: Research direction is wrong, need different insights
    
    Tier 3 (Abandonment):
    - ABANDON: Alpha doesn't exist, or max iterations reached
    
    Args:
        failure_analysis: Output from Failure Analysis Agent
        trajectory_analysis: Output from Trajectory Analyzer Agent (or None)
        state: Current workflow state
        
    Returns:
        Next action to take
    """
    
    # Check iteration limits first
    if state["current_iteration"] >= state["max_iterations"]:
        return "ABANDON"
    
    if state["strategy_iteration"] >= state["max_strategy_iterations"]:
        # Exhausted strategy iterations, must go to research or abandon
        if state["research_iteration"] < state["max_research_iterations"]:
            return "REFINE_RESEARCH"
        else:
            return "ABANDON"
    
    # Check trajectory analysis (if available)
    if trajectory_analysis:
        if trajectory_analysis.convergence_status == "DIVERGING":
            # Getting worse, need to pivot
            if state["research_iteration"] < state["max_research_iterations"]:
                return "REFINE_RESEARCH"
            else:
                return "ABANDON"
        
        if trajectory_analysis.convergence_status == "STAGNANT":
            # Not improving, need different approach
            if state["strategy_iteration"] < 3:
                # Try a few more strategy iterations
                pass  # Continue to failure analysis recommendation
            else:
                # Tried enough, go to research
                if state["research_iteration"] < state["max_research_iterations"]:
                    return "REFINE_RESEARCH"
                else:
                    return "ABANDON"
    
    # Use failure analysis recommendation
    recommendation = failure_analysis.recommendation
    
    if recommendation == "TUNE_PARAMETERS":
        return "TUNE_PARAMETERS"
    elif recommendation == "FIX_BUG":
        return "FIX_BUG"
    elif recommendation == "REFINE_ALGORITHM":
        return "REFINE_ALGORITHM"
    elif recommendation == "REFINE_RESEARCH":
        if state["research_iteration"] < state["max_research_iterations"]:
            return "REFINE_RESEARCH"
        else:
            return "ABANDON"
    elif recommendation == "ABANDON":
        return "ABANDON"
    else:
        # Default: try parameter tuning
        return "TUNE_PARAMETERS"


# =============================================================================
# EDGE LOGIC (Conditional Routing)
# =============================================================================

def route_after_quality_gate(state: WorkflowState) -> str:
    """
    Conditional edge after quality_gate node.
    
    Routes to different nodes based on next_action:
    - SUCCESS → END (workflow complete)
    - TUNE_PARAMETERS, FIX_BUG, REFINE_ALGORITHM → strategy_dev (Tier 1)
    - REFINE_RESEARCH → research_swarm (Tier 2)
    - ABANDON → END (Tier 3)
    
    Args:
        state: Current workflow state
        
    Returns:
        Name of next node or END
    """
    action = state["next_action"]
    
    if action == "SUCCESS":
        return END
    elif action in ["TUNE_PARAMETERS", "FIX_BUG", "REFINE_ALGORITHM"]:
        # Tier 1: Strategy refinement
        return "strategy_dev"
    elif action == "REFINE_RESEARCH":
        # Tier 2: Research refinement
        return "research_swarm"
    elif action == "ABANDON":
        # Tier 3: Give up
        return END
    else:
        # Should never happen, but default to END
        logger.error(f"Unknown next_action: {action}")
        return END


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def create_workflow_graph() -> StateGraph:
    """
    Create the complete LangGraph workflow.
    
    Architecture:
    - 4 Nodes: research_swarm, strategy_dev, parallel_backtest, quality_gate
    - 1 Conditional Edge: route_after_quality_gate
    - Entry Point: research_swarm
    - End Points: SUCCESS or ABANDON
    
    Returns:
        Compiled StateGraph ready for execution
    """
    
    # Create graph
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("research_swarm", research_swarm_node)
    workflow.add_node("strategy_dev", strategy_dev_node)
    workflow.add_node("parallel_backtest", parallel_backtest_node)
    workflow.add_node("quality_gate", quality_gate_node)
    
    # Set entry point
    workflow.set_entry_point("research_swarm")
    
    # Add edges
    workflow.add_edge("research_swarm", "strategy_dev")
    workflow.add_edge("strategy_dev", "parallel_backtest")
    workflow.add_edge("parallel_backtest", "quality_gate")
    
    # Add conditional edge (three-tier feedback loops)
    workflow.add_conditional_edges(
        "quality_gate",
        route_after_quality_gate,
        {
            "strategy_dev": "strategy_dev",  # Tier 1
            "research_swarm": "research_swarm",  # Tier 2
            END: END,  # Tier 3 or SUCCESS
        }
    )
    
    # Compile with checkpointing for state persistence
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)
    
    return graph


# =============================================================================
# EXECUTION FUNCTIONS
# =============================================================================

async def execute_workflow(
    ticker: str,
    user_objective: str = "Find profitable trading strategies",
    max_iterations: int = 15,
    max_strategy_iterations: int = 5,
    max_research_iterations: int = 3,
    enable_monitoring: bool = True,
) -> WorkflowState:
    """
    Execute the complete workflow for a given ticker.
    
    Args:
        ticker: Stock ticker to research (e.g., "AAPL")
        user_objective: User's goal
        max_iterations: Maximum total iterations
        max_strategy_iterations: Max strategy refinement iterations
        max_research_iterations: Max research refinement iterations
        enable_monitoring: Enable LangFuse monitoring
        
    Returns:
        Final workflow state
    """
    
    # Create initial state
    initial_state: WorkflowState = {
        "ticker": ticker,
        "user_objective": user_objective,
        "max_iterations": max_iterations,
        "max_strategy_iterations": max_strategy_iterations,
        "max_research_iterations": max_research_iterations,
        "research_findings": [],
        "technical_fact_sheet": {},
        "fundamental_fact_sheet": {},
        "sentiment_fact_sheet": {},
        "research_synthesis": "",
        "research_confidence": 0.0,
        "strategy_variants": [],
        "strategy_rationale": "",
        "expected_performance": {},
        "backtest_results": [],
        "best_variant_index": 0,
        "backtest_summary": "",
        "quality_gate_results": [],
        "passed_quality_gates": False,
        "failure_analysis": {},
        "trajectory_analysis": {},
        "next_action": "SUCCESS",
        "decision_reasoning": "",
        "current_iteration": 0,
        "strategy_iteration": 0,
        "research_iteration": 0,
        "iteration_history": [],
        "final_strategy": None,
        "final_status": "SUCCESS",
        "final_message": "",
        "workflow_id": f"workflow_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "start_time": datetime.now(),
        "end_time": None,
        "total_llm_calls": 0,
        "total_cost": 0.0,
        "error_log": [],
    }
    
    # Create graph
    graph = create_workflow_graph()
    
    # Setup monitoring
    callbacks = []
    if enable_monitoring:
        langfuse_handler = CallbackHandler()
        callbacks.append(langfuse_handler)
    
    # Execute workflow
    logger.info(f"Starting workflow for {ticker}")
    
    config = {
        "configurable": {"thread_id": initial_state["workflow_id"]},
        "callbacks": callbacks,
    }
    
    final_state = await graph.ainvoke(initial_state, config=config)
    
    # Update end time
    final_state["end_time"] = datetime.now()
    
    logger.info(f"Workflow completed: {final_state['final_status']}")
    
    return final_state


async def stream_workflow(
    ticker: str,
    user_objective: str = "Find profitable trading strategies",
    max_iterations: int = 15,
) -> None:
    """
    Execute workflow with streaming updates (for real-time monitoring).
    
    Args:
        ticker: Stock ticker to research
        user_objective: User's goal
        max_iterations: Maximum iterations
    """
    
    # Create initial state (same as execute_workflow)
    initial_state = {
        "ticker": ticker,
        "user_objective": user_objective,
        "max_iterations": max_iterations,
        # ... (same as above)
    }
    
    # Create graph
    graph = create_workflow_graph()
    
    # Stream execution
    async for event in graph.astream(initial_state):
        node_name = list(event.keys())[0]
        node_output = event[node_name]
        
        print(f"\n=== {node_name.upper()} ===")
        print(f"Output: {node_output}")
        
        # Can send updates to UI here
        # await websocket.send(json.dumps({"node": node_name, "output": node_output}))


# =============================================================================
# MAIN (For testing)
# =============================================================================

if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Test workflow execution
        result = await execute_workflow(
            ticker="AAPL",
            user_objective="Find momentum strategies with low drawdown",
            max_iterations=10,
        )
        
        print("\n" + "="*80)
        print("WORKFLOW RESULT")
        print("="*80)
        print(f"Status: {result['final_status']}")
        print(f"Message: {result['final_message']}")
        print(f"Iterations: {result['current_iteration']}")
        print(f"Strategy Iterations: {result['strategy_iteration']}")
        print(f"Research Iterations: {result['research_iteration']}")
        
        if result['final_strategy']:
            print(f"\nFinal Strategy:")
            print(f"  Name: {result['final_strategy'].name}")
            print(f"  Type: {result['final_strategy'].type}")
            print(f"  Sharpe: {result['backtest_results'][result['best_variant_index']].sharpe_ratio:.2f}")
    
    asyncio.run(main())
