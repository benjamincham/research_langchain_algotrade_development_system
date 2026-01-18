# Agent Monitoring with LangFuse

## Overview

This document defines the monitoring and observability framework for all 23 agents in the system using **LangFuse**, an open-source LLM engineering platform specifically designed for tracing and monitoring LangChain/LangGraph applications.

## Why LangFuse?

**LangFuse is the perfect fit** for our agentic system because:

1. **Native LangChain Integration**: Automatic tracing via `CallbackHandler`
2. **Zero Instrumentation Overhead**: No manual logging code required
3. **Hierarchical Traces**: Automatically captures nested agent calls
4. **Cost Tracking**: Automatic token counting and cost calculation
5. **Prompt Management**: Version control for prompts
6. **User Feedback**: Built-in annotation and scoring
7. **Production-Ready**: Dashboard, alerting, and analytics included
8. **Open Source**: Self-hostable, no vendor lock-in

## Architecture

### LangFuse Integration Stack

```
┌─────────────────────────────────────────────────────────────┐
│  LangFuse Dashboard (Web UI)                                │
│  - Trace visualization                                      │
│  - Cost analytics                                           │
│  - Performance metrics                                      │
│  - User feedback                                            │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│  LangFuse API (Cloud or Self-Hosted)                        │
│  - Trace ingestion                                          │
│  - Metrics aggregation                                      │
│  - Data storage                                             │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│  LangFuse CallbackHandler (Python SDK)                      │
│  - Automatic trace capture                                  │
│  - Token counting                                           │
│  - Cost calculation                                         │
│  - Metadata attachment                                      │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│  LangGraph Workflow                                         │
│  - Research Swarm Agent                                     │
│  - Strategy Development Agent                               │
│  - Quality Gate Agent                                       │
│  - All 23 agents automatically traced                       │
└─────────────────────────────────────────────────────────────┘
```

## Setup

### 1. Installation

```bash
pip install langfuse langchain langchain-openai
```

### 2. Environment Configuration

Add to `.env`:

```bash
# LangFuse Configuration
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com  # Or self-hosted URL

# OpenAI Configuration (already have)
OPENAI_API_KEY=sk-...
```

### 3. Initialize LangFuse Handler

```python
from langfuse.langchain import CallbackHandler

# Initialize once, use everywhere
langfuse_handler = CallbackHandler(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_BASE_URL")
)
```

## Integration with LangGraph

### Automatic Tracing for Entire Workflow

```python
from langgraph.graph import StateGraph
from langfuse.langchain import CallbackHandler

def create_monitored_graph():
    """Create LangGraph with LangFuse monitoring"""
    
    # Initialize LangFuse handler
    langfuse_handler = CallbackHandler()
    
    # Create graph
    graph = StateGraph(WorkflowState)
    
    # Add nodes (no instrumentation needed!)
    graph.add_node("research_swarm", research_swarm_node)
    graph.add_node("strategy_dev", strategy_dev_node)
    graph.add_node("parallel_backtest", parallel_backtest_node)
    graph.add_node("quality_gate", quality_gate_node)
    
    # Add edges
    graph.add_edge(START, "research_swarm")
    graph.add_edge("research_swarm", "strategy_dev")
    graph.add_edge("strategy_dev", "parallel_backtest")
    graph.add_edge("parallel_backtest", "quality_gate")
    graph.add_conditional_edges(
        "quality_gate",
        route_after_quality_gate,
        {
            "success": END,
            "tune": "strategy_dev",
            "research": "research_swarm",
            "abandon": END
        }
    )
    
    # Compile graph
    compiled_graph = graph.compile()
    
    return compiled_graph, langfuse_handler

# Usage
graph, langfuse_handler = create_monitored_graph()

# Execute with automatic tracing
result = graph.invoke(
    {
        "ticker": "AAPL",
        "workflow_id": "wf_001",
        "iteration_number": 1
    },
    config={"callbacks": [langfuse_handler]}
)
```

**That's it!** LangFuse automatically captures:
- All LLM calls (research agents, strategy dev, quality gate)
- Nested agent hierarchies (research swarm → subagents → synthesizers → leader)
- Token usage and costs
- Latencies and errors
- Input/output for every step

## Metadata and Tagging

### Add Custom Metadata to Traces

```python
from langfuse.langchain import CallbackHandler

# Create handler with metadata
langfuse_handler = CallbackHandler(
    tags=["production", "algo-trading"],
    session_id="session_20260118",
    user_id="user_123",
    metadata={
        "ticker": "AAPL",
        "workflow_id": "wf_001",
        "iteration": 1,
        "environment": "production"
    }
)

# Execute workflow
result = graph.invoke(state, config={"callbacks": [langfuse_handler]})
```

### Tag Individual Agent Calls

```python
async def research_swarm_node(state: WorkflowState) -> WorkflowState:
    """Research swarm with custom tagging"""
    
    # Create handler for this specific node
    langfuse_handler = CallbackHandler(
        tags=["research_swarm", state["ticker"]],
        metadata={
            "workflow_id": state["workflow_id"],
            "iteration": state["iteration_number"],
            "num_subagents": 15
        }
    )
    
    # Invoke research leader
    research_report = await research_leader_agent.invoke(
        state,
        config={"callbacks": [langfuse_handler]}
    )
    
    return {"research_report": research_report}
```

## Monitoring Agent Performance

### View Traces in LangFuse Dashboard

LangFuse automatically provides:

1. **Trace View**: Hierarchical view of all agent calls
   - Research Swarm
     - Price Action Analyst
     - Volume Profile Analyst
     - ... (all 15 subagents)
     - Technical Synthesizer
     - Fundamental Synthesizer
     - Sentiment Synthesizer
     - Research Leader

2. **Cost Analytics**: Token usage and cost per agent
   - Total cost per workflow
   - Cost breakdown by agent type
   - Cost trends over time

3. **Performance Metrics**: Latency and throughput
   - P50, P95, P99 latencies per agent
   - Success/failure rates
   - Error analysis

4. **User Feedback**: Manual scoring and annotation
   - Score strategies (1-5 stars)
   - Add comments
   - Flag issues

### Query Metrics Programmatically

```python
from langfuse import Langfuse

# Initialize client
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_BASE_URL")
)

# Get traces for a specific workflow
traces = langfuse.get_traces(
    name="algo_trading_workflow",
    tags=["AAPL", "production"],
    from_timestamp=datetime.now() - timedelta(days=7)
)

# Analyze agent performance
for trace in traces:
    print(f"Workflow: {trace.id}")
    print(f"Duration: {trace.duration}ms")
    print(f"Cost: ${trace.calculated_total_cost}")
    print(f"Status: {trace.status}")
    
    # Get observations (individual agent calls)
    for obs in trace.observations:
        print(f"  Agent: {obs.name}")
        print(f"  Duration: {obs.duration}ms")
        print(f"  Tokens: {obs.usage.total}")
        print(f"  Cost: ${obs.calculated_total_cost}")
```

## Agent Health Monitoring

### Automatic Error Tracking

LangFuse automatically captures errors:

```python
# Errors are automatically logged
try:
    result = graph.invoke(state, config={"callbacks": [langfuse_handler]})
except Exception as e:
    # Error is already in LangFuse with full context
    logger.error(f"Workflow failed: {e}")
```

### Custom Health Checks

```python
class AgentHealthMonitor:
    """Monitor agent health using LangFuse data"""
    
    def __init__(self):
        self.langfuse = Langfuse()
    
    def check_agent_health(self, agent_name: str, time_window_hours: int = 24) -> dict:
        """Check health of a specific agent"""
        
        # Get recent traces for this agent
        traces = self.langfuse.get_traces(
            name=agent_name,
            from_timestamp=datetime.now() - timedelta(hours=time_window_hours)
        )
        
        total = len(traces)
        if total == 0:
            return {"status": "no_data", "message": "No recent activity"}
        
        # Calculate metrics
        successful = sum(1 for t in traces if t.status == "success")
        failed = total - successful
        success_rate = successful / total
        
        avg_duration = sum(t.duration for t in traces) / total
        avg_cost = sum(t.calculated_total_cost for t in traces) / total
        
        # Determine health status
        if success_rate < 0.5:
            status = "critical"
        elif success_rate < 0.9:
            status = "degraded"
        else:
            status = "healthy"
        
        return {
            "status": status,
            "success_rate": success_rate,
            "total_invocations": total,
            "failed_invocations": failed,
            "avg_duration_ms": avg_duration,
            "avg_cost_usd": avg_cost
        }
    
    def get_failing_agents(self) -> list:
        """Get list of agents with health issues"""
        
        agent_names = [
            "price_action_analyst",
            "volume_profile_analyst",
            # ... all 23 agents
        ]
        
        failing_agents = []
        for agent_name in agent_names:
            health = self.check_agent_health(agent_name)
            if health["status"] in ["critical", "degraded"]:
                failing_agents.append({
                    "agent": agent_name,
                    "health": health
                })
        
        return failing_agents
```

## Experiment Tracking

### Track Strategy Experiments

```python
from langfuse.decorators import observe, langfuse_context

@observe()
async def run_strategy_experiment(
    ticker: str,
    strategy_code: str,
    parameters: dict
) -> BacktestMetrics:
    """Run strategy experiment with LangFuse tracking"""
    
    # Add experiment metadata
    langfuse_context.update_current_trace(
        name="strategy_experiment",
        tags=["backtest", ticker],
        metadata={
            "ticker": ticker,
            "strategy_type": parameters.get("type"),
            "parameters": parameters
        }
    )
    
    # Run backtest
    metrics = await backtest_engine.run(strategy_code, ticker, parameters)
    
    # Log results as scores
    langfuse_context.score_current_trace(
        name="sharpe_ratio",
        value=metrics.sharpe_ratio
    )
    langfuse_context.score_current_trace(
        name="max_drawdown",
        value=metrics.max_drawdown
    )
    langfuse_context.score_current_trace(
        name="total_return",
        value=metrics.total_return
    )
    
    return metrics
```

### Compare Experiments

```python
def compare_strategy_variants(ticker: str, experiment_ids: list) -> pd.DataFrame:
    """Compare multiple strategy variants"""
    
    langfuse = Langfuse()
    
    results = []
    for exp_id in experiment_ids:
        trace = langfuse.get_trace(exp_id)
        
        # Extract scores
        sharpe = next((s.value for s in trace.scores if s.name == "sharpe_ratio"), None)
        drawdown = next((s.value for s in trace.scores if s.name == "max_drawdown"), None)
        returns = next((s.value for s in trace.scores if s.name == "total_return"), None)
        
        results.append({
            "experiment_id": exp_id,
            "sharpe_ratio": sharpe,
            "max_drawdown": drawdown,
            "total_return": returns,
            "cost": trace.calculated_total_cost,
            "duration": trace.duration
        })
    
    return pd.DataFrame(results)
```

## Prompt Management

### Version Control for Prompts

```python
from langfuse import Langfuse

langfuse = Langfuse()

# Create prompt template
langfuse.create_prompt(
    name="research_analyst_prompt",
    prompt="Analyze {ticker} from a {perspective} perspective. Focus on {timeframe} data.",
    config={
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "max_tokens": 2000
    },
    labels=["research", "technical_analysis"]
)

# Use prompt in agent
prompt = langfuse.get_prompt("research_analyst_prompt")

# Compile with variables
compiled_prompt = prompt.compile(
    ticker="AAPL",
    perspective="technical",
    timeframe="daily"
)

# Use in LLM call (automatically tracked)
response = llm.invoke(compiled_prompt, config={"callbacks": [langfuse_handler]})
```

## Alerting

### Set Up Alerts in LangFuse Dashboard

LangFuse supports alerting based on:
- Error rate thresholds
- Latency thresholds
- Cost thresholds
- Custom score thresholds

### Custom Alert Logic

```python
class LangFuseAlertManager:
    """Custom alerting based on LangFuse data"""
    
    def __init__(self):
        self.langfuse = Langfuse()
    
    def check_and_alert(self):
        """Check metrics and send alerts"""
        
        monitor = AgentHealthMonitor()
        failing_agents = monitor.get_failing_agents()
        
        for agent_info in failing_agents:
            agent = agent_info["agent"]
            health = agent_info["health"]
            
            if health["status"] == "critical":
                self.send_alert(
                    severity="critical",
                    message=f"Agent {agent} is CRITICAL: {health['success_rate']:.1%} success rate",
                    data=health
                )
            elif health["status"] == "degraded":
                self.send_alert(
                    severity="warning",
                    message=f"Agent {agent} is DEGRADED: {health['success_rate']:.1%} success rate",
                    data=health
                )
    
    def send_alert(self, severity: str, message: str, data: dict):
        """Send alert via configured channels"""
        # Integrate with Slack, PagerDuty, etc.
        logger.warning(f"[{severity.upper()}] {message}")
        # slack.send_message(channel="#alerts", text=message)
```

## User Feedback Collection

### Collect Human Feedback on Strategies

```python
from langfuse import Langfuse

langfuse = Langfuse()

# After strategy is reviewed by human
def collect_strategy_feedback(
    trace_id: str,
    rating: int,  # 1-5
    comment: str
):
    """Collect human feedback on strategy"""
    
    langfuse.score(
        trace_id=trace_id,
        name="human_rating",
        value=rating,
        comment=comment
    )
```

### Analyze Feedback

```python
def analyze_strategy_feedback(ticker: str) -> dict:
    """Analyze human feedback for strategies"""
    
    langfuse = Langfuse()
    
    # Get all traces with human ratings
    traces = langfuse.get_traces(
        tags=[ticker, "strategy"],
        has_scores=True
    )
    
    ratings = []
    for trace in traces:
        human_rating = next(
            (s.value for s in trace.scores if s.name == "human_rating"),
            None
        )
        if human_rating:
            ratings.append(human_rating)
    
    if not ratings:
        return {"message": "No feedback yet"}
    
    return {
        "avg_rating": sum(ratings) / len(ratings),
        "total_ratings": len(ratings),
        "rating_distribution": {
            i: ratings.count(i) for i in range(1, 6)
        }
    }
```

## Self-Hosting LangFuse

For production deployments, self-host LangFuse:

```bash
# Docker Compose
docker-compose up -d
```

`docker-compose.yml`:

```yaml
version: '3.8'

services:
  langfuse-server:
    image: langfuse/langfuse:latest
    ports:
      - "3000:3000"
    environment:
      - DATABASE_URL=postgresql://user:password@postgres:5432/langfuse
      - NEXTAUTH_SECRET=your-secret-key
      - NEXTAUTH_URL=http://localhost:3000
    depends_on:
      - postgres

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=langfuse
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

Update `.env`:

```bash
LANGFUSE_BASE_URL=http://localhost:3000
```

## Benefits vs. Custom Prometheus/Grafana

| Feature | LangFuse | Custom Prometheus/Grafana |
|---------|----------|---------------------------|
| **Setup Time** | 5 minutes | 2-3 days |
| **LLM-Specific** | ✅ Native | ❌ Generic metrics |
| **Trace Visualization** | ✅ Built-in | ❌ Need custom UI |
| **Cost Tracking** | ✅ Automatic | ❌ Manual calculation |
| **Prompt Management** | ✅ Built-in | ❌ Not supported |
| **User Feedback** | ✅ Built-in | ❌ Need custom solution |
| **Nested Traces** | ✅ Automatic | ❌ Complex to implement |
| **Maintenance** | ✅ Low | ❌ High |

## Implementation Checklist

### Must Have (Phase 2)
- [ ] Install LangFuse SDK
- [ ] Set up LangFuse account (cloud or self-hosted)
- [ ] Add `CallbackHandler` to LangGraph workflow
- [ ] Configure environment variables
- [ ] Test trace capture for one workflow

### Should Have (Phase 3)
- [ ] Add custom metadata and tags
- [ ] Implement health monitoring
- [ ] Set up alerting
- [ ] Create prompt templates
- [ ] Implement user feedback collection

### Nice to Have (Phase 4)
- [ ] Self-host LangFuse
- [ ] Create custom dashboards
- [ ] Integrate with Slack/PagerDuty
- [ ] Implement A/B testing with LangFuse datasets

## Success Metrics

1. **Coverage**: 100% of LLM calls traced
2. **Overhead**: < 10ms latency per trace
3. **Visibility**: All 23 agents visible in dashboard
4. **Cost Tracking**: 100% accurate cost attribution
5. **Alerting**: < 5 minute time-to-alert for failures

---

**Document**: Agent Monitoring with LangFuse  
**Created**: 2026-01-18  
**Status**: Design Complete  
**Replaces**: AGENT_MONITORING.md (Prometheus/Grafana approach)
