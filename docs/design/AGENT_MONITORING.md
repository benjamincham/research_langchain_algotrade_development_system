# Agent Monitoring & Evaluation Framework

## Overview

This document defines a comprehensive monitoring and evaluation framework for all 23 agents in the system. The framework enables performance tracking, health monitoring, explainability, and continuous improvement.

## Goals

1. **Performance Tracking**: Measure which agents provide valuable insights
2. **Health Monitoring**: Detect degraded or failing agents
3. **Explainability**: Understand why agents made specific decisions
4. **Continuous Improvement**: Identify opportunities for agent optimization
5. **Resource Optimization**: Track compute costs and optimize agent usage

## Architecture

### 4-Layer Monitoring Stack

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 4: Visualization & Alerting                          │
│  - Grafana dashboards                                       │
│  - Alert manager                                            │
│  - Explainability UI                                        │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: Analytics & Aggregation                           │
│  - Agent performance analyzer                               │
│  - Trend detection                                          │
│  - Anomaly detection                                        │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│  Layer 2: Metrics Collection & Storage                      │
│  - Metrics collector                                        │
│  - Time-series database (Prometheus)                        │
│  - Event log storage                                        │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: Agent Instrumentation                             │
│  - Decorators for automatic metric collection               │
│  - Context managers for timing                              │
│  - Logging integration                                      │
└─────────────────────────────────────────────────────────────┘
```

## Metrics Schema

### Agent Performance Metrics

```python
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime

class AgentInvocationMetrics(BaseModel):
    """Metrics for a single agent invocation"""
    agent_id: str
    agent_type: str  # "research_subagent", "synthesizer", "quality_gate", etc.
    invocation_id: str
    timestamp: datetime
    
    # Performance metrics
    duration_ms: float
    token_count: int
    cost_usd: float
    
    # Quality metrics
    success: bool
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    
    # Output metrics
    output_length: int
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Context
    workflow_id: str
    iteration_number: int
    ticker: str

class AgentAggregateMetrics(BaseModel):
    """Aggregate metrics for an agent over time"""
    agent_id: str
    agent_type: str
    time_window: str  # "1h", "24h", "7d", "30d"
    
    # Invocation stats
    total_invocations: int
    successful_invocations: int
    failed_invocations: int
    success_rate: float = Field(..., ge=0.0, le=1.0)
    
    # Performance stats
    avg_duration_ms: float
    p50_duration_ms: float
    p95_duration_ms: float
    p99_duration_ms: float
    
    # Cost stats
    total_cost_usd: float
    avg_cost_per_invocation: float
    
    # Quality stats
    avg_confidence_score: Optional[float] = None
    insight_quality_score: Optional[float] = None  # From downstream success
    
    # Health status
    health_status: str  # "healthy", "degraded", "failing"
    last_success_timestamp: datetime
    last_failure_timestamp: Optional[datetime] = None

class AgentInsightQuality(BaseModel):
    """Measures quality of agent insights"""
    agent_id: str
    finding_id: str
    
    # Downstream impact
    used_in_strategy: bool
    strategy_passed_quality_gate: bool
    strategy_sharpe_ratio: Optional[float] = None
    
    # Human feedback (optional)
    human_rating: Optional[int] = Field(None, ge=1, le=5)
    human_feedback: Optional[str] = None
    
    # Calculated quality score
    quality_score: float = Field(..., ge=0.0, le=1.0)
```

## Implementation

### Layer 1: Agent Instrumentation

```python
import time
import functools
from contextlib import contextmanager
from typing import Callable, Any

class AgentMonitor:
    """Singleton for collecting agent metrics"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.metrics_buffer = []
        return cls._instance
    
    def log_invocation(self, metrics: AgentInvocationMetrics):
        """Log a single agent invocation"""
        self.metrics_buffer.append(metrics)
        
        # Flush to storage if buffer is full
        if len(self.metrics_buffer) >= 100:
            self.flush()
    
    def flush(self):
        """Flush metrics to storage"""
        # Write to time-series database
        # Write to event log
        self.metrics_buffer.clear()

def monitor_agent(agent_id: str, agent_type: str):
    """Decorator to automatically monitor agent invocations"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            monitor = AgentMonitor()
            invocation_id = generate_invocation_id()
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                
                # Extract metrics from result
                metrics = AgentInvocationMetrics(
                    agent_id=agent_id,
                    agent_type=agent_type,
                    invocation_id=invocation_id,
                    timestamp=datetime.now(),
                    duration_ms=duration_ms,
                    token_count=estimate_tokens(result),
                    cost_usd=calculate_cost(result),
                    success=True,
                    output_length=len(str(result)),
                    confidence_score=extract_confidence(result),
                    workflow_id=kwargs.get("workflow_id"),
                    iteration_number=kwargs.get("iteration_number"),
                    ticker=kwargs.get("ticker")
                )
                
                monitor.log_invocation(metrics)
                return result
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                
                metrics = AgentInvocationMetrics(
                    agent_id=agent_id,
                    agent_type=agent_type,
                    invocation_id=invocation_id,
                    timestamp=datetime.now(),
                    duration_ms=duration_ms,
                    token_count=0,
                    cost_usd=0.0,
                    success=False,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    output_length=0,
                    workflow_id=kwargs.get("workflow_id"),
                    iteration_number=kwargs.get("iteration_number"),
                    ticker=kwargs.get("ticker")
                )
                
                monitor.log_invocation(metrics)
                raise
        
        return wrapper
    return decorator

# Usage example
@monitor_agent(agent_id="price_action_analyst", agent_type="research_subagent")
async def price_action_analyst(ticker: str, workflow_id: str, iteration_number: int):
    # Agent logic here
    findings = analyze_price_action(ticker)
    return findings
```

### Layer 2: Metrics Collection & Storage

```python
from prometheus_client import Counter, Histogram, Gauge
import json

# Prometheus metrics
agent_invocations_total = Counter(
    'agent_invocations_total',
    'Total number of agent invocations',
    ['agent_id', 'agent_type', 'status']
)

agent_duration_seconds = Histogram(
    'agent_duration_seconds',
    'Agent invocation duration in seconds',
    ['agent_id', 'agent_type']
)

agent_cost_usd = Counter(
    'agent_cost_usd_total',
    'Total cost of agent invocations in USD',
    ['agent_id', 'agent_type']
)

agent_health_status = Gauge(
    'agent_health_status',
    'Agent health status (1=healthy, 0.5=degraded, 0=failing)',
    ['agent_id', 'agent_type']
)

class MetricsCollector:
    """Collects and stores agent metrics"""
    
    def __init__(self, storage_path: str = "metrics/"):
        self.storage_path = storage_path
    
    def record_invocation(self, metrics: AgentInvocationMetrics):
        """Record metrics to Prometheus and event log"""
        # Update Prometheus metrics
        status = "success" if metrics.success else "failure"
        agent_invocations_total.labels(
            agent_id=metrics.agent_id,
            agent_type=metrics.agent_type,
            status=status
        ).inc()
        
        agent_duration_seconds.labels(
            agent_id=metrics.agent_id,
            agent_type=metrics.agent_type
        ).observe(metrics.duration_ms / 1000)
        
        agent_cost_usd.labels(
            agent_id=metrics.agent_id,
            agent_type=metrics.agent_type
        ).inc(metrics.cost_usd)
        
        # Write to event log (JSONL)
        log_file = f"{self.storage_path}/agent_events.jsonl"
        with open(log_file, "a") as f:
            f.write(metrics.model_dump_json() + "\n")
    
    def get_aggregate_metrics(
        self, 
        agent_id: str, 
        time_window: str = "24h"
    ) -> AgentAggregateMetrics:
        """Calculate aggregate metrics for an agent"""
        # Query Prometheus for aggregate data
        # Or read from event log and calculate
        pass
```

### Layer 3: Analytics & Aggregation

```python
class AgentPerformanceAnalyzer:
    """Analyzes agent performance and detects issues"""
    
    def analyze_agent_health(self, agent_id: str) -> str:
        """Determine agent health status"""
        metrics = self.get_recent_metrics(agent_id, window="1h")
        
        # Check success rate
        if metrics.success_rate < 0.5:
            return "failing"
        elif metrics.success_rate < 0.9:
            return "degraded"
        
        # Check latency
        if metrics.p95_duration_ms > 30000:  # 30 seconds
            return "degraded"
        
        # Check cost anomalies
        if metrics.avg_cost_per_invocation > self.get_baseline_cost(agent_id) * 2:
            return "degraded"
        
        return "healthy"
    
    def detect_performance_regression(self, agent_id: str) -> bool:
        """Detect if agent performance is degrading over time"""
        current_metrics = self.get_recent_metrics(agent_id, window="24h")
        baseline_metrics = self.get_baseline_metrics(agent_id)
        
        # Compare duration
        if current_metrics.avg_duration_ms > baseline_metrics.avg_duration_ms * 1.5:
            return True
        
        # Compare success rate
        if current_metrics.success_rate < baseline_metrics.success_rate * 0.9:
            return True
        
        return False
    
    def rank_agents_by_insight_quality(self) -> List[tuple]:
        """Rank agents by quality of insights"""
        agent_scores = {}
        
        for agent_id in self.get_all_agent_ids():
            insights = self.get_agent_insights(agent_id)
            quality_scores = [i.quality_score for i in insights]
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
            agent_scores[agent_id] = avg_quality
        
        return sorted(agent_scores.items(), key=lambda x: x[1], reverse=True)
    
    def calculate_insight_quality_score(
        self, 
        finding_id: str,
        strategy_result: Optional[BacktestMetrics] = None
    ) -> float:
        """Calculate quality score for an agent insight"""
        score = 0.0
        
        # Was the finding used in a strategy?
        if strategy_result is not None:
            score += 0.3
            
            # Did the strategy pass quality gates?
            if strategy_result.passed_quality_gate:
                score += 0.3
                
                # How good was the strategy?
                if strategy_result.sharpe_ratio > 1.5:
                    score += 0.4
                elif strategy_result.sharpe_ratio > 1.0:
                    score += 0.2
        
        return min(score, 1.0)
```

### Layer 4: Visualization & Alerting

```python
class AlertManager:
    """Manages alerts for agent health issues"""
    
    def check_and_alert(self):
        """Check all agents and send alerts if needed"""
        analyzer = AgentPerformanceAnalyzer()
        
        for agent_id in self.get_all_agent_ids():
            health = analyzer.analyze_agent_health(agent_id)
            
            if health == "failing":
                self.send_alert(
                    severity="critical",
                    message=f"Agent {agent_id} is FAILING (success rate < 50%)",
                    agent_id=agent_id
                )
            elif health == "degraded":
                self.send_alert(
                    severity="warning",
                    message=f"Agent {agent_id} is DEGRADED (success rate < 90% or high latency)",
                    agent_id=agent_id
                )
            
            # Check for performance regression
            if analyzer.detect_performance_regression(agent_id):
                self.send_alert(
                    severity="warning",
                    message=f"Agent {agent_id} performance has regressed",
                    agent_id=agent_id
                )
    
    def send_alert(self, severity: str, message: str, agent_id: str):
        """Send alert via configured channels"""
        # Send to Slack, email, PagerDuty, etc.
        logger.warning(f"[{severity.upper()}] {message}")
```

## Explainability

### Decision Logging

```python
class AgentDecisionLogger:
    """Logs agent decisions for explainability"""
    
    def log_decision(
        self,
        agent_id: str,
        decision: str,
        reasoning: str,
        evidence: Dict[str, Any],
        alternatives_considered: List[str]
    ):
        """Log an agent decision with full context"""
        decision_log = {
            "agent_id": agent_id,
            "timestamp": datetime.now().isoformat(),
            "decision": decision,
            "reasoning": reasoning,
            "evidence": evidence,
            "alternatives_considered": alternatives_considered
        }
        
        # Store in database for later retrieval
        self.store_decision(decision_log)
    
    def explain_decision(self, decision_id: str) -> str:
        """Generate human-readable explanation of a decision"""
        decision = self.get_decision(decision_id)
        
        explanation = f"""
        Agent: {decision['agent_id']}
        Decision: {decision['decision']}
        
        Reasoning:
        {decision['reasoning']}
        
        Evidence:
        {json.dumps(decision['evidence'], indent=2)}
        
        Alternatives Considered:
        {', '.join(decision['alternatives_considered'])}
        """
        
        return explanation
```

## A/B Testing Framework

```python
class AgentABTest:
    """Framework for A/B testing agent implementations"""
    
    def __init__(
        self,
        agent_id: str,
        variant_a: Callable,
        variant_b: Callable,
        traffic_split: float = 0.5
    ):
        self.agent_id = agent_id
        self.variant_a = variant_a
        self.variant_b = variant_b
        self.traffic_split = traffic_split
    
    async def invoke(self, *args, **kwargs):
        """Invoke agent with A/B testing"""
        import random
        
        # Randomly assign to variant
        use_variant_a = random.random() < self.traffic_split
        variant = "A" if use_variant_a else "B"
        
        # Track which variant was used
        kwargs["ab_test_variant"] = variant
        
        # Invoke appropriate variant
        if use_variant_a:
            result = await self.variant_a(*args, **kwargs)
        else:
            result = await self.variant_b(*args, **kwargs)
        
        # Log result with variant info
        self.log_ab_test_result(variant, result, *args, **kwargs)
        
        return result
    
    def get_ab_test_results(self) -> Dict[str, AgentAggregateMetrics]:
        """Get results for both variants"""
        return {
            "variant_a": self.get_variant_metrics("A"),
            "variant_b": self.get_variant_metrics("B")
        }
    
    def determine_winner(self) -> str:
        """Determine which variant performs better"""
        results = self.get_ab_test_results()
        
        # Compare success rates
        if results["variant_a"].success_rate > results["variant_b"].success_rate:
            return "A"
        elif results["variant_b"].success_rate > results["variant_a"].success_rate:
            return "B"
        
        # Compare insight quality
        if results["variant_a"].insight_quality_score > results["variant_b"].insight_quality_score:
            return "A"
        else:
            return "B"
```

## Integration with Existing System

### Update Agent Implementations

```python
# Before: No monitoring
async def price_action_analyst(ticker: str):
    findings = analyze_price_action(ticker)
    return findings

# After: With monitoring
@monitor_agent(agent_id="price_action_analyst", agent_type="research_subagent")
async def price_action_analyst(ticker: str, workflow_id: str, iteration_number: int):
    findings = analyze_price_action(ticker)
    
    # Log decision for explainability
    decision_logger = AgentDecisionLogger()
    decision_logger.log_decision(
        agent_id="price_action_analyst",
        decision=f"Identified {len(findings)} patterns",
        reasoning="Based on technical indicators and price action",
        evidence={"findings": findings},
        alternatives_considered=["No patterns found"]
    )
    
    return findings
```

### Dashboard Metrics

```yaml
# Grafana dashboard configuration
dashboards:
  - name: "Agent Performance Overview"
    panels:
      - title: "Agent Invocations (24h)"
        query: "sum(rate(agent_invocations_total[24h])) by (agent_id)"
      
      - title: "Agent Success Rate"
        query: "sum(rate(agent_invocations_total{status='success'}[1h])) / sum(rate(agent_invocations_total[1h]))"
      
      - title: "Agent Latency (P95)"
        query: "histogram_quantile(0.95, agent_duration_seconds)"
      
      - title: "Agent Cost (24h)"
        query: "sum(increase(agent_cost_usd_total[24h])) by (agent_id)"
      
      - title: "Agent Health Status"
        query: "agent_health_status"
      
      - title: "Insight Quality by Agent"
        query: "avg(agent_insight_quality_score) by (agent_id)"
```

## Monitoring Checklist

### Must Have (Phase 2)
- [ ] Implement `AgentMonitor` singleton
- [ ] Add `@monitor_agent` decorator to all agents
- [ ] Set up Prometheus metrics collection
- [ ] Implement JSONL event logging
- [ ] Create basic health check endpoint

### Should Have (Phase 3)
- [ ] Implement `AgentPerformanceAnalyzer`
- [ ] Set up Grafana dashboards
- [ ] Implement alert manager
- [ ] Add decision logging for explainability
- [ ] Calculate insight quality scores

### Nice to Have (Phase 4)
- [ ] Implement A/B testing framework
- [ ] Add human feedback collection
- [ ] Create explainability UI
- [ ] Implement automated performance regression detection

## Success Metrics

1. **Coverage**: 100% of agents instrumented with monitoring
2. **Latency Overhead**: < 5ms per agent invocation
3. **Storage**: < 100MB per day of metrics data
4. **Alert Accuracy**: < 5% false positive rate
5. **Explainability**: 100% of decisions logged with reasoning

---

**Document**: Agent Monitoring & Evaluation Framework  
**Created**: 2026-01-18  
**Status**: Design Complete
