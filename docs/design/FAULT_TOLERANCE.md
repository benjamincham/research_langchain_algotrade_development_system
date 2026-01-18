# Fault Tolerance & Recovery Patterns

## Overview

This document defines fault tolerance and recovery patterns for the agentic system to ensure robustness and reliability in production environments.

## Goals

1. **Resilience**: System continues operating despite agent failures
2. **Graceful Degradation**: Reduce functionality rather than complete failure
3. **Automatic Recovery**: Self-healing without human intervention
4. **Data Integrity**: No data loss during failures
5. **Observability**: Clear visibility into failure modes

## Fault Tolerance Patterns

### 1. Circuit Breaker Pattern

Prevents cascading failures by stopping calls to failing agents.

```python
from enum import Enum
from datetime import datetime, timedelta
from typing import Callable, Any, Optional
import asyncio

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject all requests
    HALF_OPEN = "half_open"  # Testing if recovered

class CircuitBreaker:
    """Circuit breaker for agent invocations"""
    
    def __init__(
        self,
        agent_id: str,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout_seconds: int = 60
    ):
        self.agent_id = agent_id
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout_seconds = timeout_seconds
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        
        # Check if circuit is open
        if self.state == CircuitState.OPEN:
            # Check if timeout has elapsed
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is OPEN for agent {self.agent_id}"
                )
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
            
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """Handle successful invocation"""
        self.failure_count = 0
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitState.CLOSED
                logger.info(f"Circuit breaker CLOSED for agent {self.agent_id}")
    
    def _on_failure(self):
        """Handle failed invocation"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.error(f"Circuit breaker OPEN for agent {self.agent_id}")
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True
        
        elapsed = datetime.now() - self.last_failure_time
        return elapsed.total_seconds() >= self.timeout_seconds

class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open"""
    pass
```

### 2. Retry with Exponential Backoff

Automatically retry failed operations with increasing delays.

```python
import asyncio
from typing import Callable, Any, Type
import random

class RetryPolicy:
    """Retry policy with exponential backoff"""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number"""
        delay = min(
            self.base_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        
        if self.jitter:
            # Add random jitter to prevent thundering herd
            delay *= (0.5 + random.random())
        
        return delay
    
    async def execute(
        self,
        func: Callable,
        *args,
        retry_on: tuple = (Exception,),
        **kwargs
    ) -> Any:
        """Execute function with retry logic"""
        last_exception = None
        
        for attempt in range(self.max_attempts):
            try:
                result = await func(*args, **kwargs)
                
                if attempt > 0:
                    logger.info(f"Retry succeeded on attempt {attempt + 1}")
                
                return result
                
            except retry_on as e:
                last_exception = e
                
                if attempt < self.max_attempts - 1:
                    delay = self.calculate_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"All {self.max_attempts} attempts failed. "
                        f"Last error: {e}"
                    )
        
        raise last_exception

# Usage example
retry_policy = RetryPolicy(max_attempts=3, base_delay=1.0)

async def call_agent_with_retry(agent_func, *args, **kwargs):
    return await retry_policy.execute(
        agent_func,
        *args,
        retry_on=(TimeoutError, ConnectionError),
        **kwargs
    )
```

### 3. Fallback Agents

Use backup agents when primary agents fail.

```python
from typing import List, Callable, Any

class FallbackChain:
    """Chain of fallback agents"""
    
    def __init__(self, primary: Callable, fallbacks: List[Callable]):
        self.primary = primary
        self.fallbacks = fallbacks
    
    async def invoke(self, *args, **kwargs) -> Any:
        """Try primary, then fallbacks in order"""
        agents = [self.primary] + self.fallbacks
        last_exception = None
        
        for i, agent in enumerate(agents):
            try:
                result = await agent(*args, **kwargs)
                
                if i > 0:
                    logger.warning(
                        f"Primary agent failed, fallback #{i} succeeded"
                    )
                
                return result
                
            except Exception as e:
                last_exception = e
                logger.warning(f"Agent {i} failed: {e}")
                
                if i < len(agents) - 1:
                    logger.info(f"Trying fallback agent #{i + 1}...")
        
        raise FallbackExhaustedError(
            f"All agents failed. Last error: {last_exception}"
        )

class FallbackExhaustedError(Exception):
    """Raised when all fallback agents fail"""
    pass

# Example: Fallback for research subagent
async def primary_price_action_analyst(ticker: str):
    # Primary implementation using advanced LLM
    return await analyze_with_gpt4(ticker)

async def fallback_price_action_analyst(ticker: str):
    # Fallback using cheaper LLM
    return await analyze_with_gpt4_mini(ticker)

async def simple_price_action_analyst(ticker: str):
    # Simple rule-based fallback
    return simple_technical_analysis(ticker)

price_action_agent = FallbackChain(
    primary=primary_price_action_analyst,
    fallbacks=[
        fallback_price_action_analyst,
        simple_price_action_analyst
    ]
)
```

### 4. Graceful Degradation

Continue with reduced functionality when agents fail.

```python
from typing import Optional, List

class GracefulDegradationManager:
    """Manages graceful degradation of system functionality"""
    
    def __init__(self):
        self.degradation_level = 0  # 0 = full, 1 = degraded, 2 = minimal
    
    async def research_swarm_with_degradation(
        self,
        ticker: str,
        required_agents: List[str],
        optional_agents: List[str]
    ) -> ResearchReport:
        """Execute research swarm with graceful degradation"""
        
        findings = []
        failed_agents = []
        
        # Try required agents
        for agent_id in required_agents:
            try:
                finding = await self.invoke_agent(agent_id, ticker)
                findings.append(finding)
            except Exception as e:
                failed_agents.append(agent_id)
                logger.error(f"Required agent {agent_id} failed: {e}")
        
        # If too many required agents failed, raise error
        if len(failed_agents) > len(required_agents) * 0.3:  # 30% threshold
            raise CriticalDegradationError(
                f"Too many required agents failed: {failed_agents}"
            )
        
        # Try optional agents (best effort)
        for agent_id in optional_agents:
            try:
                finding = await self.invoke_agent(agent_id, ticker)
                findings.append(finding)
            except Exception as e:
                logger.warning(f"Optional agent {agent_id} failed: {e}")
                # Continue without this agent
        
        # Determine degradation level
        total_agents = len(required_agents) + len(optional_agents)
        success_rate = len(findings) / total_agents
        
        if success_rate < 0.5:
            self.degradation_level = 2  # Minimal functionality
        elif success_rate < 0.8:
            self.degradation_level = 1  # Degraded functionality
        else:
            self.degradation_level = 0  # Full functionality
        
        # Synthesize findings with degradation notice
        report = self.synthesize_findings(findings)
        report.degradation_level = self.degradation_level
        report.failed_agents = failed_agents
        
        return report

class CriticalDegradationError(Exception):
    """Raised when degradation is too severe to continue"""
    pass
```

### 5. Agent Replication

Run multiple instances of critical agents for redundancy.

```python
import asyncio
from typing import List, Any, Callable

class ReplicatedAgent:
    """Runs multiple instances of an agent for redundancy"""
    
    def __init__(
        self,
        agent_func: Callable,
        num_replicas: int = 3,
        quorum_size: int = 2
    ):
        self.agent_func = agent_func
        self.num_replicas = num_replicas
        self.quorum_size = quorum_size
    
    async def invoke(self, *args, **kwargs) -> Any:
        """Invoke all replicas and return consensus result"""
        
        # Start all replicas
        tasks = [
            self.agent_func(*args, **kwargs)
            for _ in range(self.num_replicas)
        ]
        
        # Wait for quorum
        results = []
        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                results.append(result)
                
                # Return as soon as we have quorum
                if len(results) >= self.quorum_size:
                    return self._consensus(results)
                    
            except Exception as e:
                logger.warning(f"Replica failed: {e}")
        
        # If we didn't get quorum, raise error
        if len(results) < self.quorum_size:
            raise QuorumNotReachedError(
                f"Only {len(results)} of {self.quorum_size} replicas succeeded"
            )
        
        return self._consensus(results)
    
    def _consensus(self, results: List[Any]) -> Any:
        """Determine consensus from multiple results"""
        # For now, return first result
        # Could implement voting, averaging, etc.
        return results[0]

class QuorumNotReachedError(Exception):
    """Raised when quorum is not reached"""
    pass
```

### 6. Timeout Protection

Prevent agents from hanging indefinitely.

```python
import asyncio
from typing import Callable, Any

class TimeoutProtection:
    """Protects against hanging agents"""
    
    def __init__(self, timeout_seconds: int = 30):
        self.timeout_seconds = timeout_seconds
    
    async def call_with_timeout(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute function with timeout"""
        try:
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.timeout_seconds
            )
            return result
            
        except asyncio.TimeoutError:
            logger.error(
                f"Agent timed out after {self.timeout_seconds}s"
            )
            raise AgentTimeoutError(
                f"Agent did not respond within {self.timeout_seconds}s"
            )

class AgentTimeoutError(Exception):
    """Raised when agent times out"""
    pass
```

## Integrated Fault-Tolerant Agent

Combining all patterns:

```python
class FaultTolerantAgent:
    """Agent with all fault tolerance patterns integrated"""
    
    def __init__(
        self,
        agent_id: str,
        primary_func: Callable,
        fallback_funcs: List[Callable] = None,
        timeout_seconds: int = 30,
        max_retries: int = 3,
        circuit_breaker_threshold: int = 5
    ):
        self.agent_id = agent_id
        
        # Initialize components
        self.circuit_breaker = CircuitBreaker(
            agent_id=agent_id,
            failure_threshold=circuit_breaker_threshold
        )
        self.retry_policy = RetryPolicy(max_attempts=max_retries)
        self.timeout_protection = TimeoutProtection(timeout_seconds=timeout_seconds)
        
        # Set up fallback chain
        if fallback_funcs:
            self.agent = FallbackChain(primary_func, fallback_funcs)
        else:
            self.agent = primary_func
    
    async def invoke(self, *args, **kwargs) -> Any:
        """Invoke agent with full fault tolerance"""
        
        async def protected_call():
            # Timeout protection
            return await self.timeout_protection.call_with_timeout(
                self.agent.invoke if isinstance(self.agent, FallbackChain) else self.agent,
                *args,
                **kwargs
            )
        
        # Circuit breaker + retry
        return await self.circuit_breaker.call(
            lambda: self.retry_policy.execute(protected_call),
        )

# Usage example
fault_tolerant_price_action = FaultTolerantAgent(
    agent_id="price_action_analyst",
    primary_func=primary_price_action_analyst,
    fallback_funcs=[
        fallback_price_action_analyst,
        simple_price_action_analyst
    ],
    timeout_seconds=30,
    max_retries=3,
    circuit_breaker_threshold=5
)

# Invoke with full protection
result = await fault_tolerant_price_action.invoke(ticker="AAPL")
```

## State Persistence & Recovery

### Checkpoint Management

```python
from typing import Dict, Any
import json
from datetime import datetime

class CheckpointManager:
    """Manages workflow checkpoints for recovery"""
    
    def __init__(self, checkpoint_dir: str = "checkpoints/"):
        self.checkpoint_dir = checkpoint_dir
    
    def save_checkpoint(
        self,
        workflow_id: str,
        state: Dict[str, Any],
        phase: str
    ):
        """Save workflow state checkpoint"""
        checkpoint = {
            "workflow_id": workflow_id,
            "timestamp": datetime.now().isoformat(),
            "phase": phase,
            "state": state
        }
        
        checkpoint_file = f"{self.checkpoint_dir}/{workflow_id}_latest.json"
        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint, f, indent=2)
        
        logger.info(f"Checkpoint saved for workflow {workflow_id} at phase {phase}")
    
    def load_checkpoint(self, workflow_id: str) -> Dict[str, Any]:
        """Load latest checkpoint for workflow"""
        checkpoint_file = f"{self.checkpoint_dir}/{workflow_id}_latest.json"
        
        try:
            with open(checkpoint_file, "r") as f:
                checkpoint = json.load(f)
            
            logger.info(f"Checkpoint loaded for workflow {workflow_id}")
            return checkpoint
            
        except FileNotFoundError:
            raise CheckpointNotFoundError(
                f"No checkpoint found for workflow {workflow_id}"
            )
    
    def resume_from_checkpoint(self, workflow_id: str):
        """Resume workflow from checkpoint"""
        checkpoint = self.load_checkpoint(workflow_id)
        
        # Resume from saved phase
        phase = checkpoint["phase"]
        state = checkpoint["state"]
        
        logger.info(f"Resuming workflow {workflow_id} from phase {phase}")
        
        # Continue workflow from this point
        return phase, state

class CheckpointNotFoundError(Exception):
    """Raised when checkpoint is not found"""
    pass
```

## Health Checks

```python
class AgentHealthChecker:
    """Performs health checks on agents"""
    
    async def check_agent_health(self, agent_id: str) -> bool:
        """Check if agent is healthy"""
        try:
            # Simple ping test
            result = await self.ping_agent(agent_id, timeout=5)
            return result is not None
            
        except Exception as e:
            logger.error(f"Health check failed for {agent_id}: {e}")
            return False
    
    async def check_all_agents(self) -> Dict[str, bool]:
        """Check health of all agents"""
        health_status = {}
        
        for agent_id in self.get_all_agent_ids():
            health_status[agent_id] = await self.check_agent_health(agent_id)
        
        return health_status
    
    async def wait_for_healthy(
        self,
        agent_id: str,
        timeout_seconds: int = 60
    ):
        """Wait for agent to become healthy"""
        start_time = datetime.now()
        
        while True:
            if await self.check_agent_health(agent_id):
                return True
            
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed >= timeout_seconds:
                raise AgentUnhealthyError(
                    f"Agent {agent_id} did not become healthy within {timeout_seconds}s"
                )
            
            await asyncio.sleep(5)

class AgentUnhealthyError(Exception):
    """Raised when agent is unhealthy"""
    pass
```

## Integration with LangGraph

```python
from langgraph.graph import StateGraph

def create_fault_tolerant_graph():
    """Create LangGraph with fault tolerance"""
    
    graph = StateGraph(WorkflowState)
    
    # Wrap all nodes with fault tolerance
    graph.add_node(
        "research_swarm",
        wrap_with_fault_tolerance(research_swarm_node)
    )
    
    graph.add_node(
        "strategy_dev",
        wrap_with_fault_tolerance(strategy_dev_node)
    )
    
    graph.add_node(
        "quality_gate",
        wrap_with_fault_tolerance(quality_gate_node)
    )
    
    # Add error handling edges
    graph.add_conditional_edges(
        "research_swarm",
        route_with_error_handling,
        {
            "success": "strategy_dev",
            "retry": "research_swarm",
            "fail": END
        }
    )
    
    return graph

def wrap_with_fault_tolerance(node_func: Callable) -> Callable:
    """Wrap node function with fault tolerance"""
    
    async def wrapped(state: WorkflowState) -> WorkflowState:
        checkpoint_manager = CheckpointManager()
        
        try:
            # Save checkpoint before executing
            checkpoint_manager.save_checkpoint(
                workflow_id=state["workflow_id"],
                state=state,
                phase=node_func.__name__
            )
            
            # Execute with fault tolerance
            result = await fault_tolerant_execute(node_func, state)
            return result
            
        except Exception as e:
            logger.error(f"Node {node_func.__name__} failed: {e}")
            
            # Attempt recovery
            if should_retry(e):
                return {"retry": True}
            else:
                return {"failed": True, "error": str(e)}
    
    return wrapped
```

## Fault Tolerance Checklist

### Must Have (Phase 2)
- [ ] Implement circuit breaker for all agents
- [ ] Add retry logic with exponential backoff
- [ ] Implement timeout protection
- [ ] Add checkpoint management
- [ ] Create health check endpoints

### Should Have (Phase 3)
- [ ] Implement fallback agents for critical components
- [ ] Add graceful degradation logic
- [ ] Implement agent replication for critical agents
- [ ] Add recovery from checkpoints
- [ ] Create fault tolerance monitoring

### Nice to Have (Phase 4)
- [ ] Implement quorum-based consensus
- [ ] Add self-healing capabilities
- [ ] Create fault injection testing
- [ ] Implement chaos engineering tests

## Success Metrics

1. **Availability**: > 99.9% uptime despite agent failures
2. **Recovery Time**: < 60 seconds to recover from failures
3. **Data Loss**: 0% data loss during failures
4. **Graceful Degradation**: System continues with reduced functionality
5. **Observability**: 100% of failures logged and alerted

---

**Document**: Fault Tolerance & Recovery Patterns  
**Created**: 2026-01-18  
**Status**: Design Complete
