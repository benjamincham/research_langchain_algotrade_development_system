# Fault Tolerance & Recovery with LangFuse Integration

## Overview

This document defines fault tolerance and recovery patterns for the agentic system with **LangFuse integration** for comprehensive observability of failures, retries, and recovery actions.

## Goals

1. **Resilience**: System continues operating despite agent failures
2. **Graceful Degradation**: Reduce functionality rather than complete failure
3. **Automatic Recovery**: Self-healing without human intervention
4. **Data Integrity**: No data loss during failures
5. **Observability**: Full visibility into failure modes via LangFuse

## Fault Tolerance Patterns with LangFuse

### 1. Circuit Breaker with LangFuse Tracking

```python
from enum import Enum
from datetime import datetime, timedelta
from typing import Callable, Any, Optional
import asyncio
from langfuse.decorators import observe, langfuse_context

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    """Circuit breaker with LangFuse observability"""
    
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
    
    @observe(name="circuit_breaker_call")
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        
        # Log circuit breaker state to LangFuse
        langfuse_context.update_current_observation(
            metadata={
                "agent_id": self.agent_id,
                "circuit_state": self.state.value,
                "failure_count": self.failure_count,
                "success_count": self.success_count
            }
        )
        
        # Check if circuit is open
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                
                # Log state transition
                langfuse_context.update_current_observation(
                    metadata={"state_transition": "OPEN -> HALF_OPEN"}
                )
            else:
                # Log rejection
                langfuse_context.update_current_observation(
                    level="ERROR",
                    status_message=f"Circuit breaker is OPEN for agent {self.agent_id}"
                )
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is OPEN for agent {self.agent_id}"
                )
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
            
        except Exception as e:
            self._on_failure()
            
            # Log failure to LangFuse
            langfuse_context.update_current_observation(
                level="ERROR",
                status_message=f"Circuit breaker recorded failure: {str(e)}",
                metadata={
                    "failure_count": self.failure_count,
                    "circuit_state": self.state.value
                }
            )
            raise
    
    def _on_success(self):
        """Handle successful invocation"""
        self.failure_count = 0
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitState.CLOSED
                logger.info(f"Circuit breaker CLOSED for agent {self.agent_id}")
                
                # Log recovery to LangFuse
                langfuse_context.update_current_observation(
                    metadata={"state_transition": "HALF_OPEN -> CLOSED"}
                )
    
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
        
        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        return elapsed >= self.timeout_seconds

class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open"""
    pass
```

### 2. Retry with Exponential Backoff + LangFuse

```python
import asyncio
from typing import Callable, Any
import random
from langfuse.decorators import observe, langfuse_context

class RetryPolicy:
    """Retry policy with LangFuse tracking"""
    
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
            delay *= (0.5 + random.random())
        
        return delay
    
    @observe(name="retry_execution")
    async def execute(
        self,
        func: Callable,
        *args,
        retry_on: tuple = (Exception,),
        **kwargs
    ) -> Any:
        """Execute function with retry logic and LangFuse tracking"""
        
        last_exception = None
        retry_history = []
        
        for attempt in range(self.max_attempts):
            try:
                # Log attempt
                langfuse_context.update_current_observation(
                    metadata={
                        "attempt": attempt + 1,
                        "max_attempts": self.max_attempts,
                        "retry_history": retry_history
                    }
                )
                
                result = await func(*args, **kwargs)
                
                if attempt > 0:
                    # Log successful retry
                    langfuse_context.update_current_observation(
                        metadata={
                            "retry_succeeded": True,
                            "attempts_needed": attempt + 1
                        }
                    )
                    logger.info(f"Retry succeeded on attempt {attempt + 1}")
                
                return result
                
            except retry_on as e:
                last_exception = e
                
                retry_history.append({
                    "attempt": attempt + 1,
                    "error": str(e),
                    "error_type": type(e).__name__
                })
                
                if attempt < self.max_attempts - 1:
                    delay = self.calculate_delay(attempt)
                    
                    # Log retry
                    langfuse_context.update_current_observation(
                        level="WARNING",
                        status_message=f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s",
                        metadata={
                            "error": str(e),
                            "delay_seconds": delay
                        }
                    )
                    
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    # Log final failure
                    langfuse_context.update_current_observation(
                        level="ERROR",
                        status_message=f"All {self.max_attempts} attempts failed",
                        metadata={
                            "retry_history": retry_history,
                            "final_error": str(e)
                        }
                    )
                    
                    logger.error(
                        f"All {self.max_attempts} attempts failed. "
                        f"Last error: {e}"
                    )
        
        raise last_exception
```

### 3. Fallback Agents with LangFuse

```python
from typing import List, Callable, Any
from langfuse.decorators import observe, langfuse_context

class FallbackChain:
    """Chain of fallback agents with LangFuse tracking"""
    
    def __init__(self, primary: Callable, fallbacks: List[Callable]):
        self.primary = primary
        self.fallbacks = fallbacks
    
    @observe(name="fallback_chain")
    async def invoke(self, *args, **kwargs) -> Any:
        """Try primary, then fallbacks in order"""
        agents = [self.primary] + self.fallbacks
        last_exception = None
        fallback_history = []
        
        for i, agent in enumerate(agents):
            agent_type = "primary" if i == 0 else f"fallback_{i}"
            
            try:
                # Log attempt
                langfuse_context.update_current_observation(
                    metadata={
                        "agent_index": i,
                        "agent_type": agent_type,
                        "total_agents": len(agents),
                        "fallback_history": fallback_history
                    }
                )
                
                result = await agent(*args, **kwargs)
                
                if i > 0:
                    # Log fallback success
                    langfuse_context.update_current_observation(
                        metadata={
                            "fallback_succeeded": True,
                            "fallback_level": i,
                            "primary_failed": True
                        }
                    )
                    logger.warning(
                        f"Primary agent failed, fallback #{i} succeeded"
                    )
                
                return result
                
            except Exception as e:
                last_exception = e
                
                fallback_history.append({
                    "agent_index": i,
                    "agent_type": agent_type,
                    "error": str(e),
                    "error_type": type(e).__name__
                })
                
                # Log failure
                langfuse_context.update_current_observation(
                    level="WARNING" if i < len(agents) - 1 else "ERROR",
                    status_message=f"Agent {i} ({agent_type}) failed: {str(e)}",
                    metadata={"error": str(e)}
                )
                
                logger.warning(f"Agent {i} failed: {e}")
                
                if i < len(agents) - 1:
                    logger.info(f"Trying fallback agent #{i + 1}...")
        
        # All agents failed
        langfuse_context.update_current_observation(
            level="ERROR",
            status_message="All fallback agents exhausted",
            metadata={"fallback_history": fallback_history}
        )
        
        raise FallbackExhaustedError(
            f"All agents failed. Last error: {last_exception}"
        )

class FallbackExhaustedError(Exception):
    """Raised when all fallback agents fail"""
    pass
```

### 4. Graceful Degradation with LangFuse

```python
from typing import Optional, List
from langfuse.decorators import observe, langfuse_context

class GracefulDegradationManager:
    """Manages graceful degradation with LangFuse tracking"""
    
    def __init__(self):
        self.degradation_level = 0
    
    @observe(name="research_swarm_with_degradation")
    async def research_swarm_with_degradation(
        self,
        ticker: str,
        required_agents: List[str],
        optional_agents: List[str]
    ) -> ResearchReport:
        """Execute research swarm with graceful degradation"""
        
        findings = []
        failed_agents = []
        
        # Log initial state
        langfuse_context.update_current_observation(
            metadata={
                "ticker": ticker,
                "required_agents_count": len(required_agents),
                "optional_agents_count": len(optional_agents),
                "total_agents": len(required_agents) + len(optional_agents)
            }
        )
        
        # Try required agents
        for agent_id in required_agents:
            try:
                finding = await self.invoke_agent(agent_id, ticker)
                findings.append(finding)
            except Exception as e:
                failed_agents.append(agent_id)
                logger.error(f"Required agent {agent_id} failed: {e}")
        
        # Check if too many required agents failed
        failure_rate = len(failed_agents) / len(required_agents)
        if failure_rate > 0.3:  # 30% threshold
            langfuse_context.update_current_observation(
                level="ERROR",
                status_message="Too many required agents failed",
                metadata={
                    "failed_agents": failed_agents,
                    "failure_rate": failure_rate,
                    "threshold": 0.3
                }
            )
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
        
        # Determine degradation level
        total_agents = len(required_agents) + len(optional_agents)
        success_rate = len(findings) / total_agents
        
        if success_rate < 0.5:
            self.degradation_level = 2  # Minimal
        elif success_rate < 0.8:
            self.degradation_level = 1  # Degraded
        else:
            self.degradation_level = 0  # Full
        
        # Log degradation status
        langfuse_context.update_current_observation(
            metadata={
                "degradation_level": self.degradation_level,
                "success_rate": success_rate,
                "successful_agents": len(findings),
                "failed_agents": failed_agents
            }
        )
        
        # Score the degradation
        langfuse_context.score_current_observation(
            name="degradation_level",
            value=self.degradation_level,
            comment=f"Success rate: {success_rate:.1%}"
        )
        
        # Synthesize findings
        report = self.synthesize_findings(findings)
        report.degradation_level = self.degradation_level
        report.failed_agents = failed_agents
        
        return report

class CriticalDegradationError(Exception):
    """Raised when degradation is too severe"""
    pass
```

### 5. Timeout Protection with LangFuse

```python
import asyncio
from typing import Callable, Any
from langfuse.decorators import observe, langfuse_context

class TimeoutProtection:
    """Protects against hanging agents with LangFuse tracking"""
    
    def __init__(self, timeout_seconds: int = 30):
        self.timeout_seconds = timeout_seconds
    
    @observe(name="timeout_protected_call")
    async def call_with_timeout(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute function with timeout"""
        
        # Log timeout configuration
        langfuse_context.update_current_observation(
            metadata={"timeout_seconds": self.timeout_seconds}
        )
        
        try:
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.timeout_seconds
            )
            return result
            
        except asyncio.TimeoutError:
            # Log timeout
            langfuse_context.update_current_observation(
                level="ERROR",
                status_message=f"Agent timed out after {self.timeout_seconds}s",
                metadata={"timeout_seconds": self.timeout_seconds}
            )
            
            logger.error(f"Agent timed out after {self.timeout_seconds}s")
            raise AgentTimeoutError(
                f"Agent did not respond within {self.timeout_seconds}s"
            )

class AgentTimeoutError(Exception):
    """Raised when agent times out"""
    pass
```

## Integrated Fault-Tolerant Agent with LangFuse

```python
from langfuse.decorators import observe, langfuse_context

class FaultTolerantAgent:
    """Agent with all fault tolerance patterns + LangFuse"""
    
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
    
    @observe(name="fault_tolerant_agent_invoke")
    async def invoke(self, *args, **kwargs) -> Any:
        """Invoke agent with full fault tolerance + LangFuse tracking"""
        
        # Log agent configuration
        langfuse_context.update_current_observation(
            name=f"agent_{self.agent_id}",
            metadata={
                "agent_id": self.agent_id,
                "has_fallbacks": isinstance(self.agent, FallbackChain),
                "timeout_seconds": self.timeout_protection.timeout_seconds,
                "max_retries": self.retry_policy.max_attempts,
                "circuit_breaker_threshold": self.circuit_breaker.failure_threshold
            }
        )
        
        async def protected_call():
            # Timeout protection
            return await self.timeout_protection.call_with_timeout(
                self.agent.invoke if isinstance(self.agent, FallbackChain) else self.agent,
                *args,
                **kwargs
            )
        
        # Circuit breaker + retry
        try:
            result = await self.circuit_breaker.call(
                lambda: self.retry_policy.execute(protected_call),
            )
            
            # Log success
            langfuse_context.update_current_observation(
                metadata={"success": True}
            )
            
            return result
            
        except Exception as e:
            # Log final failure
            langfuse_context.update_current_observation(
                level="ERROR",
                status_message=f"Agent {self.agent_id} failed after all fault tolerance attempts",
                metadata={
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            raise
```

## Checkpoint Management with LangFuse

```python
from typing import Dict, Any
import json
from datetime import datetime
from langfuse.decorators import observe, langfuse_context

class CheckpointManager:
    """Manages workflow checkpoints with LangFuse tracking"""
    
    def __init__(self, checkpoint_dir: str = "checkpoints/"):
        self.checkpoint_dir = checkpoint_dir
    
    @observe(name="save_checkpoint")
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
        
        # Log checkpoint to LangFuse
        langfuse_context.update_current_observation(
            metadata={
                "workflow_id": workflow_id,
                "phase": phase,
                "checkpoint_file": checkpoint_file
            }
        )
        
        logger.info(f"Checkpoint saved for workflow {workflow_id} at phase {phase}")
    
    @observe(name="load_checkpoint")
    def load_checkpoint(self, workflow_id: str) -> Dict[str, Any]:
        """Load latest checkpoint for workflow"""
        checkpoint_file = f"{self.checkpoint_dir}/{workflow_id}_latest.json"
        
        try:
            with open(checkpoint_file, "r") as f:
                checkpoint = json.load(f)
            
            # Log checkpoint load
            langfuse_context.update_current_observation(
                metadata={
                    "workflow_id": workflow_id,
                    "checkpoint_phase": checkpoint["phase"],
                    "checkpoint_timestamp": checkpoint["timestamp"]
                }
            )
            
            logger.info(f"Checkpoint loaded for workflow {workflow_id}")
            return checkpoint
            
        except FileNotFoundError:
            langfuse_context.update_current_observation(
                level="ERROR",
                status_message=f"No checkpoint found for workflow {workflow_id}"
            )
            raise CheckpointNotFoundError(
                f"No checkpoint found for workflow {workflow_id}"
            )
    
    @observe(name="resume_from_checkpoint")
    def resume_from_checkpoint(self, workflow_id: str):
        """Resume workflow from checkpoint"""
        checkpoint = self.load_checkpoint(workflow_id)
        
        phase = checkpoint["phase"]
        state = checkpoint["state"]
        
        # Log resume
        langfuse_context.update_current_observation(
            metadata={
                "workflow_id": workflow_id,
                "resume_phase": phase,
                "checkpoint_age_seconds": (
                    datetime.now() - datetime.fromisoformat(checkpoint["timestamp"])
                ).total_seconds()
            }
        )
        
        logger.info(f"Resuming workflow {workflow_id} from phase {phase}")
        
        return phase, state

class CheckpointNotFoundError(Exception):
    """Raised when checkpoint is not found"""
    pass
```

## Monitoring Fault Tolerance in LangFuse

### View Retry and Fallback Traces

In LangFuse dashboard, you'll see:

1. **Retry Attempts**: Nested observations showing each retry
2. **Fallback Chains**: Hierarchical view of primary → fallback attempts
3. **Circuit Breaker State**: Metadata showing state transitions
4. **Timeout Events**: Clearly marked timeout errors
5. **Degradation Levels**: Scores showing system degradation

### Query Fault Tolerance Metrics

```python
from langfuse import Langfuse

langfuse = Langfuse()

# Get traces with retries
traces_with_retries = langfuse.get_traces(
    name="retry_execution",
    from_timestamp=datetime.now() - timedelta(days=1)
)

# Analyze retry patterns
retry_stats = {
    "total_retries": 0,
    "successful_retries": 0,
    "failed_retries": 0,
    "avg_attempts": []
}

for trace in traces_with_retries:
    metadata = trace.metadata
    if metadata.get("retry_succeeded"):
        retry_stats["successful_retries"] += 1
        retry_stats["avg_attempts"].append(metadata["attempts_needed"])
    else:
        retry_stats["failed_retries"] += 1
    
    retry_stats["total_retries"] += 1

print(f"Retry Success Rate: {retry_stats['successful_retries'] / retry_stats['total_retries']:.1%}")
print(f"Avg Attempts Needed: {sum(retry_stats['avg_attempts']) / len(retry_stats['avg_attempts']):.1f}")
```

## Integration with LangGraph

```python
from langgraph.graph import StateGraph
from langfuse.langchain import CallbackHandler

def create_fault_tolerant_graph():
    """Create LangGraph with fault tolerance + LangFuse"""
    
    # Initialize LangFuse
    langfuse_handler = CallbackHandler()
    
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
    
    # Add edges
    graph.add_conditional_edges(
        "research_swarm",
        route_with_error_handling,
        {
            "success": "strategy_dev",
            "retry": "research_swarm",
            "fail": END
        }
    )
    
    return graph, langfuse_handler

def wrap_with_fault_tolerance(node_func: Callable) -> Callable:
    """Wrap node function with fault tolerance + LangFuse"""
    
    @observe(name=f"node_{node_func.__name__}")
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
            fault_tolerant_func = FaultTolerantAgent(
                agent_id=node_func.__name__,
                primary_func=node_func,
                timeout_seconds=300,
                max_retries=3
            )
            
            result = await fault_tolerant_func.invoke(state)
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

## Success Metrics

1. **Availability**: > 99.9% uptime despite agent failures
2. **Recovery Time**: < 60 seconds to recover from failures
3. **Data Loss**: 0% data loss during failures
4. **Observability**: 100% of failures traced in LangFuse
5. **Retry Success Rate**: > 80% of retries succeed

---

**Document**: Fault Tolerance & Recovery with LangFuse Integration  
**Created**: 2026-01-18  
**Status**: Design Complete  
**Replaces**: FAULT_TOLERANCE.md (without LangFuse integration)
