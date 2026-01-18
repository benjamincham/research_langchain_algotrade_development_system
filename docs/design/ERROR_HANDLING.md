# Error Handling and Recovery Design

## Overview

This document describes the comprehensive error handling and recovery mechanisms for the trading research system. The design ensures resilience, graceful degradation, and the ability to resume from failures.

## Error Categories

| Category | Examples | Severity | Recovery Strategy |
|----------|----------|----------|-------------------|
| **API Errors** | Rate limits, timeouts, auth failures | High | Exponential backoff, fallback |
| **Agent Errors** | Subagent failures, context overflow | Medium | Partial results, retry |
| **Tool Errors** | Tool execution failures, invalid inputs | Medium | Skip and log, alternative tool |
| **Data Errors** | Missing data, invalid format | Low | Default values, skip symbol |
| **Code Errors** | Generated code syntax/runtime errors | Medium | Regenerate with feedback |
| **System Errors** | Memory issues, disk full | Critical | Checkpoint and alert |

## Error Handling Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ERROR HANDLING SYSTEM                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    ERROR DETECTOR                                │   │
│  │  • Exception catching at all levels                              │   │
│  │  • Timeout monitoring                                            │   │
│  │  • Health checks                                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    ERROR CLASSIFIER                              │   │
│  │  • Categorize error type                                         │   │
│  │  • Determine severity                                            │   │
│  │  • Select recovery strategy                                      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    RECOVERY EXECUTOR                             │   │
│  │  • Execute recovery strategy                                     │   │
│  │  • Track retry attempts                                          │   │
│  │  • Escalate if needed                                            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    CHECKPOINT MANAGER                            │   │
│  │  • Save state before risky operations                            │   │
│  │  • Restore from last good state                                  │   │
│  │  • Manage checkpoint lifecycle                                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Recovery Strategies

### 1. Exponential Backoff (API Errors)

```python
import asyncio
from typing import TypeVar, Callable, Any
import random

T = TypeVar('T')

class ExponentialBackoff:
    """Exponential backoff with jitter for API calls."""
    
    def __init__(
        self,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    async def execute(
        self,
        func: Callable[..., T],
        *args,
        **kwargs
    ) -> T:
        """Execute function with exponential backoff."""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
            except RateLimitError as e:
                last_exception = e
                delay = self._calculate_delay(attempt)
                await asyncio.sleep(delay)
            except TimeoutError as e:
                last_exception = e
                delay = self._calculate_delay(attempt)
                await asyncio.sleep(delay)
            except AuthenticationError as e:
                # Don't retry auth errors
                raise e
        
        raise MaxRetriesExceeded(
            f"Failed after {self.max_retries} attempts",
            last_exception=last_exception
        )
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with optional jitter."""
        delay = min(
            self.base_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        
        if self.jitter:
            delay = delay * (0.5 + random.random())
        
        return delay
```

### 2. Partial Results (Swarm Failures)

```python
class PartialResultHandler:
    """Handle partial results from swarm execution."""
    
    def __init__(self, min_success_rate: float = 0.5):
        self.min_success_rate = min_success_rate
    
    async def handle_swarm_results(
        self,
        results: list[tuple[SubagentSpec, Any]],
        retry_func: Callable
    ) -> SwarmResult:
        """
        Handle mixed success/failure results from swarm.
        
        Args:
            results: List of (spec, result) tuples where result may be Exception
            retry_func: Function to retry failed subagents
            
        Returns:
            SwarmResult with successful results and failure info
        """
        successful = []
        failed = []
        
        for spec, result in results:
            if isinstance(result, Exception):
                failed.append((spec, result))
            else:
                successful.append((spec, result))
        
        success_rate = len(successful) / len(results)
        
        if success_rate >= self.min_success_rate:
            # Proceed with partial results
            return SwarmResult(
                results=[r for _, r in successful],
                partial=True,
                failed_agents=[s.name for s, _ in failed],
                success_rate=success_rate
            )
        else:
            # Retry failed agents
            retry_results = await self._retry_failed(failed, retry_func)
            
            # Combine results
            all_successful = successful + [
                (s, r) for s, r in retry_results 
                if not isinstance(r, Exception)
            ]
            
            final_success_rate = len(all_successful) / len(results)
            
            if final_success_rate >= self.min_success_rate:
                return SwarmResult(
                    results=[r for _, r in all_successful],
                    partial=True,
                    success_rate=final_success_rate
                )
            else:
                raise InsufficientResultsError(
                    f"Only {final_success_rate:.0%} of subagents succeeded"
                )
    
    async def _retry_failed(
        self,
        failed: list[tuple[SubagentSpec, Exception]],
        retry_func: Callable
    ) -> list[tuple[SubagentSpec, Any]]:
        """Retry failed subagents."""
        retry_results = []
        
        for spec, original_error in failed:
            try:
                result = await retry_func(spec)
                retry_results.append((spec, result))
            except Exception as e:
                retry_results.append((spec, e))
        
        return retry_results
```

### 3. Code Regeneration (Code Errors)

```python
class CodeErrorHandler:
    """Handle errors in generated code."""
    
    def __init__(self, llm, max_regenerations: int = 3):
        self.llm = llm
        self.max_regenerations = max_regenerations
    
    async def handle_code_error(
        self,
        original_code: str,
        error: Exception,
        context: dict
    ) -> str:
        """
        Regenerate code based on error feedback.
        
        Args:
            original_code: The code that failed
            error: The exception that occurred
            context: Additional context (strategy spec, etc.)
            
        Returns:
            Fixed code
        """
        error_info = self._extract_error_info(error)
        
        for attempt in range(self.max_regenerations):
            prompt = self._build_fix_prompt(
                original_code=original_code,
                error_info=error_info,
                context=context,
                attempt=attempt
            )
            
            fixed_code = await self.llm.generate(prompt)
            
            # Validate fixed code
            validation_result = await self._validate_code(fixed_code)
            
            if validation_result.passed:
                return fixed_code
            else:
                error_info = validation_result.error_info
        
        raise CodeGenerationFailed(
            f"Failed to fix code after {self.max_regenerations} attempts"
        )
    
    def _extract_error_info(self, error: Exception) -> dict:
        """Extract useful information from error."""
        return {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "line_number": self._get_line_number(error)
        }
    
    def _build_fix_prompt(
        self,
        original_code: str,
        error_info: dict,
        context: dict,
        attempt: int
    ) -> str:
        """Build prompt for code fixing."""
        return f"""
The following code has an error. Please fix it.

## Original Code
```python
{original_code}
```

## Error Information
- Type: {error_info['type']}
- Message: {error_info['message']}
- Line: {error_info.get('line_number', 'unknown')}

## Context
{context}

## Instructions
1. Identify the root cause of the error
2. Fix the code while maintaining the original intent
3. Ensure the fix doesn't introduce new errors
4. Return only the fixed code, no explanations

Attempt {attempt + 1} of {self.max_regenerations}
"""
```

### 4. Circuit Breaker (System Protection)

```python
from enum import Enum
from datetime import datetime, timedelta

class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


class CircuitBreaker:
    """Circuit breaker pattern for system protection."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        half_open_max_calls: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.half_open_calls = 0
    
    async def execute(
        self,
        func: Callable[..., T],
        *args,
        **kwargs
    ) -> T:
        """Execute function with circuit breaker protection."""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
            else:
                raise CircuitOpenError("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        
        elapsed = datetime.now() - self.last_failure_time
        return elapsed.total_seconds() >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful execution."""
        if self.state == CircuitState.HALF_OPEN:
            self.half_open_calls += 1
            if self.half_open_calls >= self.half_open_max_calls:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
        else:
            self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
        elif self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
```

## Checkpoint System

```python
from dataclasses import dataclass
from typing import Optional
import json
import gzip

@dataclass
class Checkpoint:
    """A checkpoint of pipeline state."""
    id: str
    phase: str
    state: dict
    created_at: datetime
    metadata: dict


class CheckpointManager:
    """Manages checkpoints for pipeline recovery."""
    
    def __init__(self, storage_path: str = "./checkpoints"):
        self.storage_path = storage_path
        self.checkpoints: dict[str, Checkpoint] = {}
    
    async def create_checkpoint(
        self,
        phase: str,
        state: dict,
        metadata: Optional[dict] = None
    ) -> str:
        """Create a checkpoint before risky operation."""
        checkpoint_id = f"{phase}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        checkpoint = Checkpoint(
            id=checkpoint_id,
            phase=phase,
            state=state,
            created_at=datetime.now(),
            metadata=metadata or {}
        )
        
        # Save to disk
        await self._save_checkpoint(checkpoint)
        
        # Keep in memory
        self.checkpoints[checkpoint_id] = checkpoint
        
        return checkpoint_id
    
    async def restore_checkpoint(
        self,
        checkpoint_id: str
    ) -> dict:
        """Restore state from checkpoint."""
        if checkpoint_id in self.checkpoints:
            return self.checkpoints[checkpoint_id].state
        
        # Load from disk
        checkpoint = await self._load_checkpoint(checkpoint_id)
        return checkpoint.state
    
    async def get_latest_checkpoint(
        self,
        phase: Optional[str] = None
    ) -> Optional[Checkpoint]:
        """Get the most recent checkpoint, optionally for a specific phase."""
        checkpoints = list(self.checkpoints.values())
        
        if phase:
            checkpoints = [c for c in checkpoints if c.phase == phase]
        
        if not checkpoints:
            return None
        
        return max(checkpoints, key=lambda c: c.created_at)
    
    async def _save_checkpoint(self, checkpoint: Checkpoint):
        """Save checkpoint to disk."""
        filepath = f"{self.storage_path}/{checkpoint.id}.json.gz"
        
        data = {
            "id": checkpoint.id,
            "phase": checkpoint.phase,
            "state": checkpoint.state,
            "created_at": checkpoint.created_at.isoformat(),
            "metadata": checkpoint.metadata
        }
        
        with gzip.open(filepath, 'wt') as f:
            json.dump(data, f)
    
    async def _load_checkpoint(self, checkpoint_id: str) -> Checkpoint:
        """Load checkpoint from disk."""
        filepath = f"{self.storage_path}/{checkpoint_id}.json.gz"
        
        with gzip.open(filepath, 'rt') as f:
            data = json.load(f)
        
        return Checkpoint(
            id=data["id"],
            phase=data["phase"],
            state=data["state"],
            created_at=datetime.fromisoformat(data["created_at"]),
            metadata=data["metadata"]
        )
```

## Error Handler Integration

```python
class PipelineErrorHandler:
    """Central error handler for the pipeline."""
    
    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        memory: MemoryManager
    ):
        self.checkpoint_manager = checkpoint_manager
        self.memory = memory
        self.backoff = ExponentialBackoff()
        self.partial_handler = PartialResultHandler()
        self.code_handler = CodeErrorHandler(llm=get_llm())
        self.circuit_breakers: dict[str, CircuitBreaker] = {}
    
    def get_circuit_breaker(self, name: str) -> CircuitBreaker:
        """Get or create a circuit breaker for a service."""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker()
        return self.circuit_breakers[name]
    
    async def handle_error(
        self,
        error: Exception,
        phase: str,
        state: dict,
        context: dict
    ) -> RecoveryResult:
        """
        Handle an error and attempt recovery.
        
        Args:
            error: The exception that occurred
            phase: Current pipeline phase
            state: Current pipeline state
            context: Additional context
            
        Returns:
            RecoveryResult with action to take
        """
        error_type = self._classify_error(error)
        
        # Log the error
        await self._log_error(error, phase, context)
        
        # Select recovery strategy
        if error_type == ErrorType.API_RATE_LIMIT:
            return await self._handle_rate_limit(error, context)
        
        elif error_type == ErrorType.API_TIMEOUT:
            return await self._handle_timeout(error, context)
        
        elif error_type == ErrorType.SUBAGENT_FAILURE:
            return await self._handle_subagent_failure(error, state, context)
        
        elif error_type == ErrorType.CODE_ERROR:
            return await self._handle_code_error(error, state, context)
        
        elif error_type == ErrorType.CONTEXT_OVERFLOW:
            return await self._handle_context_overflow(error, state, phase)
        
        elif error_type == ErrorType.CRITICAL:
            return await self._handle_critical_error(error, state, phase)
        
        else:
            # Unknown error - checkpoint and notify
            await self.checkpoint_manager.create_checkpoint(phase, state)
            return RecoveryResult(
                action=RecoveryAction.NOTIFY_USER,
                message=f"Unknown error: {error}"
            )
    
    def _classify_error(self, error: Exception) -> ErrorType:
        """Classify error into a category."""
        if isinstance(error, RateLimitError):
            return ErrorType.API_RATE_LIMIT
        elif isinstance(error, TimeoutError):
            return ErrorType.API_TIMEOUT
        elif isinstance(error, SubagentError):
            return ErrorType.SUBAGENT_FAILURE
        elif isinstance(error, (SyntaxError, RuntimeError)):
            return ErrorType.CODE_ERROR
        elif isinstance(error, ContextOverflowError):
            return ErrorType.CONTEXT_OVERFLOW
        elif isinstance(error, (MemoryError, OSError)):
            return ErrorType.CRITICAL
        else:
            return ErrorType.UNKNOWN
    
    async def _handle_rate_limit(
        self,
        error: Exception,
        context: dict
    ) -> RecoveryResult:
        """Handle API rate limit errors."""
        # Wait and retry
        return RecoveryResult(
            action=RecoveryAction.RETRY_WITH_BACKOFF,
            delay=self.backoff._calculate_delay(context.get("attempt", 0))
        )
    
    async def _handle_context_overflow(
        self,
        error: Exception,
        state: dict,
        phase: str
    ) -> RecoveryResult:
        """Handle context window overflow."""
        # Create checkpoint
        checkpoint_id = await self.checkpoint_manager.create_checkpoint(
            phase=phase,
            state=state,
            metadata={"reason": "context_overflow"}
        )
        
        # Summarize and compress context
        return RecoveryResult(
            action=RecoveryAction.COMPRESS_CONTEXT,
            checkpoint_id=checkpoint_id
        )
    
    async def _log_error(
        self,
        error: Exception,
        phase: str,
        context: dict
    ):
        """Log error for debugging and learning."""
        await self.memory.store_lesson(
            lesson=f"Error in {phase}: {type(error).__name__}: {str(error)}",
            strategy_id=context.get("strategy_id", "unknown"),
            iteration=context.get("iteration", 0),
            failure_type="error",
            feedback={"traceback": traceback.format_exc()},
            severity="major"
        )
```

## Integration with Pipeline

```python
# In main workflow
async def run_phase_with_error_handling(
    phase_func: Callable,
    phase_name: str,
    state: PipelineState,
    error_handler: PipelineErrorHandler
) -> PipelineState:
    """Run a pipeline phase with error handling."""
    
    # Create checkpoint before phase
    checkpoint_id = await error_handler.checkpoint_manager.create_checkpoint(
        phase=phase_name,
        state=state
    )
    
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            # Execute phase
            new_state = await phase_func(state)
            return new_state
            
        except Exception as e:
            # Handle error
            recovery = await error_handler.handle_error(
                error=e,
                phase=phase_name,
                state=state,
                context={"attempt": attempt, "checkpoint_id": checkpoint_id}
            )
            
            if recovery.action == RecoveryAction.RETRY_WITH_BACKOFF:
                await asyncio.sleep(recovery.delay)
                continue
            
            elif recovery.action == RecoveryAction.SKIP_AND_CONTINUE:
                return {**state, "errors": state.get("errors", []) + [str(e)]}
            
            elif recovery.action == RecoveryAction.RESTORE_CHECKPOINT:
                state = await error_handler.checkpoint_manager.restore_checkpoint(
                    recovery.checkpoint_id
                )
                continue
            
            elif recovery.action == RecoveryAction.NOTIFY_USER:
                raise UserNotificationRequired(recovery.message)
            
            else:
                raise e
    
    raise MaxRetriesExceeded(f"Phase {phase_name} failed after {max_attempts} attempts")
```
