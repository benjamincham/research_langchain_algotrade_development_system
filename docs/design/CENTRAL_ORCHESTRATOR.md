# Central Orchestrator - Queue-and-Worker Pattern

## Overview

The Central Orchestrator is the heart of the system, managing the entire workflow from research to strategy validation. It coordinates all phases, manages parallel execution of backtests using a queue-and-worker pattern, tracks experiments, and makes intelligent routing decisions based on feedback loops.

## Design Philosophy

**Keep It Simple**: Use proven patterns (queue-and-worker) instead of over-engineering.

**Key Principles**:
1. **Single Source of Truth**: Orchestrator owns all workflow state
2. **Queue-Based Execution**: Tasks go to queue, workers pick them up when resources available
3. **Automatic Retry**: Failed tasks go back to queue
4. **Resource-Aware**: Only run tasks if resources available
5. **Hierarchical Experiments**: Parent experiments with child variants

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CENTRAL ORCHESTRATOR                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                      STATE MANAGER                                 │  │
│  │  • Current workflow phase                                          │  │
│  │  • Active experiments (parent + child variants)                    │  │
│  │  • Iteration counters (strategy, research, total)                  │  │
│  │  • Quality gate thresholds                                         │  │
│  │  • Research directive                                              │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                              │                                           │
│                              ▼                                           │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                      TASK QUEUE                                    │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐          │  │
│  │  │ Task 1   │  │ Task 2   │  │ Task 3   │  │ Task 4   │  ...     │  │
│  │  │Variant A │  │Variant B │  │Variant C │  │Variant D │          │  │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘          │  │
│  │                                                                     │  │
│  │  Operations: enqueue(), dequeue(), requeue_failed()                │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                              │                                           │
│                              ▼                                           │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                      WORKER POOL                                   │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                         │  │
│  │  │ Worker 1 │  │ Worker 2 │  │ Worker 3 │  ...                    │  │
│  │  │  IDLE    │  │  BUSY    │  │  IDLE    │                         │  │
│  │  └──────────┘  └──────────┘  └──────────┘                         │  │
│  │                                                                     │  │
│  │  Max Workers: Configurable (default: 5)                            │  │
│  │  Resource Check: CPU, Memory available?                            │  │
│  │  Retry Logic: Failed tasks → back to queue                         │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                              │                                           │
│                              ▼                                           │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                      ROUTING ENGINE                                │  │
│  │  Three-Tier Feedback Loop Decision Logic:                          │  │
│  │                                                                     │  │
│  │  Tier 1: Strategy Refinement                                       │  │
│  │    • TUNE_PARAMETERS → Generate variants, enqueue                  │  │
│  │    • FIX_BUG → Fix code, enqueue single test                       │  │
│  │    • REFINE_ALGORITHM → Redesign, enqueue variants                 │  │
│  │                                                                     │  │
│  │  Tier 2: Research Refinement                                       │  │
│  │    • REFINE_RESEARCH → Go back to Research Swarm Phase             │  │
│  │                                                                     │  │
│  │  Tier 3: Abandonment Decision                                      │  │
│  │    • ABANDON → Mark experiment failed, log lessons                 │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                              │                                           │
│                              ▼                                           │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                      EXPERIMENT TRACKER                            │  │
│  │  Hierarchical Structure:                                           │  │
│  │                                                                     │  │
│  │  Experiment_001 (parent)                                           │  │
│  │  ├── Variant_A (child)                                             │  │
│  │  │   ├── Iteration 1                                               │  │
│  │  │   └── Iteration 2                                               │  │
│  │  ├── Variant_B (child)                                             │  │
│  │  │   └── Iteration 1                                               │  │
│  │  └── Variant_C (child)                                             │  │
│  │      └── Iteration 1                                               │  │
│  │                                                                     │  │
│  │  Thread-safe logging to JSONL files                                │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## State Management

### WorkflowState Schema

```python
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Literal
from datetime import datetime

class WorkflowState(BaseModel):
    """
    Complete state of the workflow managed by Central Orchestrator.
    """
    
    # Identification
    workflow_id: str = Field(description="Unique workflow ID")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Current Phase
    current_phase: Literal[
        "INITIALIZATION",
        "TOOL_DEVELOPMENT",
        "RESEARCH_SWARM",
        "STRATEGY_DEVELOPMENT",
        "BACKTESTING",
        "QUALITY_GATE",
        "COMPLETED",
        "FAILED"
    ] = "INITIALIZATION"
    
    # Research Context
    research_directive: str = Field(description="User's research objective")
    ticker: str = Field(description="Target ticker symbol")
    timeframe: str = Field(default="1d", description="Timeframe for analysis")
    
    # Quality Criteria
    quality_criteria: Dict[str, float] = Field(
        description="Quality gate thresholds",
        example={
            "sharpe_ratio": 1.0,
            "max_drawdown": 0.20,
            "win_rate": 0.50
        }
    )
    
    # Iteration Counters
    strategy_iteration: int = Field(default=0, description="Current strategy iteration")
    research_iteration: int = Field(default=0, description="Current research iteration")
    total_iterations: int = Field(default=0, description="Total iterations across all")
    
    max_strategy_iterations: int = Field(default=5)
    max_research_iterations: int = Field(default=3)
    max_total_iterations: int = Field(default=15)
    
    # Experiments
    current_experiment_id: Optional[str] = None
    active_variants: List[str] = Field(default_factory=list)
    
    # Results
    best_strategy: Optional[Dict] = None
    best_metrics: Optional[Dict[str, float]] = None
    
    # Lessons Learned
    lessons_learned: List[str] = Field(default_factory=list)
    
    # Resource Management
    max_parallel_workers: int = Field(default=5)
    current_active_workers: int = Field(default=0)
    
    # Status
    status: Literal["RUNNING", "PAUSED", "COMPLETED", "FAILED"] = "RUNNING"
    failure_reason: Optional[str] = None


class StateManager:
    """
    Manages workflow state with persistence.
    """
    
    def __init__(self, workflow_id: str, storage_dir: str = "workflows"):
        self.workflow_id = workflow_id
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.state_file = self.storage_dir / f"{workflow_id}_state.json"
        self.state: WorkflowState = self._load_or_create_state()
    
    def _load_or_create_state(self) -> WorkflowState:
        """Load state from file or create new."""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                state_dict = json.load(f)
                return WorkflowState(**state_dict)
        else:
            return WorkflowState(workflow_id=self.workflow_id)
    
    def save(self):
        """Save state to file."""
        self.state.updated_at = datetime.utcnow()
        with open(self.state_file, 'w') as f:
            f.write(self.state.model_dump_json(indent=2))
    
    def update_phase(self, phase: str):
        """Update current phase."""
        self.state.current_phase = phase
        self.save()
    
    def increment_strategy_iteration(self):
        """Increment strategy iteration counter."""
        self.state.strategy_iteration += 1
        self.state.total_iterations += 1
        self.save()
    
    def increment_research_iteration(self):
        """Increment research iteration counter."""
        self.state.research_iteration += 1
        self.state.total_iterations += 1
        # Reset strategy iteration when going back to research
        self.state.strategy_iteration = 0
        self.save()
    
    def add_lesson_learned(self, lesson: str):
        """Add a lesson learned."""
        self.state.lessons_learned.append(lesson)
        self.save()
    
    def update_best_strategy(self, strategy: Dict, metrics: Dict[str, float]):
        """Update best strategy found so far."""
        self.state.best_strategy = strategy
        self.state.best_metrics = metrics
        self.save()
    
    def check_iteration_limits(self) -> tuple[bool, str]:
        """
        Check if iteration limits exceeded.
        
        Returns:
            (exceeded, reason)
        """
        if self.state.total_iterations >= self.state.max_total_iterations:
            return True, "Total iteration limit exceeded"
        
        if self.state.strategy_iteration >= self.state.max_strategy_iterations:
            return True, "Strategy iteration limit exceeded"
        
        if self.state.research_iteration >= self.state.max_research_iterations:
            return True, "Research iteration limit exceeded"
        
        return False, ""
```

---

## Task Queue

### Queue-and-Worker Pattern

```python
from queue import Queue, Empty
from typing import Dict, Any, Optional
import asyncio

class BacktestTask(BaseModel):
    """A single backtest task."""
    
    task_id: str
    experiment_id: str
    variant_id: str
    strategy_code: str
    strategy_parameters: Dict[str, Any]
    ticker: str
    timeframe: str
    priority: int = Field(default=0, description="Higher = more important")
    retry_count: int = Field(default=0)
    max_retries: int = Field(default=3)


class TaskQueue:
    """
    Thread-safe task queue for backtest jobs.
    
    Simple queue-and-worker pattern:
    - Tasks are added to queue
    - Workers pick up tasks when available
    - Failed tasks go back to queue (with retry limit)
    """
    
    def __init__(self, max_retries: int = 3):
        self.queue = Queue()
        self.max_retries = max_retries
        self.in_progress: Dict[str, BacktestTask] = {}
        self.completed: Dict[str, Any] = {}
        self.failed: Dict[str, tuple[BacktestTask, str]] = {}
        self.logger = get_logger(__name__)
    
    def enqueue(self, task: BacktestTask):
        """Add task to queue."""
        self.queue.put(task)
        self.logger.info(f"Enqueued task {task.task_id} (variant: {task.variant_id})")
    
    def enqueue_batch(self, tasks: List[BacktestTask]):
        """Add multiple tasks to queue."""
        for task in tasks:
            self.enqueue(task)
    
    def dequeue(self, timeout: float = 1.0) -> Optional[BacktestTask]:
        """
        Get next task from queue.
        
        Args:
            timeout: How long to wait for a task (seconds)
        
        Returns:
            BacktestTask if available, None if queue empty
        """
        try:
            task = self.queue.get(timeout=timeout)
            self.in_progress[task.task_id] = task
            return task
        except Empty:
            return None
    
    def mark_completed(self, task_id: str, result: Any):
        """Mark task as completed successfully."""
        if task_id in self.in_progress:
            task = self.in_progress.pop(task_id)
            self.completed[task_id] = result
            self.logger.info(f"Task {task_id} completed successfully")
    
    def mark_failed(self, task_id: str, error: str):
        """
        Mark task as failed and requeue if retries available.
        
        Args:
            task_id: Task ID
            error: Error message
        """
        if task_id not in self.in_progress:
            self.logger.warning(f"Task {task_id} not in progress, cannot mark failed")
            return
        
        task = self.in_progress.pop(task_id)
        task.retry_count += 1
        
        if task.retry_count < task.max_retries:
            # Requeue for retry
            self.logger.warning(
                f"Task {task_id} failed (attempt {task.retry_count}/{task.max_retries}). "
                f"Requeuing. Error: {error}"
            )
            self.enqueue(task)
        else:
            # Max retries exceeded
            self.logger.error(
                f"Task {task_id} failed after {task.retry_count} attempts. "
                f"Giving up. Error: {error}"
            )
            self.failed[task_id] = (task, error)
    
    def get_stats(self) -> Dict[str, int]:
        """Get queue statistics."""
        return {
            "queued": self.queue.qsize(),
            "in_progress": len(self.in_progress),
            "completed": len(self.completed),
            "failed": len(self.failed)
        }
    
    def is_empty(self) -> bool:
        """Check if queue is empty and no tasks in progress."""
        return self.queue.empty() and len(self.in_progress) == 0
```

---

## Worker Pool

### Resource-Aware Workers

```python
import psutil
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Any

class WorkerPool:
    """
    Pool of workers that execute backtest tasks.
    
    Features:
    - Configurable number of workers
    - Resource checking (CPU, memory)
    - Automatic task pickup from queue
    - Error handling and retry
    """
    
    def __init__(
        self,
        max_workers: int = 5,
        task_queue: TaskQueue = None,
        cpu_threshold: float = 0.8,  # Don't start new tasks if CPU > 80%
        memory_threshold: float = 0.8  # Don't start new tasks if memory > 80%
    ):
        self.max_workers = max_workers
        self.task_queue = task_queue
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.active_workers = 0
        self.logger = get_logger(__name__)
    
    def check_resources(self) -> tuple[bool, str]:
        """
        Check if resources are available to start new task.
        
        Returns:
            (available, reason)
        """
        # Check CPU usage
        cpu_percent = psutil.cpu_percent(interval=1) / 100.0
        if cpu_percent > self.cpu_threshold:
            return False, f"CPU usage too high: {cpu_percent:.1%}"
        
        # Check memory usage
        memory_percent = psutil.virtual_memory().percent / 100.0
        if memory_percent > self.memory_threshold:
            return False, f"Memory usage too high: {memory_percent:.1%}"
        
        # Check worker availability
        if self.active_workers >= self.max_workers:
            return False, f"All {self.max_workers} workers busy"
        
        return True, "Resources available"
    
    async def run_task(
        self,
        task: BacktestTask,
        executor_func: Callable[[BacktestTask], Any]
    ):
        """
        Execute a single backtest task.
        
        Args:
            task: BacktestTask to execute
            executor_func: Function that executes the backtest
        """
        self.active_workers += 1
        self.logger.info(
            f"Worker starting task {task.task_id} "
            f"({self.active_workers}/{self.max_workers} active)"
        )
        
        try:
            # Run backtest in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                executor_func,
                task
            )
            
            # Mark completed
            self.task_queue.mark_completed(task.task_id, result)
            
        except Exception as e:
            # Mark failed (will requeue if retries available)
            error_msg = f"{type(e).__name__}: {str(e)}"
            self.logger.error(f"Task {task.task_id} failed: {error_msg}")
            self.task_queue.mark_failed(task.task_id, error_msg)
        
        finally:
            self.active_workers -= 1
            self.logger.info(
                f"Worker finished task {task.task_id} "
                f"({self.active_workers}/{self.max_workers} active)"
            )
    
    async def process_queue(
        self,
        executor_func: Callable[[BacktestTask], Any],
        check_interval: float = 1.0
    ):
        """
        Continuously process tasks from queue.
        
        Args:
            executor_func: Function that executes backtests
            check_interval: How often to check queue (seconds)
        """
        self.logger.info(f"Worker pool started with {self.max_workers} workers")
        
        while True:
            # Check if resources available
            resources_available, reason = self.check_resources()
            
            if not resources_available:
                self.logger.debug(f"Waiting for resources: {reason}")
                await asyncio.sleep(check_interval)
                continue
            
            # Try to get task from queue
            task = self.task_queue.dequeue(timeout=check_interval)
            
            if task is None:
                # Queue empty, check if we're done
                if self.task_queue.is_empty():
                    self.logger.info("Queue empty and no tasks in progress")
                    break
                
                # Tasks still in progress, wait
                await asyncio.sleep(check_interval)
                continue
            
            # Start task in background
            asyncio.create_task(self.run_task(task, executor_func))
        
        self.logger.info("Worker pool finished processing all tasks")
    
    def shutdown(self):
        """Shutdown worker pool."""
        self.logger.info("Shutting down worker pool")
        self.executor.shutdown(wait=True)
```

---

## Routing Engine

### Three-Tier Feedback Loop Implementation

```python
from src.agents.failure_analysis import FailureAnalysisAgent
from src.agents.trajectory_analysis import LLMTrajectoryAnalyzer

class RoutingEngine:
    """
    Implements three-tier feedback loop routing logic.
    
    Tier 1: Strategy Refinement (tune, fix, refine)
    Tier 2: Research Refinement (go back to research)
    Tier 3: Abandonment Decision (give up)
    """
    
    def __init__(self, state_manager: StateManager):
        self.state_manager = state_manager
        self.failure_agent = FailureAnalysisAgent()
        self.trajectory_analyzer = LLMTrajectoryAnalyzer()
        self.logger = get_logger(__name__)
    
    async def determine_next_action(
        self,
        experiment_records: List[ExperimentRecord],
        quality_gate_results: Dict[str, Any]
    ) -> tuple[str, Dict[str, Any]]:
        """
        Determine next action based on quality gate results and experiment history.
        
        Returns:
            (action, action_details)
            
        Actions:
            - "TUNE_PARAMETERS": Generate parameter variants
            - "FIX_BUG": Fix implementation bug
            - "REFINE_ALGORITHM": Redesign strategy logic
            - "REFINE_RESEARCH": Go back to research swarm
            - "ABANDON": Give up on this direction
            - "SUCCESS": Strategy passed quality gates
        """
        # Check if any variant passed
        if any(r.gate_passed for r in experiment_records):
            best_record = max(
                (r for r in experiment_records if r.gate_passed),
                key=lambda r: r.gate_score
            )
            return "SUCCESS", {"best_record": best_record}
        
        # All variants failed - perform failure analysis
        latest_record = experiment_records[-1]
        
        failure_analysis = await self.failure_agent.analyze_failure(
            strategy_code=latest_record.strategy_code,
            research_findings=[],  # TODO: Get from memory
            backtest_metrics=latest_record.metrics,
            quality_gate_results=quality_gate_results,
            iteration_history=[r.dict() for r in experiment_records],
            current_iteration=self.state_manager.state.strategy_iteration,
            max_iterations=self.state_manager.state.max_strategy_iterations
        )
        
        # Perform trajectory analysis if we have enough data
        if len(experiment_records) >= 2:
            trajectory_analysis = await self.trajectory_analyzer.analyze_trajectory(
                experiment_records,
                TrajectoryAnalyzer(experiment_records).analyze_trajectory()
            )
            
            # Use trajectory analysis to override failure analysis if needed
            if trajectory_analysis['next_action'] == 'ABANDON':
                return "ABANDON", {
                    "reason": "Trajectory analysis indicates no convergence",
                    "trajectory_analysis": trajectory_analysis
                }
            
            elif trajectory_analysis['next_action'] == 'PIVOT':
                return "REFINE_RESEARCH", {
                    "reason": "Trajectory analysis recommends pivoting to research",
                    "trajectory_analysis": trajectory_analysis
                }
        
        # Check iteration limits
        exceeded, reason = self.state_manager.check_iteration_limits()
        if exceeded:
            return "ABANDON", {"reason": reason}
        
        # Route based on failure classification
        if failure_analysis.recommendation == "TUNE_PARAMETERS":
            return "TUNE_PARAMETERS", {
                "failure_analysis": failure_analysis,
                "specific_actions": failure_analysis.specific_actions
            }
        
        elif failure_analysis.recommendation == "FIX_BUG":
            return "FIX_BUG", {
                "failure_analysis": failure_analysis,
                "bug_locations": failure_analysis.bug_detection['bug_locations']
            }
        
        elif failure_analysis.recommendation == "REFINE_ALGORITHM":
            # Check if we should go to Tier 2 instead
            if self.state_manager.state.strategy_iteration >= 3:
                # After 3 strategy iterations, consider going to research
                return "REFINE_RESEARCH", {
                    "reason": "Multiple strategy refinements failed, need better research"
                }
            else:
                return "REFINE_ALGORITHM", {
                    "failure_analysis": failure_analysis,
                    "design_suggestions": failure_analysis.specific_actions
                }
        
        elif failure_analysis.recommendation == "REFINE_RESEARCH":
            return "REFINE_RESEARCH", {
                "failure_analysis": failure_analysis,
                "new_research_directive": failure_analysis.specific_actions
            }
        
        elif failure_analysis.recommendation == "ABANDON":
            return "ABANDON", {
                "failure_analysis": failure_analysis,
                "reason": failure_analysis.reasoning
            }
        
        else:
            # Fallback
            return "ABANDON", {"reason": "Unknown failure classification"}
```

---

## Central Orchestrator Implementation

### Main Orchestrator Class

```python
class CentralOrchestrator:
    """
    Central Orchestrator manages the entire workflow.
    
    Responsibilities:
    - Coordinate all workflow phases
    - Manage parallel execution via queue-and-worker pattern
    - Track experiments hierarchically
    - Make routing decisions via three-tier feedback loops
    - Enforce resource limits and iteration bounds
    """
    
    def __init__(
        self,
        max_parallel_workers: int = 5,
        max_strategy_iterations: int = 5,
        max_research_iterations: int = 3,
        max_total_iterations: int = 15,
        cpu_threshold: float = 0.8,
        memory_threshold: float = 0.8
    ):
        self.workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize components
        self.state_manager = StateManager(self.workflow_id)
        self.state_manager.state.max_parallel_workers = max_parallel_workers
        self.state_manager.state.max_strategy_iterations = max_strategy_iterations
        self.state_manager.state.max_research_iterations = max_research_iterations
        self.state_manager.state.max_total_iterations = max_total_iterations
        
        self.task_queue = TaskQueue()
        self.worker_pool = WorkerPool(
            max_workers=max_parallel_workers,
            task_queue=self.task_queue,
            cpu_threshold=cpu_threshold,
            memory_threshold=memory_threshold
        )
        self.routing_engine = RoutingEngine(self.state_manager)
        
        self.experiment_logger = None  # Will be created per experiment
        self.logger = get_logger(__name__)
    
    async def run_workflow(
        self,
        ticker: str,
        research_directive: str,
        quality_criteria: Dict[str, float],
        timeframe: str = "1d"
    ) -> Dict[str, Any]:
        """
        Run the complete workflow from research to validated strategy.
        
        Args:
            ticker: Target ticker symbol
            research_directive: User's research objective
            quality_criteria: Quality gate thresholds
            timeframe: Timeframe for analysis
        
        Returns:
            Workflow result with best strategy (if found) and all experiment data
        """
        self.logger.info(f"Starting workflow {self.workflow_id}")
        self.logger.info(f"Ticker: {ticker}, Directive: {research_directive}")
        
        # Initialize state
        self.state_manager.state.ticker = ticker
        self.state_manager.state.research_directive = research_directive
        self.state_manager.state.quality_criteria = quality_criteria
        self.state_manager.state.timeframe = timeframe
        self.state_manager.save()
        
        try:
            # Phase 1: Tool Development (if needed)
            await self._run_tool_development_phase()
            
            # Main loop: Research → Strategy → Backtest → Quality Gate
            while True:
                # Phase 2: Research Swarm
                research_findings = await self._run_research_swarm_phase()
                
                # Phase 3: Strategy Development (generates N variants)
                strategy_variants = await self._run_strategy_development_phase(
                    research_findings
                )
                
                # Phase 4: Parallel Backtesting
                backtest_results = await self._run_parallel_backtesting_phase(
                    strategy_variants
                )
                
                # Phase 5: Quality Gate Validation
                action, action_details = await self._run_quality_gate_phase(
                    backtest_results
                )
                
                # Route based on action
                if action == "SUCCESS":
                    self.logger.info("✅ Strategy passed quality gates!")
                    self.state_manager.state.status = "COMPLETED"
                    self.state_manager.save()
                    return self._build_success_result(action_details)
                
                elif action == "ABANDON":
                    self.logger.info("❌ Abandoning workflow")
                    self.state_manager.state.status = "FAILED"
                    self.state_manager.state.failure_reason = action_details['reason']
                    self.state_manager.save()
                    return self._build_failure_result(action_details)
                
                elif action == "REFINE_RESEARCH":
                    self.logger.info("🔄 Going back to research (Tier 2)")
                    self.state_manager.increment_research_iteration()
                    # Continue loop (will run research again)
                
                elif action in ["TUNE_PARAMETERS", "FIX_BUG", "REFINE_ALGORITHM"]:
                    self.logger.info(f"🔄 Strategy refinement: {action} (Tier 1)")
                    self.state_manager.increment_strategy_iteration()
                    # Continue loop (will generate new variants)
                
                else:
                    raise ValueError(f"Unknown action: {action}")
        
        except Exception as e:
            self.logger.error(f"Workflow failed with error: {e}")
            self.state_manager.state.status = "FAILED"
            self.state_manager.state.failure_reason = str(e)
            self.state_manager.save()
            raise
    
    async def _run_parallel_backtesting_phase(
        self,
        strategy_variants: List[Dict]
    ) -> List[ExperimentRecord]:
        """
        Run backtests for all strategy variants in parallel using queue-and-worker pattern.
        """
        self.logger.info(f"Starting parallel backtesting of {len(strategy_variants)} variants")
        self.state_manager.update_phase("BACKTESTING")
        
        # Create experiment logger
        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_logger = ExperimentLogger(experiment_id)
        self.state_manager.state.current_experiment_id = experiment_id
        
        # Create backtest tasks and enqueue
        tasks = []
        for i, variant in enumerate(strategy_variants):
            task = BacktestTask(
                task_id=f"{experiment_id}_variant_{i}",
                experiment_id=experiment_id,
                variant_id=f"variant_{i}",
                strategy_code=variant['code'],
                strategy_parameters=variant['parameters'],
                ticker=self.state_manager.state.ticker,
                timeframe=self.state_manager.state.timeframe
            )
            tasks.append(task)
        
        self.task_queue.enqueue_batch(tasks)
        
        # Process queue with worker pool
        await self.worker_pool.process_queue(
            executor_func=self._execute_backtest
        )
        
        # Collect results
        experiment_records = []
        for task_id, result in self.task_queue.completed.items():
            experiment_records.append(result)
        
        self.logger.info(
            f"Parallel backtesting complete: "
            f"{len(experiment_records)} succeeded, "
            f"{len(self.task_queue.failed)} failed"
        )
        
        return experiment_records
    
    def _execute_backtest(self, task: BacktestTask) -> ExperimentRecord:
        """
        Execute a single backtest task.
        
        This is the function that workers call to run backtests.
        """
        self.logger.info(f"Executing backtest for {task.variant_id}")
        
        # TODO: Implement actual backtesting logic
        # For now, placeholder
        
        # Run backtest
        metrics = {
            "sharpe_ratio": 0.85,  # Placeholder
            "max_drawdown": 0.20,
            "win_rate": 0.50
        }
        
        # Evaluate quality gate
        gate_passed = self._evaluate_quality_gate(metrics)
        gate_score = self._compute_gate_score(metrics)
        
        # Log to experiment tracker
        record = self.experiment_logger.log_iteration(
            strategy_name=f"Strategy_{task.variant_id}",
            strategy_code=task.strategy_code,
            strategy_description="",
            parameters=task.strategy_parameters,
            metrics=metrics,
            gate_passed=gate_passed,
            gate_score=gate_score,
            failed_criteria=[],
            action_taken="",
            parameter_changes={}
        )
        
        return record
    
    # ... (other phase implementations)
    
    def _build_success_result(self, action_details: Dict) -> Dict[str, Any]:
        """Build success result."""
        return {
            "status": "SUCCESS",
            "workflow_id": self.workflow_id,
            "best_strategy": action_details['best_record'].strategy_code,
            "best_metrics": action_details['best_record'].metrics,
            "total_iterations": self.state_manager.state.total_iterations,
            "lessons_learned": self.state_manager.state.lessons_learned
        }
    
    def _build_failure_result(self, action_details: Dict) -> Dict[str, Any]:
        """Build failure result."""
        return {
            "status": "FAILED",
            "workflow_id": self.workflow_id,
            "reason": action_details['reason'],
            "total_iterations": self.state_manager.state.total_iterations,
            "lessons_learned": self.state_manager.state.lessons_learned
        }
```

---

## Summary

### Key Design Decisions

1. **Queue-and-Worker Pattern**: Simple, proven pattern for parallel execution
2. **Resource-Aware Execution**: Check CPU/memory before starting new tasks
3. **Automatic Retry**: Failed tasks go back to queue (up to max retries)
4. **Hierarchical Experiments**: Parent experiments with child variants
5. **Three-Tier Routing**: Strategy refinement → Research refinement → Abandonment

### Components

- **StateManager**: Manages workflow state with persistence
- **TaskQueue**: Thread-safe queue for backtest tasks
- **WorkerPool**: Resource-aware workers that execute tasks
- **RoutingEngine**: Three-tier feedback loop decision logic
- **CentralOrchestrator**: Main coordinator that ties everything together

### Benefits

✅ **Simple**: Queue-and-worker is a well-understood pattern  
✅ **Robust**: Automatic retry for failed tasks  
✅ **Resource-Aware**: Only runs tasks when resources available  
✅ **Scalable**: Easy to adjust number of workers  
✅ **Transparent**: Complete state tracking and logging

---

**End of Central Orchestrator Design**
