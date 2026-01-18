# Memory Architecture Design

## Overview

The memory system uses ChromaDB as the vector store for semantic search and retrieval, combined with structured storage for state management. The architecture supports both short-term conversational memory and long-term knowledge persistence with lineage tracking.

## Memory Types

| Type | Purpose | Storage | Retention |
|------|---------|---------|-----------|
| **Short-term** | Current session state | LangGraph Checkpointer | Session duration |
| **Semantic** | Research findings, strategies | ChromaDB | Permanent with archiving |
| **Episodic** | Lessons learned, failures | ChromaDB | 90 days active, then archived |
| **Procedural** | Tool definitions, workflows | ChromaDB + JSON | Permanent |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         MEMORY SYSTEM                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    SHORT-TERM MEMORY                             │   │
│  │  ┌─────────────────┐  ┌─────────────────┐                       │   │
│  │  │ LangGraph       │  │ Conversation    │                       │   │
│  │  │ Checkpointer    │  │ Buffer          │                       │   │
│  │  └─────────────────┘  └─────────────────┘                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    CHROMADB VECTOR STORE                         │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────┐  │   │
│  │  │ research_   │ │ strategy_   │ │ lessons_    │ │ market_   │  │   │
│  │  │ findings    │ │ library     │ │ learned     │ │ regimes   │  │   │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └───────────┘  │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐               │   │
│  │  │ backtest_   │ │ tool_       │ │ hypotheses  │               │   │
│  │  │ results     │ │ definitions │ │             │               │   │
│  │  └─────────────┘ └─────────────┘ └─────────────┘               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    LINEAGE TRACKER                               │   │
│  │  • UUID-based identification                                     │   │
│  │  • Parent-child relationships                                    │   │
│  │  • Provenance metadata                                           │   │
│  │  • Version history                                               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    ARCHIVE MANAGER                               │   │
│  │  • 90-day inactivity archiving                                   │   │
│  │  • Compression for archived data                                 │   │
│  │  • Retrieval from archive                                        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## ChromaDB Collections

### Collection Schemas

#### research_findings

```python
RESEARCH_FINDINGS_SCHEMA = {
    "name": "research_findings",
    "metadata_fields": {
        "id": "uuid",
        "parent_id": "uuid | null",
        "source": "string",  # "market_research", "technical", etc.
        "topic": "string",
        "confidence": "float",
        "created_at": "datetime",
        "updated_at": "datetime",
        "session_id": "string",
        "agent_id": "string",
        "tags": "list[string]"
    },
    "embedding_field": "content",  # The finding text
    "distance_metric": "cosine"
}
```

#### strategy_library

```python
STRATEGY_LIBRARY_SCHEMA = {
    "name": "strategy_library",
    "metadata_fields": {
        "id": "uuid",
        "parent_id": "uuid | null",  # Research finding that led to this
        "name": "string",
        "version": "int",
        "status": "string",  # "draft", "tested", "approved", "deprecated"
        "asset_class": "string",
        "time_horizon": "string",
        "sharpe_ratio": "float",
        "max_drawdown": "float",
        "win_rate": "float",
        "created_at": "datetime",
        "approved_at": "datetime | null",
        "tags": "list[string]"
    },
    "embedding_field": "description",
    "additional_fields": {
        "code": "string",  # Strategy code
        "parameters": "json",
        "metrics": "json"
    }
}
```

#### lessons_learned

```python
LESSONS_LEARNED_SCHEMA = {
    "name": "lessons_learned",
    "metadata_fields": {
        "id": "uuid",
        "strategy_id": "uuid",
        "iteration": "int",
        "failure_type": "string",  # "quality_gate", "backtest", "optimization"
        "severity": "string",  # "minor", "major", "critical"
        "created_at": "datetime",
        "resolved": "bool",
        "resolution_id": "uuid | null"
    },
    "embedding_field": "lesson",  # The lesson text
    "additional_fields": {
        "feedback": "json",
        "improvement_applied": "string"
    }
}
```

#### market_regimes

```python
MARKET_REGIMES_SCHEMA = {
    "name": "market_regimes",
    "metadata_fields": {
        "id": "uuid",
        "regime_type": "string",  # "bull", "bear", "sideways", "volatile"
        "start_date": "date",
        "end_date": "date | null",
        "volatility": "float",
        "trend_strength": "float",
        "created_at": "datetime"
    },
    "embedding_field": "description",
    "additional_fields": {
        "indicators": "json",
        "characteristics": "json"
    }
}
```

## Memory Manager Implementation

```python
import chromadb
from chromadb.config import Settings
from uuid import uuid4
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

class MemoryManager:
    """Central memory management for the trading system."""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory
        ))
        self._init_collections()
        self.lineage_tracker = LineageTracker()
    
    def _init_collections(self):
        """Initialize all collections."""
        self.research_findings = self.client.get_or_create_collection(
            name="research_findings",
            metadata={"hnsw:space": "cosine"}
        )
        self.strategy_library = self.client.get_or_create_collection(
            name="strategy_library",
            metadata={"hnsw:space": "cosine"}
        )
        self.lessons_learned = self.client.get_or_create_collection(
            name="lessons_learned",
            metadata={"hnsw:space": "cosine"}
        )
        self.market_regimes = self.client.get_or_create_collection(
            name="market_regimes",
            metadata={"hnsw:space": "cosine"}
        )
        self.tool_definitions = self.client.get_or_create_collection(
            name="tool_definitions",
            metadata={"hnsw:space": "cosine"}
        )
    
    # ==================== Research Findings ====================
    
    async def store_research_finding(
        self,
        content: str,
        source: str,
        topic: str,
        confidence: float,
        parent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        tags: List[str] = []
    ) -> str:
        """Store a research finding with embedding."""
        finding_id = str(uuid4())
        
        metadata = {
            "id": finding_id,
            "parent_id": parent_id,
            "source": source,
            "topic": topic,
            "confidence": confidence,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "session_id": session_id,
            "agent_id": agent_id,
            "tags": ",".join(tags)
        }
        
        self.research_findings.add(
            documents=[content],
            metadatas=[metadata],
            ids=[finding_id]
        )
        
        # Track lineage
        if parent_id:
            self.lineage_tracker.add_relationship(parent_id, finding_id)
        
        return finding_id
    
    async def query_research_findings(
        self,
        query: str,
        n_results: int = 10,
        source_filter: Optional[str] = None,
        min_confidence: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Query research findings by semantic similarity."""
        where_filter = {}
        if source_filter:
            where_filter["source"] = source_filter
        if min_confidence > 0:
            where_filter["confidence"] = {"$gte": min_confidence}
        
        results = self.research_findings.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter if where_filter else None
        )
        
        return self._format_results(results)
    
    # ==================== Strategy Library ====================
    
    async def store_strategy(
        self,
        name: str,
        description: str,
        code: str,
        parameters: dict,
        metrics: dict,
        parent_id: Optional[str] = None,
        status: str = "draft"
    ) -> str:
        """Store a trading strategy."""
        strategy_id = str(uuid4())
        
        # Check for existing versions
        existing = self.strategy_library.get(
            where={"name": name}
        )
        version = len(existing["ids"]) + 1 if existing["ids"] else 1
        
        metadata = {
            "id": strategy_id,
            "parent_id": parent_id,
            "name": name,
            "version": version,
            "status": status,
            "asset_class": parameters.get("asset_class", "unknown"),
            "time_horizon": parameters.get("time_horizon", "unknown"),
            "sharpe_ratio": metrics.get("sharpe_ratio", 0.0),
            "max_drawdown": metrics.get("max_drawdown", 0.0),
            "win_rate": metrics.get("win_rate", 0.0),
            "created_at": datetime.now().isoformat(),
            "approved_at": None,
            "code": code,
            "parameters": str(parameters),
            "metrics": str(metrics)
        }
        
        self.strategy_library.add(
            documents=[description],
            metadatas=[metadata],
            ids=[strategy_id]
        )
        
        if parent_id:
            self.lineage_tracker.add_relationship(parent_id, strategy_id)
        
        return strategy_id
    
    async def get_similar_strategies(
        self,
        description: str,
        n_results: int = 5,
        status_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Find similar strategies by description."""
        where_filter = {}
        if status_filter:
            where_filter["status"] = status_filter
        
        results = self.strategy_library.query(
            query_texts=[description],
            n_results=n_results,
            where=where_filter if where_filter else None
        )
        
        return self._format_results(results)
    
    # ==================== Lessons Learned ====================
    
    async def store_lesson(
        self,
        lesson: str,
        strategy_id: str,
        iteration: int,
        failure_type: str,
        feedback: dict,
        severity: str = "major"
    ) -> str:
        """Store a lesson learned from a failed attempt."""
        lesson_id = str(uuid4())
        
        metadata = {
            "id": lesson_id,
            "strategy_id": strategy_id,
            "iteration": iteration,
            "failure_type": failure_type,
            "severity": severity,
            "created_at": datetime.now().isoformat(),
            "resolved": False,
            "resolution_id": None,
            "feedback": str(feedback)
        }
        
        self.lessons_learned.add(
            documents=[lesson],
            metadatas=[metadata],
            ids=[lesson_id]
        )
        
        self.lineage_tracker.add_relationship(strategy_id, lesson_id)
        
        return lesson_id
    
    async def get_relevant_lessons(
        self,
        context: str,
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Get lessons relevant to current context."""
        results = self.lessons_learned.query(
            query_texts=[context],
            n_results=n_results,
            where={"resolved": False}
        )
        
        return self._format_results(results)
    
    # ==================== Utility Methods ====================
    
    def _format_results(self, results: dict) -> List[Dict[str, Any]]:
        """Format ChromaDB results into list of dicts."""
        formatted = []
        for i, doc in enumerate(results["documents"][0]):
            item = {
                "id": results["ids"][0][i],
                "content": doc,
                "distance": results["distances"][0][i] if results.get("distances") else None,
                **results["metadatas"][0][i]
            }
            formatted.append(item)
        return formatted
    
    async def get_lineage(self, item_id: str) -> Dict[str, Any]:
        """Get full lineage of an item."""
        return self.lineage_tracker.get_lineage(item_id)


class LineageTracker:
    """Tracks parent-child relationships between memory items."""
    
    def __init__(self):
        self.relationships: Dict[str, List[str]] = {}  # parent -> children
        self.reverse_relationships: Dict[str, str] = {}  # child -> parent
    
    def add_relationship(self, parent_id: str, child_id: str):
        """Add a parent-child relationship."""
        if parent_id not in self.relationships:
            self.relationships[parent_id] = []
        self.relationships[parent_id].append(child_id)
        self.reverse_relationships[child_id] = parent_id
    
    def get_children(self, parent_id: str) -> List[str]:
        """Get all children of a parent."""
        return self.relationships.get(parent_id, [])
    
    def get_parent(self, child_id: str) -> Optional[str]:
        """Get parent of a child."""
        return self.reverse_relationships.get(child_id)
    
    def get_lineage(self, item_id: str) -> Dict[str, Any]:
        """Get full lineage tree for an item."""
        # Get ancestors
        ancestors = []
        current = item_id
        while current in self.reverse_relationships:
            parent = self.reverse_relationships[current]
            ancestors.append(parent)
            current = parent
        
        # Get descendants
        def get_descendants(node_id: str) -> Dict[str, Any]:
            children = self.get_children(node_id)
            return {
                "id": node_id,
                "children": [get_descendants(c) for c in children]
            }
        
        return {
            "item_id": item_id,
            "ancestors": list(reversed(ancestors)),
            "descendants": get_descendants(item_id)
        }
```

## Archive Manager

```python
class ArchiveManager:
    """Manages archiving of old memory items."""
    
    def __init__(self, memory: MemoryManager, archive_path: str = "./archive"):
        self.memory = memory
        self.archive_path = archive_path
        self.archive_after_days = 90
    
    async def run_archival(self):
        """Archive items older than threshold."""
        cutoff_date = datetime.now() - timedelta(days=self.archive_after_days)
        
        # Archive lessons learned
        old_lessons = self.memory.lessons_learned.get(
            where={"created_at": {"$lt": cutoff_date.isoformat()}}
        )
        
        if old_lessons["ids"]:
            await self._archive_items("lessons_learned", old_lessons)
            self.memory.lessons_learned.delete(ids=old_lessons["ids"])
        
        # Archive old research findings with low confidence
        old_findings = self.memory.research_findings.get(
            where={
                "$and": [
                    {"created_at": {"$lt": cutoff_date.isoformat()}},
                    {"confidence": {"$lt": 0.5}}
                ]
            }
        )
        
        if old_findings["ids"]:
            await self._archive_items("research_findings", old_findings)
            self.memory.research_findings.delete(ids=old_findings["ids"])
    
    async def _archive_items(self, collection_name: str, items: dict):
        """Archive items to compressed storage."""
        archive_file = f"{self.archive_path}/{collection_name}_{datetime.now().strftime('%Y%m%d')}.json.gz"
        
        import gzip
        import json
        
        with gzip.open(archive_file, 'wt') as f:
            json.dump(items, f)
    
    async def retrieve_from_archive(
        self,
        collection_name: str,
        query: str
    ) -> List[Dict[str, Any]]:
        """Search archived items."""
        # Implementation for searching compressed archives
        pass
```

## Session State Management

```python
from langgraph.checkpoint import MemorySaver

class SessionStateManager:
    """Manages session state for LangGraph workflows."""
    
    def __init__(self):
        self.checkpointer = MemorySaver()
    
    def get_checkpointer(self):
        """Get the checkpointer for LangGraph."""
        return self.checkpointer
    
    async def save_checkpoint(
        self,
        thread_id: str,
        state: dict,
        checkpoint_id: Optional[str] = None
    ):
        """Save a checkpoint."""
        config = {"configurable": {"thread_id": thread_id}}
        if checkpoint_id:
            config["configurable"]["checkpoint_id"] = checkpoint_id
        
        await self.checkpointer.aput(config, state)
    
    async def load_checkpoint(
        self,
        thread_id: str,
        checkpoint_id: Optional[str] = None
    ) -> Optional[dict]:
        """Load a checkpoint."""
        config = {"configurable": {"thread_id": thread_id}}
        if checkpoint_id:
            config["configurable"]["checkpoint_id"] = checkpoint_id
        
        return await self.checkpointer.aget(config)
    
    async def list_checkpoints(self, thread_id: str) -> List[str]:
        """List all checkpoints for a thread."""
        config = {"configurable": {"thread_id": thread_id}}
        return [c.checkpoint_id async for c in self.checkpointer.alist(config)]
```

## Integration with Pipeline

```python
# Memory initialization in main workflow
async def init_memory() -> MemoryManager:
    """Initialize memory system."""
    memory = MemoryManager(persist_directory="./data/chroma_db")
    return memory

# Usage in research phase
async def research_phase(state: PipelineState) -> PipelineState:
    memory = state["memory"]
    
    # Query relevant past research
    relevant_research = await memory.query_research_findings(
        query=state["research_objective"],
        n_results=5,
        min_confidence=0.7
    )
    
    # Query relevant lessons
    relevant_lessons = await memory.get_relevant_lessons(
        context=state["research_objective"],
        n_results=3
    )
    
    # Execute research with context
    findings = await research_swarm.execute_research(
        objective=state["research_objective"],
        context={
            "past_research": relevant_research,
            "lessons": relevant_lessons
        }
    )
    
    # Store new findings
    for finding in findings:
        await memory.store_research_finding(
            content=finding["content"],
            source=finding["source"],
            topic=finding["topic"],
            confidence=finding["confidence"],
            session_id=state["session_id"]
        )
    
    return {**state, "research_findings": findings}
```
