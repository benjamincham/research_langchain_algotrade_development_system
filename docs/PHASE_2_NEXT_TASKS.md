# Phase 2: Next Two Tasks - Detailed Implementation Plan

**Date**: 2026-01-18  
**Status**: Ready to implement  
**Prerequisites**: ✅ MemoryManager implemented and tested

---

## Task 2.3: Implement Collection Wrapper Classes

### Overview

Create type-safe wrapper classes for each of the 4 ChromaDB collections. These classes provide:
- **Pydantic schemas** for type validation
- **Helper methods** for common operations
- **Business logic** specific to each collection
- **Better developer experience** with autocomplete and type hints

### Architecture

```
src/memory/
├── memory_manager.py          # ✅ Already implemented
├── collections/
│   ├── __init__.py
│   ├── base_collection.py     # Abstract base class
│   ├── research_findings.py   # Research findings collection
│   ├── strategy_library.py    # Strategy library collection
│   ├── lessons_learned.py     # Lessons learned collection
│   └── market_regimes.py      # Market regimes collection
```

---

## 2.3.1: BaseCollection (Abstract Base Class)

### Purpose
Provide common functionality for all collection wrappers.

### Implementation

**File**: `src/memory/collections/base_collection.py`

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from chromadb import Collection
from loguru import logger


class BaseCollection(ABC):
    """
    Abstract base class for all collection wrappers.
    
    Provides common functionality:
    - Add, get, update, delete operations
    - Search with metadata filtering
    - Batch operations
    - Export/import
    """
    
    def __init__(self, collection: Collection):
        """
        Initialize collection wrapper.
        
        Args:
            collection: ChromaDB collection instance
        """
        self.collection = collection
        self.collection_name = collection.name
    
    @abstractmethod
    def get_schema(self) -> type[BaseModel]:
        """
        Get the Pydantic schema for this collection.
        
        Returns:
            Pydantic model class
        """
        pass
    
    def add(
        self,
        doc_id: str,
        content: str,
        metadata: Dict[str, Any]
    ) -> None:
        """Add a document to the collection."""
        # Validate metadata against schema
        schema = self.get_schema()
        validated = schema(**metadata)
        
        self.collection.add(
            ids=[doc_id],
            documents=[content],
            metadatas=[validated.model_dump()]
        )
        logger.debug(f"Added {doc_id} to {self.collection_name}")
    
    def get(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by ID."""
        result = self.collection.get(ids=[doc_id])
        if result['ids']:
            return {
                'id': result['ids'][0],
                'document': result['documents'][0],
                'metadata': result['metadatas'][0]
            }
        return None
    
    def search(
        self,
        query: str,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search documents by semantic similarity."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where
        )
        
        # Convert to list of dicts
        documents = []
        for i in range(len(results['ids'][0])):
            documents.append({
                'id': results['ids'][0][i],
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })
        return documents
    
    def update_metadata(
        self,
        doc_id: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """Update metadata for a document."""
        try:
            # Validate metadata
            schema = self.get_schema()
            validated = schema(**metadata)
            
            self.collection.update(
                ids=[doc_id],
                metadatas=[validated.model_dump()]
            )
            return True
        except Exception as e:
            logger.error(f"Error updating metadata: {e}")
            return False
    
    def delete(self, doc_id: str) -> bool:
        """Delete a document."""
        try:
            self.collection.delete(ids=[doc_id])
            return True
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False
    
    def count(self) -> int:
        """Get document count."""
        return self.collection.count()
    
    def batch_add(
        self,
        documents: List[Dict[str, Any]]
    ) -> None:
        """Add multiple documents at once."""
        ids = [doc['id'] for doc in documents]
        contents = [doc['content'] for doc in documents]
        metadatas = [doc['metadata'] for doc in documents]
        
        # Validate all metadata
        schema = self.get_schema()
        validated_metadatas = [
            schema(**meta).model_dump()
            for meta in metadatas
        ]
        
        self.collection.add(
            ids=ids,
            documents=contents,
            metadatas=validated_metadatas
        )
        logger.info(f"Batch added {len(documents)} documents to {self.collection_name}")
```

**Acceptance Criteria**:
- ✅ Abstract base class with common methods
- ✅ Pydantic validation for all metadata
- ✅ CRUD operations (add, get, update, delete)
- ✅ Search with metadata filtering
- ✅ Batch operations
- ✅ Error handling and logging

---

## 2.3.2: ResearchFindingsCollection

### Purpose
Manage research findings from the research swarm.

### Pydantic Schema

```python
from pydantic import BaseModel, Field
from typing import Literal
from datetime import datetime


class ResearchFindingMetadata(BaseModel):
    """Metadata schema for research findings."""
    
    ticker: str = Field(..., description="Stock ticker symbol")
    type: Literal["technical", "fundamental", "sentiment", "pattern"] = Field(
        ..., description="Type of research finding"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    agent_id: str = Field(..., description="ID of the agent that generated this finding")
    timestamp: str = Field(..., description="ISO format timestamp")
    source: str = Field(..., description="Data source (e.g., yahoo_finance)")
    timeframe: str = Field(..., description="Timeframe (e.g., 1d, 1h)")
    tags: List[str] = Field(default_factory=list, description="Additional tags")
```

### Implementation

**File**: `src/memory/collections/research_findings.py`

```python
from .base_collection import BaseCollection
from pydantic import BaseModel, Field
from typing import List, Literal, Dict, Any


class ResearchFindingMetadata(BaseModel):
    """Metadata schema for research findings."""
    ticker: str
    type: Literal["technical", "fundamental", "sentiment", "pattern"]
    confidence: float = Field(ge=0.0, le=1.0)
    agent_id: str
    timestamp: str
    source: str
    timeframe: str
    tags: List[str] = Field(default_factory=list)


class ResearchFindingsCollection(BaseCollection):
    """Collection wrapper for research findings."""
    
    def get_schema(self) -> type[BaseModel]:
        return ResearchFindingMetadata
    
    def get_by_ticker(
        self,
        ticker: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get all findings for a specific ticker."""
        return self.search(
            query=f"research findings for {ticker}",
            n_results=limit,
            where={"ticker": ticker}
        )
    
    def get_by_type(
        self,
        finding_type: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get all findings of a specific type."""
        return self.search(
            query=f"{finding_type} research findings",
            n_results=limit,
            where={"type": finding_type}
        )
    
    def get_high_confidence(
        self,
        min_confidence: float = 0.8,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get high-confidence findings."""
        return self.search(
            query="high confidence research findings",
            n_results=limit,
            where={"confidence": {"$gte": min_confidence}}
        )
```

**Acceptance Criteria**:
- ✅ Pydantic schema with validation
- ✅ Helper methods (get_by_ticker, get_by_type, get_high_confidence)
- ✅ Inherits from BaseCollection
- ✅ Type hints and documentation

---

## 2.3.3: StrategyLibraryCollection

### Pydantic Schema

```python
class StrategyMetadata(BaseModel):
    """Metadata schema for trading strategies."""
    
    name: str = Field(..., description="Strategy name")
    type: Literal["momentum", "mean_reversion", "breakout", "arbitrage"] = Field(
        ..., description="Strategy type"
    )
    tickers: List[str] = Field(..., description="Tickers this strategy applies to")
    timeframe: str = Field(..., description="Timeframe (e.g., 1d)")
    parameters: Dict[str, Any] = Field(..., description="Strategy parameters")
    code: str = Field(..., description="Strategy code (base64 encoded)")
    performance_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Performance metrics (sharpe_ratio, max_drawdown, etc.)"
    )
    created_at: str = Field(..., description="ISO format timestamp")
    updated_at: str = Field(..., description="ISO format timestamp")
    parent_strategy_id: Optional[str] = Field(None, description="Parent strategy ID if derived")
    tags: List[str] = Field(default_factory=list)
```

### Helper Methods

```python
def get_by_type(self, strategy_type: str) -> List[Dict[str, Any]]
def get_top_performers(self, metric: str = "sharpe_ratio", limit: int = 10) -> List[Dict[str, Any]]
def get_for_ticker(self, ticker: str) -> List[Dict[str, Any]]
def update_performance(self, strategy_id: str, metrics: Dict[str, float]) -> bool
```

---

## 2.3.4: LessonsLearnedCollection

### Pydantic Schema

```python
class LessonMetadata(BaseModel):
    """Metadata schema for lessons learned."""
    
    type: Literal["failure", "success", "optimization", "insight"] = Field(
        ..., description="Type of lesson"
    )
    severity: Literal["low", "medium", "high", "critical"] = Field(
        ..., description="Severity/importance"
    )
    context: Dict[str, Any] = Field(..., description="Context (ticker, strategy, etc.)")
    timestamp: str = Field(..., description="ISO format timestamp")
    tags: List[str] = Field(default_factory=list)
```

### Helper Methods

```python
def get_failures(self, limit: int = 100) -> List[Dict[str, Any]]
def get_critical_lessons(self, limit: int = 100) -> List[Dict[str, Any]]
def get_by_context(self, context_key: str, context_value: str) -> List[Dict[str, Any]]
```

---

## 2.3.5: MarketRegimesCollection

### Pydantic Schema

```python
class MarketRegimeMetadata(BaseModel):
    """Metadata schema for market regimes."""
    
    regime_type: Literal[
        "bull_low_vol", "bull_high_vol",
        "bear_low_vol", "bear_high_vol",
        "sideways_low_vol", "sideways_high_vol",
        "crisis"
    ] = Field(..., description="Market regime type")
    start_date: str = Field(..., description="Regime start date")
    end_date: Optional[str] = Field(None, description="Regime end date (None if current)")
    volatility: float = Field(..., ge=0.0, description="Volatility level")
    indicators: Dict[str, float] = Field(..., description="Technical indicators")
    tickers: List[str] = Field(..., description="Tickers analyzed")
    confidence: float = Field(..., ge=0.0, le=1.0)
```

### Helper Methods

```python
def get_current_regime(self, ticker: str) -> Optional[Dict[str, Any]]
def get_by_regime_type(self, regime_type: str) -> List[Dict[str, Any]]
def get_historical_regimes(self, ticker: str, start_date: str, end_date: str) -> List[Dict[str, Any]]
```

---

## Task 2.4: Implement Lineage Tracker

### Overview

Track parent-child relationships between documents to enable:
- **Strategy evolution tracking**: Which strategies were derived from which
- **Research lineage**: Which findings led to which strategies
- **Dependency analysis**: What depends on what
- **Impact analysis**: If we change X, what's affected?

### Architecture

```
src/memory/
├── lineage_tracker.py         # Lineage tracking system
└── lineage_graph.json         # Persisted lineage graph
```

---

## 2.4.1: Lineage Data Structure

### Graph Structure

```python
{
    "nodes": {
        "finding_001": {
            "type": "research_finding",
            "collection": "research_findings",
            "created_at": "2026-01-18T10:00:00",
            "metadata": {...}
        },
        "strategy_001": {
            "type": "strategy",
            "collection": "strategy_library",
            "created_at": "2026-01-18T11:00:00",
            "metadata": {...}
        }
    },
    "edges": [
        {
            "from": "finding_001",
            "to": "strategy_001",
            "relationship": "derived_from",
            "timestamp": "2026-01-18T11:00:00"
        }
    ]
}
```

### Relationships

- **derived_from**: Strategy derived from finding/strategy
- **based_on**: Strategy based on research
- **refined_from**: Strategy refined from previous version
- **informed_by**: Decision informed by lesson
- **applies_to**: Strategy applies to regime

---

## 2.4.2: LineageTracker Implementation

**File**: `src/memory/lineage_tracker.py`

```python
from typing import Dict, List, Optional, Literal
from pathlib import Path
import json
from datetime import datetime
from loguru import logger


RelationshipType = Literal[
    "derived_from",
    "based_on",
    "refined_from",
    "informed_by",
    "applies_to"
]


class LineageTracker:
    """
    Track parent-child relationships between documents.
    
    Maintains a directed acyclic graph (DAG) of relationships.
    """
    
    def __init__(self, persist_path: str = "./data/lineage_graph.json"):
        """
        Initialize lineage tracker.
        
        Args:
            persist_path: Path to persist the lineage graph
        """
        self.persist_path = Path(persist_path)
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing graph or create new
        if self.persist_path.exists():
            with open(self.persist_path, 'r') as f:
                self.graph = json.load(f)
        else:
            self.graph = {"nodes": {}, "edges": []}
        
        logger.info(f"LineageTracker initialized with {len(self.graph['nodes'])} nodes")
    
    def add_node(
        self,
        node_id: str,
        node_type: str,
        collection: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Add a node to the lineage graph.
        
        Args:
            node_id: Unique identifier
            node_type: Type of node (research_finding, strategy, etc.)
            collection: ChromaDB collection name
            metadata: Additional metadata
        """
        self.graph['nodes'][node_id] = {
            "type": node_type,
            "collection": collection,
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self._save()
        logger.debug(f"Added node: {node_id}")
    
    def add_edge(
        self,
        from_id: str,
        to_id: str,
        relationship: RelationshipType
    ) -> None:
        """
        Add an edge (relationship) between two nodes.
        
        Args:
            from_id: Source node ID
            to_id: Target node ID
            relationship: Type of relationship
        """
        # Check for circular dependencies
        if self._would_create_cycle(from_id, to_id):
            raise ValueError(f"Adding edge {from_id} -> {to_id} would create a cycle")
        
        edge = {
            "from": from_id,
            "to": to_id,
            "relationship": relationship,
            "timestamp": datetime.now().isoformat()
        }
        self.graph['edges'].append(edge)
        self._save()
        logger.debug(f"Added edge: {from_id} --[{relationship}]--> {to_id}")
    
    def get_parents(self, node_id: str) -> List[Dict]:
        """
        Get all parent nodes (nodes that this node derives from).
        
        Args:
            node_id: Node ID
        
        Returns:
            List of parent nodes with relationship info
        """
        parents = []
        for edge in self.graph['edges']:
            if edge['to'] == node_id:
                parent_node = self.graph['nodes'].get(edge['from'])
                if parent_node:
                    parents.append({
                        "node_id": edge['from'],
                        "relationship": edge['relationship'],
                        "node_data": parent_node
                    })
        return parents
    
    def get_children(self, node_id: str) -> List[Dict]:
        """
        Get all child nodes (nodes that derive from this node).
        
        Args:
            node_id: Node ID
        
        Returns:
            List of child nodes with relationship info
        """
        children = []
        for edge in self.graph['edges']:
            if edge['from'] == node_id:
                child_node = self.graph['nodes'].get(edge['to'])
                if child_node:
                    children.append({
                        "node_id": edge['to'],
                        "relationship": edge['relationship'],
                        "node_data": child_node
                    })
        return children
    
    def get_lineage(self, node_id: str, max_depth: int = 10) -> Dict:
        """
        Get complete lineage (ancestors and descendants) for a node.
        
        Args:
            node_id: Node ID
            max_depth: Maximum depth to traverse
        
        Returns:
            Dictionary with ancestors and descendants
        """
        return {
            "node_id": node_id,
            "ancestors": self._get_ancestors(node_id, max_depth),
            "descendants": self._get_descendants(node_id, max_depth)
        }
    
    def _get_ancestors(
        self,
        node_id: str,
        max_depth: int,
        current_depth: int = 0,
        visited: Optional[set] = None
    ) -> List[Dict]:
        """Recursively get all ancestors."""
        if visited is None:
            visited = set()
        
        if current_depth >= max_depth or node_id in visited:
            return []
        
        visited.add(node_id)
        ancestors = []
        
        for parent in self.get_parents(node_id):
            ancestors.append({
                **parent,
                "depth": current_depth + 1
            })
            # Recursively get parent's ancestors
            ancestors.extend(
                self._get_ancestors(
                    parent['node_id'],
                    max_depth,
                    current_depth + 1,
                    visited
                )
            )
        
        return ancestors
    
    def _get_descendants(
        self,
        node_id: str,
        max_depth: int,
        current_depth: int = 0,
        visited: Optional[set] = None
    ) -> List[Dict]:
        """Recursively get all descendants."""
        if visited is None:
            visited = set()
        
        if current_depth >= max_depth or node_id in visited:
            return []
        
        visited.add(node_id)
        descendants = []
        
        for child in self.get_children(node_id):
            descendants.append({
                **child,
                "depth": current_depth + 1
            })
            # Recursively get child's descendants
            descendants.extend(
                self._get_descendants(
                    child['node_id'],
                    max_depth,
                    current_depth + 1,
                    visited
                )
            )
        
        return descendants
    
    def _would_create_cycle(self, from_id: str, to_id: str) -> bool:
        """Check if adding an edge would create a cycle."""
        # If to_id can reach from_id, adding edge would create cycle
        descendants = self._get_descendants(to_id, max_depth=100)
        return any(d['node_id'] == from_id for d in descendants)
    
    def _save(self) -> None:
        """Persist graph to disk."""
        with open(self.persist_path, 'w') as f:
            json.dump(self.graph, f, indent=2)
    
    def export_dot(self, output_path: str) -> None:
        """
        Export lineage graph to DOT format for visualization.
        
        Args:
            output_path: Path to output .dot file
        """
        lines = ["digraph Lineage {"]
        lines.append("  rankdir=LR;")
        lines.append("  node [shape=box];")
        
        # Add nodes
        for node_id, node_data in self.graph['nodes'].items():
            label = f"{node_id}\\n({node_data['type']})"
            lines.append(f'  "{node_id}" [label="{label}"];')
        
        # Add edges
        for edge in self.graph['edges']:
            label = edge['relationship']
            lines.append(f'  "{edge["from"]}" -> "{edge["to"]}" [label="{label}"];')
        
        lines.append("}")
        
        with open(output_path, 'w') as f:
            f.write("\n".join(lines))
        
        logger.info(f"Exported lineage graph to {output_path}")
```

**Acceptance Criteria**:
- ✅ Add nodes and edges
- ✅ Get parents and children
- ✅ Get complete lineage (ancestors + descendants)
- ✅ Prevent circular dependencies
- ✅ Persist to JSON
- ✅ Export to DOT format for visualization
- ✅ Recursive traversal with max depth

---

## Testing Plan

### Unit Tests (Task 2.5)

**File**: `tests/unit/test_collections.py`

```python
def test_research_findings_collection_add()
def test_research_findings_collection_get_by_ticker()
def test_research_findings_collection_get_high_confidence()
def test_strategy_library_collection_add()
def test_strategy_library_collection_get_top_performers()
def test_lessons_learned_collection_add()
def test_lessons_learned_collection_get_failures()
def test_market_regimes_collection_add()
def test_market_regimes_collection_get_current_regime()
```

**File**: `tests/unit/test_lineage_tracker.py`

```python
def test_lineage_tracker_add_node()
def test_lineage_tracker_add_edge()
def test_lineage_tracker_get_parents()
def test_lineage_tracker_get_children()
def test_lineage_tracker_get_lineage()
def test_lineage_tracker_prevent_cycles()
def test_lineage_tracker_persistence()
def test_lineage_tracker_export_dot()
```

### Integration Tests (Task 2.6)

**File**: `tests/integration/test_memory_system.py`

```python
def test_end_to_end_research_to_strategy_flow()
def test_lineage_tracking_with_collections()
def test_memory_system_with_multiple_iterations()
```

---

## Effort Estimate

| Task | Description | Effort | Tests |
|------|-------------|--------|-------|
| 2.3.1 | BaseCollection | 2 hours | 5 tests |
| 2.3.2 | ResearchFindingsCollection | 1 hour | 3 tests |
| 2.3.3 | StrategyLibraryCollection | 1 hour | 3 tests |
| 2.3.4 | LessonsLearnedCollection | 1 hour | 3 tests |
| 2.3.5 | MarketRegimesCollection | 1 hour | 3 tests |
| 2.4 | LineageTracker | 3 hours | 8 tests |
| 2.5 | Unit Tests | 2 hours | 25 tests |
| 2.6 | Integration Tests | 1 hour | 3 tests |
| **Total** | **12 hours** | **53 tests** |

---

## Summary

**Task 2.3: Collection Wrapper Classes**
- 5 classes (1 base + 4 concrete)
- Pydantic schemas for type safety
- Helper methods for common operations
- ~400 lines of code

**Task 2.4: Lineage Tracker**
- DAG-based graph structure
- Prevent circular dependencies
- Recursive lineage traversal
- Export to DOT format
- ~300 lines of code

**Total**: ~700 lines of production code + 53 tests

**Ready to implement**: All specifications complete, schemas defined, acceptance criteria clear.
