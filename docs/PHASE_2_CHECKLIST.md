# Phase 2: Memory System - Passing Criteria Checklist

**Phase**: Memory System  
**Status**: ⏳ Not Started (30% Design Complete)  
**Estimated Effort**: 3.5 days  
**Dependencies**: Phase 1 (Core Infrastructure)  
**Target Completion**: 2026-01-23

---

## Overview

Phase 2 implements the persistent memory system using ChromaDB, enabling the system to store and retrieve research findings, strategies, lessons learned, and market regime data. The memory system includes lineage tracking for parent-child relationships, archiving for old data, and session state management for workflow persistence.

---

## Functional Objectives

### 2.1 ChromaDB Integration

**Objective**: Connect to ChromaDB and manage collections

**Tasks**:
- [ ] Install ChromaDB (`chromadb>=0.4.0`)
- [ ] Create `MemoryManager` class in `memory/memory_manager.py`
- [ ] Initialize ChromaDB client with persistent storage
- [ ] Implement collection creation and management
- [ ] Implement CRUD operations (Create, Read, Update, Delete)
- [ ] Add error handling for database operations
- [ ] Add logging for all database operations

**Acceptance Criteria**:
- [ ] ChromaDB client initializes successfully
- [ ] Client uses persistent storage (not in-memory)
- [ ] Can create collections programmatically
- [ ] Can add documents to collections
- [ ] Can query documents by similarity
- [ ] Can query documents by metadata filters
- [ ] Can update existing documents
- [ ] Can delete documents
- [ ] Database operations are logged
- [ ] Errors are caught and logged with clear messages

**Test Cases**:

```python
# Test 1: Initialize ChromaDB client
def test_chromadb_initialization():
    manager = MemoryManager()
    assert manager.client is not None
    assert manager.client._settings.persist_directory is not None

# Test 2: Create collection
def test_create_collection():
    manager = MemoryManager()
    collection = manager.create_collection("test_collection")
    assert collection is not None
    assert collection.name == "test_collection"

# Test 3: Add document
def test_add_document():
    manager = MemoryManager()
    collection = manager.get_collection("research_findings")
    
    doc_id = collection.add(
        documents=["AAPL shows strong momentum"],
        metadatas=[{"ticker": "AAPL", "type": "technical"}],
        ids=["finding_001"]
    )
    
    assert doc_id is not None

# Test 4: Query by similarity
def test_query_similarity():
    manager = MemoryManager()
    collection = manager.get_collection("research_findings")
    
    results = collection.query(
        query_texts=["Apple stock momentum"],
        n_results=5
    )
    
    assert len(results["documents"]) > 0
    assert "AAPL" in results["documents"][0][0]

# Test 5: Query by metadata
def test_query_metadata():
    manager = MemoryManager()
    collection = manager.get_collection("research_findings")
    
    results = collection.get(
        where={"ticker": "AAPL"}
    )
    
    assert len(results["documents"]) > 0

# Test 6: Update document
def test_update_document():
    manager = MemoryManager()
    collection = manager.get_collection("research_findings")
    
    collection.update(
        ids=["finding_001"],
        documents=["AAPL shows strong momentum (updated)"]
    )
    
    result = collection.get(ids=["finding_001"])
    assert "updated" in result["documents"][0]

# Test 7: Delete document
def test_delete_document():
    manager = MemoryManager()
    collection = manager.get_collection("research_findings")
    
    collection.delete(ids=["finding_001"])
    
    result = collection.get(ids=["finding_001"])
    assert len(result["documents"]) == 0

# Test 8: Error handling
def test_error_handling():
    manager = MemoryManager()
    
    # Try to get non-existent collection
    with pytest.raises(ValueError, match="Collection .* does not exist"):
        manager.get_collection("nonexistent_collection")
```

**Implementation Checklist**:
- [ ] `MemoryManager.__init__()` initializes ChromaDB client
- [ ] `MemoryManager.create_collection()` creates new collection
- [ ] `MemoryManager.get_collection()` retrieves existing collection
- [ ] `MemoryManager.list_collections()` lists all collections
- [ ] `MemoryManager.delete_collection()` deletes collection
- [ ] All methods have error handling
- [ ] All methods have logging

---

### 2.2 Research Findings Storage

**Objective**: Store and retrieve research findings with embeddings

**Tasks**:
- [ ] Create `ResearchFindingsCollection` class in `memory/collections/research_findings.py`
- [ ] Define metadata schema for research findings
- [ ] Implement `add_finding()` method
- [ ] Implement `get_finding()` method
- [ ] Implement `search_findings()` method (similarity search)
- [ ] Implement `filter_findings()` method (metadata filters)
- [ ] Add validation for finding data
- [ ] Add automatic embedding generation

**Acceptance Criteria**:
- [ ] Can store research finding with text and metadata
- [ ] Metadata includes: ticker, type (technical/fundamental/sentiment), confidence, timestamp, agent_id
- [ ] Embeddings are generated automatically
- [ ] Can retrieve finding by ID
- [ ] Can search findings by similarity (semantic search)
- [ ] Can filter findings by metadata (e.g., ticker="AAPL", type="technical")
- [ ] Can retrieve top N most relevant findings for a query
- [ ] Invalid data raises validation errors

**Test Cases**:

```python
# Test 1: Add research finding
def test_add_research_finding():
    collection = ResearchFindingsCollection()
    
    finding_id = collection.add_finding(
        text="AAPL shows strong upward momentum with RSI at 65",
        metadata={
            "ticker": "AAPL",
            "type": "technical",
            "confidence": 0.85,
            "agent_id": "technical_analyst_001"
        }
    )
    
    assert finding_id is not None
    assert finding_id.startswith("finding_")

# Test 2: Retrieve finding by ID
def test_get_finding():
    collection = ResearchFindingsCollection()
    
    finding = collection.get_finding("finding_001")
    
    assert finding is not None
    assert finding["text"] == "AAPL shows strong upward momentum with RSI at 65"
    assert finding["metadata"]["ticker"] == "AAPL"

# Test 3: Search findings by similarity
def test_search_findings():
    collection = ResearchFindingsCollection()
    
    results = collection.search_findings(
        query="Apple stock technical analysis",
        n_results=5
    )
    
    assert len(results) > 0
    assert results[0]["metadata"]["ticker"] == "AAPL"
    assert results[0]["metadata"]["type"] == "technical"

# Test 4: Filter findings by metadata
def test_filter_findings():
    collection = ResearchFindingsCollection()
    
    results = collection.filter_findings(
        filters={"ticker": "AAPL", "type": "technical"}
    )
    
    assert len(results) > 0
    for result in results:
        assert result["metadata"]["ticker"] == "AAPL"
        assert result["metadata"]["type"] == "technical"

# Test 5: Validation
def test_finding_validation():
    collection = ResearchFindingsCollection()
    
    # Missing required field
    with pytest.raises(ValueError, match="Missing required field: ticker"):
        collection.add_finding(
            text="Some finding",
            metadata={"type": "technical"}  # Missing ticker
        )
    
    # Invalid confidence
    with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
        collection.add_finding(
            text="Some finding",
            metadata={"ticker": "AAPL", "confidence": 1.5}
        )

# Test 6: Get related findings
def test_get_related_findings():
    collection = ResearchFindingsCollection()
    
    # Get findings related to a specific finding
    related = collection.get_related_findings(
        finding_id="finding_001",
        n_results=5
    )
    
    assert len(related) > 0
    # Should not include the original finding
    assert all(r["id"] != "finding_001" for r in related)
```

**Metadata Schema**:
```python
{
    "ticker": str,           # Required: Stock ticker (e.g., "AAPL")
    "type": str,             # Required: "technical" | "fundamental" | "sentiment" | "pattern"
    "confidence": float,     # Required: 0.0 to 1.0
    "agent_id": str,         # Required: ID of agent that created finding
    "timestamp": str,        # Auto-generated: ISO 8601 timestamp
    "source": str,           # Optional: Data source
    "timeframe": str,        # Optional: "1d" | "1w" | "1m" | "1y"
    "tags": List[str]        # Optional: Custom tags
}
```

---

### 2.3 Strategy Library

**Objective**: Store and retrieve trading strategies with code and metadata

**Tasks**:
- [ ] Create `StrategyLibraryCollection` class in `memory/collections/strategy_library.py`
- [ ] Define metadata schema for strategies
- [ ] Implement `add_strategy()` method
- [ ] Implement `get_strategy()` method
- [ ] Implement `search_strategies()` method
- [ ] Implement `filter_strategies()` method
- [ ] Add validation for strategy data
- [ ] Add code storage and retrieval

**Acceptance Criteria**:
- [ ] Can store strategy with code, description, and metadata
- [ ] Metadata includes: name, type, tickers, timeframe, parameters, performance_metrics
- [ ] Can retrieve strategy by ID
- [ ] Can search strategies by similarity (description)
- [ ] Can filter strategies by metadata (e.g., type="momentum", ticker="AAPL")
- [ ] Can retrieve strategies with similar characteristics
- [ ] Code is stored securely and can be retrieved
- [ ] Invalid data raises validation errors

**Test Cases**:

```python
# Test 1: Add strategy
def test_add_strategy():
    collection = StrategyLibraryCollection()
    
    strategy_id = collection.add_strategy(
        name="AAPL Momentum Strategy",
        description="Momentum-based strategy for AAPL using RSI and MACD",
        code="class AAPLMomentumStrategy(bt.Strategy): ...",
        metadata={
            "type": "momentum",
            "tickers": ["AAPL"],
            "timeframe": "1d",
            "parameters": {"rsi_period": 14, "macd_fast": 12},
            "performance_metrics": {
                "sharpe_ratio": 1.8,
                "max_drawdown": 0.12,
                "win_rate": 0.65
            }
        }
    )
    
    assert strategy_id is not None
    assert strategy_id.startswith("strategy_")

# Test 2: Retrieve strategy
def test_get_strategy():
    collection = StrategyLibraryCollection()
    
    strategy = collection.get_strategy("strategy_001")
    
    assert strategy is not None
    assert strategy["name"] == "AAPL Momentum Strategy"
    assert "class AAPLMomentumStrategy" in strategy["code"]
    assert strategy["metadata"]["type"] == "momentum"

# Test 3: Search strategies by description
def test_search_strategies():
    collection = StrategyLibraryCollection()
    
    results = collection.search_strategies(
        query="momentum strategy using technical indicators",
        n_results=5
    )
    
    assert len(results) > 0
    assert results[0]["metadata"]["type"] == "momentum"

# Test 4: Filter strategies by performance
def test_filter_strategies_by_performance():
    collection = StrategyLibraryCollection()
    
    results = collection.filter_strategies(
        filters={
            "type": "momentum",
            "performance_metrics.sharpe_ratio": {"$gte": 1.5}
        }
    )
    
    assert len(results) > 0
    for result in results:
        assert result["metadata"]["performance_metrics"]["sharpe_ratio"] >= 1.5

# Test 5: Get similar strategies
def test_get_similar_strategies():
    collection = StrategyLibraryCollection()
    
    similar = collection.get_similar_strategies(
        strategy_id="strategy_001",
        n_results=5
    )
    
    assert len(similar) > 0
    # Should have similar types or tickers
    assert any(s["metadata"]["type"] == "momentum" for s in similar)

# Test 6: Update strategy performance
def test_update_strategy_performance():
    collection = StrategyLibraryCollection()
    
    collection.update_performance(
        strategy_id="strategy_001",
        performance_metrics={
            "sharpe_ratio": 2.1,
            "max_drawdown": 0.10,
            "win_rate": 0.70
        }
    )
    
    strategy = collection.get_strategy("strategy_001")
    assert strategy["metadata"]["performance_metrics"]["sharpe_ratio"] == 2.1
```

**Metadata Schema**:
```python
{
    "name": str,                    # Required: Strategy name
    "type": str,                    # Required: "momentum" | "mean_reversion" | "breakout" | "arbitrage"
    "tickers": List[str],           # Required: List of tickers
    "timeframe": str,               # Required: "1m" | "5m" | "1h" | "1d" | "1w"
    "parameters": Dict[str, Any],   # Required: Strategy parameters
    "performance_metrics": {        # Required: Performance metrics
        "sharpe_ratio": float,
        "sortino_ratio": float,
        "max_drawdown": float,
        "win_rate": float,
        "total_return": float
    },
    "timestamp": str,               # Auto-generated: ISO 8601 timestamp
    "version": str,                 # Optional: Version number
    "status": str,                  # Optional: "draft" | "testing" | "active" | "deprecated"
    "tags": List[str]               # Optional: Custom tags
}
```

---

### 2.4 Lessons Learned

**Objective**: Store and retrieve lessons learned from past strategies

**Tasks**:
- [ ] Create `LessonsLearnedCollection` class in `memory/collections/lessons_learned.py`
- [ ] Define metadata schema for lessons
- [ ] Implement `add_lesson()` method
- [ ] Implement `get_lesson()` method
- [ ] Implement `search_lessons()` method
- [ ] Implement `get_relevant_lessons()` method
- [ ] Add validation for lesson data

**Acceptance Criteria**:
- [ ] Can store lesson with text, context, and metadata
- [ ] Metadata includes: category, severity, strategy_id, outcome
- [ ] Can retrieve lesson by ID
- [ ] Can search lessons by similarity
- [ ] Can get relevant lessons for a given context
- [ ] Can filter lessons by category or severity
- [ ] Invalid data raises validation errors

**Test Cases**:

```python
# Test 1: Add lesson
def test_add_lesson():
    collection = LessonsLearnedCollection()
    
    lesson_id = collection.add_lesson(
        text="Momentum strategies perform poorly in low volatility environments",
        context="AAPL momentum strategy failed during Q2 2023 consolidation",
        metadata={
            "category": "market_conditions",
            "severity": "high",
            "strategy_id": "strategy_001",
            "outcome": "failure",
            "metrics": {"sharpe_ratio": 0.3, "max_drawdown": 0.25}
        }
    )
    
    assert lesson_id is not None
    assert lesson_id.startswith("lesson_")

# Test 2: Search lessons
def test_search_lessons():
    collection = LessonsLearnedCollection()
    
    results = collection.search_lessons(
        query="momentum strategy low volatility",
        n_results=5
    )
    
    assert len(results) > 0
    assert "momentum" in results[0]["text"].lower()

# Test 3: Get relevant lessons for context
def test_get_relevant_lessons():
    collection = LessonsLearnedCollection()
    
    lessons = collection.get_relevant_lessons(
        context="Developing momentum strategy for AAPL",
        n_results=5
    )
    
    assert len(lessons) > 0
    # Should include lessons about momentum strategies

# Test 4: Filter by severity
def test_filter_by_severity():
    collection = LessonsLearnedCollection()
    
    results = collection.filter_lessons(
        filters={"severity": "high"}
    )
    
    assert len(results) > 0
    for result in results:
        assert result["metadata"]["severity"] == "high"
```

**Metadata Schema**:
```python
{
    "category": str,          # Required: "market_conditions" | "parameters" | "risk_management" | "execution"
    "severity": str,          # Required: "low" | "medium" | "high" | "critical"
    "strategy_id": str,       # Optional: Related strategy ID
    "outcome": str,           # Required: "success" | "failure" | "mixed"
    "metrics": Dict,          # Optional: Related metrics
    "timestamp": str,         # Auto-generated: ISO 8601 timestamp
    "tags": List[str]         # Optional: Custom tags
}
```

---

### 2.5 Market Regimes

**Objective**: Store and retrieve market regime data

**Tasks**:
- [ ] Create `MarketRegimesCollection` class in `memory/collections/market_regimes.py`
- [ ] Define metadata schema for regimes
- [ ] Implement `add_regime()` method
- [ ] Implement `get_current_regime()` method
- [ ] Implement `get_regime_history()` method
- [ ] Implement `detect_regime()` method
- [ ] Add validation for regime data

**Acceptance Criteria**:
- [ ] Can store market regime with characteristics and metadata
- [ ] Metadata includes: regime_type, volatility, trend, start_date, end_date
- [ ] Can retrieve current market regime
- [ ] Can retrieve regime history for a ticker
- [ ] Can detect regime based on market data
- [ ] Invalid data raises validation errors

**Test Cases**:

```python
# Test 1: Add regime
def test_add_regime():
    collection = MarketRegimesCollection()
    
    regime_id = collection.add_regime(
        ticker="AAPL",
        regime_type="bull_high_vol",
        characteristics={
            "volatility": 0.35,
            "trend": "upward",
            "momentum": 0.75
        },
        metadata={
            "start_date": "2023-01-01",
            "end_date": "2023-03-31",
            "confidence": 0.90
        }
    )
    
    assert regime_id is not None

# Test 2: Get current regime
def test_get_current_regime():
    collection = MarketRegimesCollection()
    
    regime = collection.get_current_regime(ticker="AAPL")
    
    assert regime is not None
    assert regime["ticker"] == "AAPL"
    assert regime["regime_type"] in ["bull_high_vol", "bull_low_vol", "bear_high_vol", "bear_low_vol", "sideways", "crisis"]

# Test 3: Get regime history
def test_get_regime_history():
    collection = MarketRegimesCollection()
    
    history = collection.get_regime_history(
        ticker="AAPL",
        start_date="2023-01-01",
        end_date="2023-12-31"
    )
    
    assert len(history) > 0
    # Should be sorted by date
    assert history[0]["metadata"]["start_date"] <= history[-1]["metadata"]["start_date"]

# Test 4: Detect regime from data
def test_detect_regime():
    collection = MarketRegimesCollection()
    
    # Mock market data
    market_data = {
        "prices": [100, 102, 105, 103, 108],
        "volumes": [1000000, 1200000, 1100000, 1300000, 1400000]
    }
    
    regime = collection.detect_regime(ticker="AAPL", market_data=market_data)
    
    assert regime is not None
    assert "regime_type" in regime
    assert "confidence" in regime
```

**Metadata Schema**:
```python
{
    "ticker": str,                  # Required: Stock ticker
    "regime_type": str,             # Required: "bull_high_vol" | "bull_low_vol" | "bear_high_vol" | "bear_low_vol" | "sideways" | "crisis"
    "characteristics": {            # Required: Regime characteristics
        "volatility": float,        # 0.0 to 1.0
        "trend": str,               # "upward" | "downward" | "sideways"
        "momentum": float           # -1.0 to 1.0
    },
    "start_date": str,              # Required: ISO 8601 date
    "end_date": str,                # Optional: ISO 8601 date (None if current)
    "confidence": float,            # Required: 0.0 to 1.0
    "timestamp": str                # Auto-generated: ISO 8601 timestamp
}
```

---

### 2.6 Lineage Tracking

**Objective**: Track parent-child relationships between entities

**Tasks**:
- [ ] Complete `LineageTracker` class in `memory/lineage_tracker.py`
- [ ] Implement `link_parent_child()` method
- [ ] Implement `get_ancestors()` method
- [ ] Implement `get_descendants()` method
- [ ] Implement `get_siblings()` method
- [ ] Implement `get_lineage_tree()` method
- [ ] Add validation for lineage operations
- [ ] Add visualization helpers

**Acceptance Criteria**:
- [ ] Can link parent-child relationships
- [ ] Can query all ancestors of an entity
- [ ] Can query all descendants of an entity
- [ ] Can query siblings (entities with same parent)
- [ ] Can get full lineage tree
- [ ] Circular references are prevented
- [ ] Invalid operations raise errors

**Test Cases**:

```python
# Test 1: Link parent-child
def test_link_parent_child():
    tracker = LineageTracker()
    
    tracker.link_parent_child(
        parent_id="finding_001",
        parent_type="research_finding",
        child_id="strategy_001",
        child_type="strategy"
    )
    
    # Verify link exists
    children = tracker.get_children("finding_001")
    assert "strategy_001" in [c["id"] for c in children]

# Test 2: Get ancestors
def test_get_ancestors():
    tracker = LineageTracker()
    
    # Setup: finding_001 -> strategy_001 -> backtest_001
    tracker.link_parent_child("finding_001", "research_finding", "strategy_001", "strategy")
    tracker.link_parent_child("strategy_001", "strategy", "backtest_001", "backtest")
    
    ancestors = tracker.get_ancestors("backtest_001")
    
    assert len(ancestors) == 2
    assert ancestors[0]["id"] == "strategy_001"
    assert ancestors[1]["id"] == "finding_001"

# Test 3: Get descendants
def test_get_descendants():
    tracker = LineageTracker()
    
    # Setup: finding_001 -> strategy_001 -> backtest_001
    tracker.link_parent_child("finding_001", "research_finding", "strategy_001", "strategy")
    tracker.link_parent_child("strategy_001", "strategy", "backtest_001", "backtest")
    
    descendants = tracker.get_descendants("finding_001")
    
    assert len(descendants) == 2
    assert "strategy_001" in [d["id"] for d in descendants]
    assert "backtest_001" in [d["id"] for d in descendants]

# Test 4: Get siblings
def test_get_siblings():
    tracker = LineageTracker()
    
    # Setup: finding_001 -> strategy_001, strategy_002, strategy_003
    tracker.link_parent_child("finding_001", "research_finding", "strategy_001", "strategy")
    tracker.link_parent_child("finding_001", "research_finding", "strategy_002", "strategy")
    tracker.link_parent_child("finding_001", "research_finding", "strategy_003", "strategy")
    
    siblings = tracker.get_siblings("strategy_001")
    
    assert len(siblings) == 2
    assert "strategy_002" in [s["id"] for s in siblings]
    assert "strategy_003" in [s["id"] for s in siblings]

# Test 5: Prevent circular references
def test_prevent_circular_reference():
    tracker = LineageTracker()
    
    tracker.link_parent_child("finding_001", "research_finding", "strategy_001", "strategy")
    
    # Try to create circular reference
    with pytest.raises(ValueError, match="Circular reference detected"):
        tracker.link_parent_child("strategy_001", "strategy", "finding_001", "research_finding")

# Test 6: Get lineage tree
def test_get_lineage_tree():
    tracker = LineageTracker()
    
    # Setup complex tree
    tracker.link_parent_child("finding_001", "research_finding", "strategy_001", "strategy")
    tracker.link_parent_child("finding_001", "research_finding", "strategy_002", "strategy")
    tracker.link_parent_child("strategy_001", "strategy", "backtest_001", "backtest")
    
    tree = tracker.get_lineage_tree("finding_001")
    
    assert tree["id"] == "finding_001"
    assert len(tree["children"]) == 2
    assert tree["children"][0]["id"] == "strategy_001"
    assert len(tree["children"][0]["children"]) == 1
```

**Lineage Data Model**:
```python
{
    "parent_id": str,        # ID of parent entity
    "parent_type": str,      # Type of parent (research_finding, strategy, etc.)
    "child_id": str,         # ID of child entity
    "child_type": str,       # Type of child
    "relationship": str,     # "derived_from" | "generated_from" | "tested_by"
    "timestamp": str         # ISO 8601 timestamp
}
```

---

### 2.7 Archiving

**Objective**: Archive old data to reduce database size

**Tasks**:
- [ ] Create `ArchiveManager` class in `memory/archive_manager.py`
- [ ] Implement `archive_old_data()` method
- [ ] Implement `restore_archived_data()` method
- [ ] Implement `list_archives()` method
- [ ] Add compression for archived data
- [ ] Add automatic archiving scheduler

**Acceptance Criteria**:
- [ ] Can archive data older than N days
- [ ] Archived data is compressed
- [ ] Can restore archived data
- [ ] Can list all archives
- [ ] Archiving doesn't affect active data
- [ ] Automatic archiving runs on schedule

**Test Cases**:

```python
# Test 1: Archive old data
def test_archive_old_data():
    manager = ArchiveManager()
    
    # Archive data older than 30 days
    archived_count = manager.archive_old_data(
        collection="research_findings",
        days_old=30
    )
    
    assert archived_count > 0

# Test 2: Restore archived data
def test_restore_archived_data():
    manager = ArchiveManager()
    
    # Restore specific archive
    restored_count = manager.restore_archived_data(
        archive_id="archive_001"
    )
    
    assert restored_count > 0

# Test 3: List archives
def test_list_archives():
    manager = ArchiveManager()
    
    archives = manager.list_archives()
    
    assert len(archives) > 0
    assert "archive_id" in archives[0]
    assert "collection" in archives[0]
    assert "archived_date" in archives[0]

# Test 4: Compression
def test_compression():
    manager = ArchiveManager()
    
    # Archive and check compression
    archive_id = manager.archive_old_data("research_findings", days_old=30)
    archive_info = manager.get_archive_info(archive_id)
    
    assert archive_info["compressed"] is True
    assert archive_info["compression_ratio"] > 1.0
```

---

### 2.8 Session State Management

**Objective**: Persist workflow state across interruptions

**Tasks**:
- [ ] Create `SessionState` class in `memory/session_state.py`
- [ ] Implement LangGraph checkpointer integration
- [ ] Implement state serialization
- [ ] Implement state deserialization
- [ ] Implement state cleanup
- [ ] Add error handling for state operations

**Acceptance Criteria**:
- [ ] Workflow state persists across process restarts
- [ ] Can save workflow state at any point
- [ ] Can restore workflow state
- [ ] Can list all saved states
- [ ] Can delete old states
- [ ] State operations are thread-safe

**Test Cases**:

```python
# Test 1: Save state
def test_save_state():
    session = SessionState()
    
    state_id = session.save_state(
        workflow_id="workflow_001",
        state={
            "current_phase": "research",
            "findings": ["finding_001", "finding_002"],
            "progress": 0.5
        }
    )
    
    assert state_id is not None

# Test 2: Restore state
def test_restore_state():
    session = SessionState()
    
    state = session.restore_state(workflow_id="workflow_001")
    
    assert state is not None
    assert state["current_phase"] == "research"
    assert len(state["findings"]) == 2

# Test 3: Workflow continuation
def test_workflow_continuation():
    # Simulate workflow interruption
    session = SessionState()
    
    # Save state before interruption
    session.save_state("workflow_001", {"current_phase": "research", "progress": 0.5})
    
    # Simulate process restart
    # ...
    
    # Restore and continue
    state = session.restore_state("workflow_001")
    assert state["progress"] == 0.5
    
    # Continue workflow
    state["progress"] = 1.0
    session.save_state("workflow_001", state)
```

---

## Integration Tests

### Integration Test 1: End-to-End Memory Flow

**Objective**: Store finding → Create strategy → Link lineage → Retrieve

```python
def test_e2e_memory_flow():
    # 1. Store research finding
    findings_collection = ResearchFindingsCollection()
    finding_id = findings_collection.add_finding(
        text="AAPL shows strong momentum",
        metadata={"ticker": "AAPL", "type": "technical", "confidence": 0.85, "agent_id": "tech_001"}
    )
    
    # 2. Create strategy based on finding
    strategy_collection = StrategyLibraryCollection()
    strategy_id = strategy_collection.add_strategy(
        name="AAPL Momentum",
        description="Momentum strategy for AAPL",
        code="class AAPLMomentum(bt.Strategy): pass",
        metadata={
            "type": "momentum",
            "tickers": ["AAPL"],
            "timeframe": "1d",
            "parameters": {},
            "performance_metrics": {"sharpe_ratio": 1.5}
        }
    )
    
    # 3. Link lineage
    tracker = LineageTracker()
    tracker.link_parent_child(finding_id, "research_finding", strategy_id, "strategy")
    
    # 4. Retrieve and verify
    ancestors = tracker.get_ancestors(strategy_id)
    assert len(ancestors) == 1
    assert ancestors[0]["id"] == finding_id
    
    # 5. Search related findings
    related = findings_collection.get_related_findings(finding_id, n_results=5)
    assert len(related) >= 0
```

**Acceptance Criteria**:
- [ ] Test passes end-to-end
- [ ] All entities are stored correctly
- [ ] Lineage is tracked correctly
- [ ] Retrieval works as expected

---

### Integration Test 2: Cross-Collection Queries

**Objective**: Query across multiple collections

```python
def test_cross_collection_queries():
    # Find all strategies derived from technical findings about AAPL
    
    # 1. Get technical findings for AAPL
    findings = ResearchFindingsCollection()
    aapl_findings = findings.filter_findings(
        filters={"ticker": "AAPL", "type": "technical"}
    )
    
    # 2. Get strategies derived from these findings
    tracker = LineageTracker()
    strategies = []
    for finding in aapl_findings:
        descendants = tracker.get_descendants(finding["id"])
        strategy_ids = [d["id"] for d in descendants if d["type"] == "strategy"]
        strategies.extend(strategy_ids)
    
    # 3. Get strategy details
    strategy_collection = StrategyLibraryCollection()
    strategy_details = [strategy_collection.get_strategy(sid) for sid in strategies]
    
    # 4. Verify
    assert len(strategy_details) > 0
    for strategy in strategy_details:
        assert "AAPL" in strategy["metadata"]["tickers"]
```

**Acceptance Criteria**:
- [ ] Can query across collections
- [ ] Lineage tracking enables cross-collection queries
- [ ] Results are accurate

---

### Integration Test 3: Archive and Restore

**Objective**: Archive old data and restore it

```python
def test_archive_restore():
    # 1. Create old findings (simulate old data)
    findings = ResearchFindingsCollection()
    old_finding_id = findings.add_finding(
        text="Old finding",
        metadata={
            "ticker": "AAPL",
            "type": "technical",
            "confidence": 0.8,
            "agent_id": "tech_001",
            "timestamp": "2023-01-01T00:00:00Z"  # Old timestamp
        }
    )
    
    # 2. Archive old data
    archive_manager = ArchiveManager()
    archived_count = archive_manager.archive_old_data("research_findings", days_old=30)
    assert archived_count > 0
    
    # 3. Verify data is archived (not in active collection)
    with pytest.raises(ValueError):
        findings.get_finding(old_finding_id)
    
    # 4. Restore archived data
    archives = archive_manager.list_archives()
    archive_id = archives[0]["archive_id"]
    restored_count = archive_manager.restore_archived_data(archive_id)
    assert restored_count > 0
    
    # 5. Verify data is restored
    restored_finding = findings.get_finding(old_finding_id)
    assert restored_finding is not None
```

**Acceptance Criteria**:
- [ ] Old data is archived correctly
- [ ] Archived data is not in active collection
- [ ] Archived data can be restored
- [ ] Restored data is identical to original

---

## Performance Requirements

### Latency

- [ ] ChromaDB initialization < 1 second
- [ ] Add document < 100ms
- [ ] Query by ID < 50ms
- [ ] Similarity search (top 10) < 500ms
- [ ] Metadata filter query < 200ms
- [ ] Lineage query (3 levels deep) < 300ms

### Throughput

- [ ] Can handle 100 concurrent writes
- [ ] Can handle 1000 concurrent reads
- [ ] Can store 100,000+ documents per collection
- [ ] Can handle 10,000+ lineage relationships

### Storage

- [ ] Database size < 1GB for 10,000 documents
- [ ] Compression ratio > 2x for archived data
- [ ] Automatic cleanup of old data

---

## Security Requirements

- [ ] Database files have restricted permissions
- [ ] No sensitive data in logs
- [ ] Encrypted storage (optional, for production)
- [ ] Backup strategy documented

---

## Phase 2 Completion Checklist

### Must Pass (Critical)

- [ ] ChromaDB client initializes successfully
- [ ] All 4 collections created (research_findings, strategy_library, lessons_learned, market_regimes)
- [ ] Can store and retrieve research findings
- [ ] Can store and retrieve strategies
- [ ] Lineage tracker links parent-child relationships
- [ ] Can query ancestors and descendants
- [ ] Archive manager archives old data
- [ ] Session state persists across restarts
- [ ] Integration test 1 passes (end-to-end flow)
- [ ] All unit tests pass

### Should Pass (High Priority)

- [ ] Cross-collection queries work
- [ ] Archive and restore test passes
- [ ] Performance requirements met
- [ ] All documentation complete
- [ ] Code review completed

### Nice to Have (Medium Priority)

- [ ] Automatic backup system
- [ ] Database migration tools
- [ ] Performance monitoring dashboard
- [ ] Advanced query optimization

---

## Sign-Off Criteria

Phase 2 is considered complete when:

1. ✅ All "Must Pass" items are checked
2. ✅ All integration tests pass
3. ✅ Documentation is complete
4. ✅ Code review completed
5. ✅ No critical bugs or performance issues

**Sign-Off Date**: _____________  
**Approved By**: _____________

---

## Next Phase

Once Phase 2 is complete, proceed to **Phase 3: Tool Registry & Validation**.
