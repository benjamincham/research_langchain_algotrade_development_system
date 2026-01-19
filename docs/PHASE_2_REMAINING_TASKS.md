# Phase 2: Remaining Tasks Implementation Plan

**Status**: 33% Complete (4/12 tasks done)  
**Remaining**: 8 tasks (67%)  
**Estimated Effort**: 8 hours

---

## Completed Tasks ✅

- [x] Task 2.1: Install ChromaDB
- [x] Task 2.2: Implement MemoryManager
- [x] Task 2.3.1: Implement BaseCollection
- [x] Task 2.4: Implement LineageTracker

---

## Dependency Analysis

### Dependency Graph

```
BaseCollection (DONE)
    ↓
    ├─→ ResearchFindingsCollection (Task 2.3.2)
    ├─→ StrategyLibraryCollection (Task 2.3.3)
    ├─→ LessonsLearnedCollection (Task 2.3.4)
    └─→ MarketRegimesCollection (Task 2.3.5)
            ↓
            └─→ ArchiveManager (Task 2.5)
                    ↓ (uses all collections)
                    ├─→ Unit Tests (Task 2.6)
                    └─→ Integration Tests (Task 2.7)
```

### Key Dependencies

1. **All Collection Classes** depend on `BaseCollection` ✅ (already implemented)
2. **ArchiveManager** depends on all 4 collection classes (must be implemented first)
3. **Unit Tests** can be written in parallel with implementation
4. **Integration Tests** require all components to be complete

### Critical Path

```
BaseCollection → Collection Classes → ArchiveManager → Integration Tests
```

**Parallelization Opportunity**: All 4 collection classes can be implemented in parallel (no inter-dependencies)

---

## Remaining Tasks Breakdown

### Task 2.3.2: ResearchFindingsCollection

**Dependencies**: BaseCollection ✅  
**Effort**: 1 hour  
**Priority**: HIGH (required by ArchiveManager)

**Implementation**:
```python
# File: src/memory/collection_wrappers/research_findings.py

from pydantic import BaseModel, Field
from typing import Literal
from .base_collection import BaseCollection

class ResearchFindingMetadata(BaseModel):
    """Metadata schema for research findings."""
    ticker: str = Field(..., description="Stock ticker symbol")
    type: Literal["technical", "fundamental", "sentiment", "pattern"] = Field(
        ..., description="Type of research finding"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    agent_id: str = Field(..., description="ID of agent that generated finding")
    timestamp: str = Field(..., description="ISO timestamp")
    source: str = Field(..., description="Data source")
    timeframe: str = Field(..., description="Timeframe (e.g., '1D', '1W')")
    tags: list[str] = Field(default_factory=list, description="Tags for categorization")

class ResearchFindingsCollection(BaseCollection):
    """Collection for research findings."""
    
    def get_schema(self) -> type[BaseModel]:
        return ResearchFindingMetadata
    
    def get_by_ticker(self, ticker: str, limit: int = 10):
        """Get findings for a specific ticker."""
        return self.get_by_metadata({"ticker": ticker}, limit=limit)
    
    def get_by_type(self, finding_type: str, limit: int = 10):
        """Get findings by type."""
        return self.get_by_metadata({"type": finding_type}, limit=limit)
    
    def get_high_confidence(self, min_confidence: float = 0.8, limit: int = 10):
        """Get high-confidence findings."""
        return self.get_by_metadata({"confidence": {"$gte": min_confidence}}, limit=limit)
```

**Tests** (6 tests):
1. Add research finding with valid metadata
2. Add research finding with invalid metadata (should raise ValidationError)
3. Get findings by ticker
4. Get findings by type
5. Get high-confidence findings
6. Search findings by semantic similarity

**Acceptance Criteria**:
- ✅ Inherits from BaseCollection
- ✅ Implements get_schema() with ResearchFindingMetadata
- ✅ Implements 3 helper methods (get_by_ticker, get_by_type, get_high_confidence)
- ✅ All 6 tests pass

---

### Task 2.3.3: StrategyLibraryCollection

**Dependencies**: BaseCollection ✅  
**Effort**: 1.5 hours  
**Priority**: HIGH (required by ArchiveManager)

**Implementation**:
```python
# File: src/memory/collection_wrappers/strategy_library.py

from pydantic import BaseModel, Field
from typing import Literal, Dict, List
from .base_collection import BaseCollection

class PerformanceMetrics(BaseModel):
    """Performance metrics sub-schema."""
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    max_drawdown: float = Field(..., description="Maximum drawdown")
    total_return: float = Field(..., description="Total return")
    win_rate: float = Field(..., ge=0.0, le=1.0, description="Win rate")

class StrategyMetadata(BaseModel):
    """Metadata schema for strategies."""
    name: str = Field(..., description="Strategy name")
    type: Literal["momentum", "mean_reversion", "breakout", "arbitrage"] = Field(
        ..., description="Strategy type"
    )
    tickers: List[str] = Field(..., description="Tickers this strategy applies to")
    timeframe: str = Field(..., description="Timeframe")
    parameters: Dict[str, float] = Field(..., description="Strategy parameters")
    performance_metrics: PerformanceMetrics = Field(..., description="Performance metrics")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")

class StrategyLibraryCollection(BaseCollection):
    """Collection for trading strategies."""
    
    def get_schema(self) -> type[BaseModel]:
        return StrategyMetadata
    
    def get_by_type(self, strategy_type: str, limit: int = 10):
        """Get strategies by type."""
        return self.get_by_metadata({"type": strategy_type}, limit=limit)
    
    def get_top_performers(self, min_sharpe: float = 1.5, limit: int = 10):
        """Get top-performing strategies."""
        return self.get_by_metadata({"performance_metrics.sharpe_ratio": {"$gte": min_sharpe}}, limit=limit)
    
    def get_for_ticker(self, ticker: str, limit: int = 10):
        """Get strategies for a specific ticker."""
        # Note: ChromaDB doesn't support array contains, so we need to get all and filter
        all_strategies = self.get_all()
        return [s for s in all_strategies if ticker in s['metadata']['tickers']][:limit]
    
    def update_performance(self, strategy_id: str, metrics: PerformanceMetrics):
        """Update performance metrics for a strategy."""
        return self.update_metadata(strategy_id, {"performance_metrics": metrics.model_dump()})
```

**Tests** (6 tests):
1. Add strategy with valid metadata
2. Add strategy with invalid metadata (should raise ValidationError)
3. Get strategies by type
4. Get top performers by Sharpe ratio
5. Get strategies for specific ticker
6. Update performance metrics

**Acceptance Criteria**:
- ✅ Inherits from BaseCollection
- ✅ Implements get_schema() with StrategyMetadata
- ✅ Implements 4 helper methods
- ✅ All 6 tests pass

---

### Task 2.3.4: LessonsLearnedCollection

**Dependencies**: BaseCollection ✅  
**Effort**: 1 hour  
**Priority**: MEDIUM

**Implementation**:
```python
# File: src/memory/collection_wrappers/lessons_learned.py

from pydantic import BaseModel, Field
from typing import Literal
from .base_collection import BaseCollection

class LessonMetadata(BaseModel):
    """Metadata schema for lessons learned."""
    type: Literal["failure", "success", "optimization", "insight"] = Field(
        ..., description="Type of lesson"
    )
    severity: Literal["critical", "high", "medium", "low"] = Field(
        ..., description="Severity/importance level"
    )
    context: str = Field(..., description="Context where lesson was learned")
    timestamp: str = Field(..., description="ISO timestamp")
    tags: list[str] = Field(default_factory=list, description="Tags for categorization")

class LessonsLearnedCollection(BaseCollection):
    """Collection for lessons learned."""
    
    def get_schema(self) -> type[BaseModel]:
        return LessonMetadata
    
    def get_failures(self, limit: int = 10):
        """Get failure lessons."""
        return self.get_by_metadata({"type": "failure"}, limit=limit)
    
    def get_critical_lessons(self, limit: int = 10):
        """Get critical severity lessons."""
        return self.get_by_metadata({"severity": "critical"}, limit=limit)
    
    def get_by_context(self, context: str, limit: int = 10):
        """Get lessons by context."""
        return self.get_by_metadata({"context": context}, limit=limit)
```

**Tests** (5 tests):
1. Add lesson with valid metadata
2. Get failure lessons
3. Get critical lessons
4. Get lessons by context
5. Search lessons by semantic similarity

**Acceptance Criteria**:
- ✅ Inherits from BaseCollection
- ✅ Implements get_schema() with LessonMetadata
- ✅ Implements 3 helper methods
- ✅ All 5 tests pass

---

### Task 2.3.5: MarketRegimesCollection

**Dependencies**: BaseCollection ✅  
**Effort**: 1 hour  
**Priority**: MEDIUM

**Implementation**:
```python
# File: src/memory/collection_wrappers/market_regimes.py

from pydantic import BaseModel, Field
from typing import Literal, Dict
from .base_collection import BaseCollection

class RegimeMetadata(BaseModel):
    """Metadata schema for market regimes."""
    regime_type: Literal[
        "bull_high_vol", "bull_low_vol",
        "bear_high_vol", "bear_low_vol",
        "sideways_high_vol", "sideways_low_vol",
        "crisis"
    ] = Field(..., description="Market regime type")
    start_date: str = Field(..., description="Regime start date")
    end_date: str = Field(..., description="Regime end date (empty if current)")
    volatility: float = Field(..., ge=0.0, description="Average volatility")
    indicators: Dict[str, float] = Field(..., description="Technical indicators")

class MarketRegimesCollection(BaseCollection):
    """Collection for market regimes."""
    
    def get_schema(self) -> type[BaseModel]:
        return RegimeMetadata
    
    def get_current_regime(self):
        """Get the current market regime (end_date is empty)."""
        regimes = self.get_by_metadata({"end_date": ""}, limit=1)
        return regimes[0] if regimes else None
    
    def get_by_regime_type(self, regime_type: str, limit: int = 10):
        """Get regimes by type."""
        return self.get_by_metadata({"regime_type": regime_type}, limit=limit)
    
    def get_historical_regimes(self, start_date: str, end_date: str):
        """Get regimes within a date range."""
        # Note: ChromaDB metadata filtering is limited, may need custom logic
        all_regimes = self.get_all()
        return [
            r for r in all_regimes
            if r['metadata']['start_date'] >= start_date and r['metadata']['start_date'] <= end_date
        ]
```

**Tests** (5 tests):
1. Add regime with valid metadata
2. Get current regime
3. Get regimes by type
4. Get historical regimes by date range
5. Update regime end_date

**Acceptance Criteria**:
- ✅ Inherits from BaseCollection
- ✅ Implements get_schema() with RegimeMetadata
- ✅ Implements 3 helper methods
- ✅ All 5 tests pass

---

### Task 2.5: ArchiveManager

**Dependencies**: All 4 collection classes ✅  
**Effort**: 2 hours  
**Priority**: HIGH

**Implementation**:
```python
# File: src/memory/archive_manager.py

from typing import Optional, Dict, Any
from pathlib import Path
import json
import gzip
from datetime import datetime
from loguru import logger
from chromadb import Collection

class ArchiveManager:
    """
    Manage archiving and restoration of old data.
    
    Archives data to compressed JSON files to reduce memory usage
    while maintaining the ability to restore when needed.
    """
    
    def __init__(self, archive_dir: str = "./data/archives"):
        """Initialize archive manager."""
        self.archive_dir = Path(archive_dir)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized ArchiveManager with archive_dir={archive_dir}")
    
    def archive_collection(
        self,
        collection: Collection,
        older_than_days: int = 90,
        compress: bool = True
    ) -> Dict[str, Any]:
        """
        Archive old documents from a collection.
        
        Args:
            collection: ChromaDB collection
            older_than_days: Archive documents older than this many days
            compress: Whether to compress the archive file
        
        Returns:
            Dictionary with archive stats (archived_count, archive_path)
        """
        collection_name = collection.name
        cutoff_date = datetime.now().timestamp() - (older_than_days * 24 * 60 * 60)
        
        # Get all documents
        all_docs = collection.get()
        
        # Filter old documents
        old_docs = []
        old_ids = []
        
        for i, doc_id in enumerate(all_docs['ids']):
            metadata = all_docs['metadatas'][i]
            timestamp = metadata.get('timestamp', '')
            
            # Parse timestamp and check if old
            if timestamp:
                try:
                    doc_timestamp = datetime.fromisoformat(timestamp).timestamp()
                    if doc_timestamp < cutoff_date:
                        old_docs.append({
                            'id': doc_id,
                            'document': all_docs['documents'][i],
                            'metadata': metadata
                        })
                        old_ids.append(doc_id)
                except Exception as e:
                    logger.warning(f"Error parsing timestamp for {doc_id}: {e}")
        
        if not old_docs:
            logger.info(f"No documents to archive in {collection_name}")
            return {"archived_count": 0, "archive_path": None}
        
        # Create archive file
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{collection_name}_{timestamp_str}.json"
        if compress:
            filename += ".gz"
        
        archive_path = self.archive_dir / filename
        
        # Write archive
        archive_data = {
            "collection": collection_name,
            "archived_at": datetime.now().isoformat(),
            "older_than_days": older_than_days,
            "document_count": len(old_docs),
            "documents": old_docs
        }
        
        if compress:
            with gzip.open(archive_path, 'wt', encoding='utf-8') as f:
                json.dump(archive_data, f, indent=2)
        else:
            with open(archive_path, 'w') as f:
                json.dump(archive_data, f, indent=2)
        
        # Delete archived documents from collection
        collection.delete(ids=old_ids)
        
        logger.info(f"Archived {len(old_docs)} documents from {collection_name} to {archive_path}")
        
        return {
            "archived_count": len(old_docs),
            "archive_path": str(archive_path)
        }
    
    def restore_archive(
        self,
        archive_path: str,
        collection: Collection
    ) -> int:
        """
        Restore documents from an archive file.
        
        Args:
            archive_path: Path to archive file
            collection: ChromaDB collection to restore to
        
        Returns:
            Number of documents restored
        """
        archive_path = Path(archive_path)
        
        if not archive_path.exists():
            raise FileNotFoundError(f"Archive file not found: {archive_path}")
        
        # Read archive
        if archive_path.suffix == '.gz':
            with gzip.open(archive_path, 'rt', encoding='utf-8') as f:
                archive_data = json.load(f)
        else:
            with open(archive_path, 'r') as f:
                archive_data = json.load(f)
        
        documents = archive_data['documents']
        
        # Restore to collection
        ids = [doc['id'] for doc in documents]
        docs = [doc['document'] for doc in documents]
        metadatas = [doc['metadata'] for doc in documents]
        
        collection.add(ids=ids, documents=docs, metadatas=metadatas)
        
        logger.info(f"Restored {len(documents)} documents from {archive_path} to {collection.name}")
        
        return len(documents)
    
    def list_archives(self, collection_name: Optional[str] = None) -> list[Dict[str, Any]]:
        """
        List all archive files.
        
        Args:
            collection_name: Filter by collection name (optional)
        
        Returns:
            List of archive info dictionaries
        """
        archives = []
        
        for archive_file in self.archive_dir.glob("*.json*"):
            # Parse filename
            parts = archive_file.stem.replace('.json', '').split('_')
            if len(parts) >= 3:
                col_name = '_'.join(parts[:-2])
                
                if collection_name and col_name != collection_name:
                    continue
                
                archives.append({
                    "collection": col_name,
                    "path": str(archive_file),
                    "size_bytes": archive_file.stat().st_size,
                    "created_at": datetime.fromtimestamp(archive_file.stat().st_ctime).isoformat()
                })
        
        return sorted(archives, key=lambda x: x['created_at'], reverse=True)
    
    def delete_archive(self, archive_path: str) -> bool:
        """
        Delete an archive file.
        
        Args:
            archive_path: Path to archive file
        
        Returns:
            True if successful, False otherwise
        """
        try:
            Path(archive_path).unlink()
            logger.info(f"Deleted archive: {archive_path}")
            return True
        except Exception as e:
            logger.error(f"Error deleting archive {archive_path}: {e}")
            return False
```

**Tests** (5 tests):
1. Archive old documents from collection
2. Restore documents from archive
3. List archives (all and filtered by collection)
4. Delete archive file
5. Archive with compression enabled

**Acceptance Criteria**:
- ✅ Can archive documents older than N days
- ✅ Can restore documents from archive
- ✅ Supports compression (gzip)
- ✅ Can list and delete archives
- ✅ All 5 tests pass

---

### Task 2.6: Unit Tests

**Dependencies**: All components implemented  
**Effort**: 1.5 hours  
**Priority**: HIGH

**Test Files**:
1. `tests/unit/test_research_findings_collection.py` (6 tests)
2. `tests/unit/test_strategy_library_collection.py` (6 tests)
3. `tests/unit/test_lessons_learned_collection.py` (5 tests)
4. `tests/unit/test_market_regimes_collection.py` (5 tests)
5. `tests/unit/test_archive_manager.py` (5 tests)

**Total**: 27 new unit tests

**Acceptance Criteria**:
- ✅ All 27 unit tests pass
- ✅ Test coverage > 80% for new code
- ✅ Tests use pytest fixtures
- ✅ Tests are isolated (no shared state)

---

### Task 2.7: Integration Tests

**Dependencies**: All components + unit tests  
**Effort**: 1 hour  
**Priority**: HIGH

**Test File**: `tests/integration/test_phase2_memory_system.py`

**Integration Tests** (3 tests):

1. **Test Complete Research Flow**:
   - Add research findings to ResearchFindingsCollection
   - Track lineage with LineageTracker
   - Search findings by ticker
   - Verify lineage relationships

2. **Test Strategy Evolution Flow**:
   - Add strategy v1 to StrategyLibraryCollection
   - Add strategy v2 (refined from v1)
   - Track lineage (v1 → v2)
   - Update performance metrics
   - Verify lineage shows evolution

3. **Test Archive and Restore Flow**:
   - Add old research findings (90+ days old)
   - Archive old findings with ArchiveManager
   - Verify findings removed from collection
   - Restore from archive
   - Verify findings restored correctly

**Acceptance Criteria**:
- ✅ All 3 integration tests pass
- ✅ Tests cover end-to-end workflows
- ✅ Tests verify cross-component integration
- ✅ Tests use realistic data

---

## Execution Plan

### Recommended Order

**Day 1** (4 hours):
1. Task 2.3.2: ResearchFindingsCollection (1 hour)
2. Task 2.3.3: StrategyLibraryCollection (1.5 hours)
3. Task 2.3.4: LessonsLearnedCollection (1 hour)
4. Task 2.3.5: MarketRegimesCollection (0.5 hours)

**Day 2** (4 hours):
5. Task 2.5: ArchiveManager (2 hours)
6. Task 2.6: Unit Tests (1.5 hours)
7. Task 2.7: Integration Tests (0.5 hours)

**Total**: 8 hours (1 day for experienced developer, 2 days for learning)

### Parallelization Strategy

**Can be done in parallel**:
- All 4 collection classes (Tasks 2.3.2-2.3.5)
- Unit tests can be written alongside implementation

**Must be sequential**:
- ArchiveManager must wait for all collections
- Integration tests must wait for all components

---

## Testing Strategy

### Unit Test Coverage

| Component | Tests | Coverage Target |
|-----------|-------|-----------------|
| ResearchFindingsCollection | 6 | 90% |
| StrategyLibraryCollection | 6 | 90% |
| LessonsLearnedCollection | 5 | 90% |
| MarketRegimesCollection | 5 | 90% |
| ArchiveManager | 5 | 85% |
| **Total** | **27** | **88%** |

### Integration Test Coverage

| Flow | Components Tested | Complexity |
|------|-------------------|------------|
| Research Flow | ResearchFindings + LineageTracker | Medium |
| Strategy Evolution | StrategyLibrary + LineageTracker | Medium |
| Archive/Restore | All Collections + ArchiveManager | High |

---

## Acceptance Criteria (Phase 2 Complete)

### Must Pass (All Required)

- [ ] All 4 collection classes implemented and tested
- [ ] ArchiveManager implemented and tested
- [ ] 27 unit tests passing
- [ ] 3 integration tests passing
- [ ] Code coverage > 85%
- [ ] All classes properly documented
- [ ] No critical bugs or errors

### Should Pass (Recommended)

- [ ] Performance benchmarks met (< 500ms for most operations)
- [ ] Memory usage reasonable (< 100MB for 10K documents)
- [ ] Archive compression working (> 50% size reduction)

---

## Risk Assessment

### High Risks

1. **ChromaDB metadata filtering limitations**
   - **Impact**: Some helper methods may not work as expected
   - **Mitigation**: Use get_all() and filter in Python if needed

2. **Archive/restore data integrity**
   - **Impact**: Data loss if archive fails
   - **Mitigation**: Test thoroughly, add checksums

### Medium Risks

1. **Test isolation issues**
   - **Impact**: Flaky tests
   - **Mitigation**: Use pytest fixtures, clean up after each test

---

## Success Metrics

**Phase 2 Complete When**:
- ✅ 100% of tasks complete (12/12)
- ✅ 100% of tests passing (43/43)
- ✅ Code coverage > 85%
- ✅ Documentation complete
- ✅ All acceptance criteria met

**Current Status**: 33% complete (4/12 tasks)  
**Remaining Effort**: 8 hours  
**Target Completion**: 1-2 days
