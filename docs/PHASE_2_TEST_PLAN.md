# Phase 2: Memory System - Test Plan

**Total Tests**: 43 (16 existing + 27 new)  
**Current Coverage**: 11.3% (Phase 1 only)  
**Target Coverage**: 85%+

---

## Test Structure

```
tests/
├── unit/
│   ├── test_memory_manager.py ✅ (existing, 6 tests)
│   ├── test_base_collection.py ✅ (existing, 5 tests)
│   ├── test_lineage_tracker.py ✅ (existing, 5 tests)
│   ├── test_research_findings_collection.py ⏳ (6 tests)
│   ├── test_strategy_library_collection.py ⏳ (6 tests)
│   ├── test_lessons_learned_collection.py ⏳ (5 tests)
│   ├── test_market_regimes_collection.py ⏳ (5 tests)
│   └── test_archive_manager.py ⏳ (5 tests)
└── integration/
    └── test_phase2_memory_system.py ⏳ (3 tests)
```

---

## Unit Tests Breakdown

### 1. ResearchFindingsCollection (6 tests)

**File**: `tests/unit/test_research_findings_collection.py`

```python
import pytest
from datetime import datetime
from pydantic import ValidationError
from src.memory.collection_wrappers.research_findings import (
    ResearchFindingsCollection,
    ResearchFindingMetadata
)

class TestResearchFindingsCollection:
    """Test suite for ResearchFindingsCollection."""
    
    @pytest.fixture
    def collection(self, memory_manager):
        """Fixture for research findings collection."""
        return ResearchFindingsCollection(
            memory_manager.client,
            "test_research_findings"
        )
    
    def test_add_valid_finding(self, collection):
        """Test adding a research finding with valid metadata."""
        metadata = ResearchFindingMetadata(
            ticker="AAPL",
            type="technical",
            confidence=0.85,
            agent_id="agent_001",
            timestamp=datetime.now().isoformat(),
            source="yahoo_finance",
            timeframe="1D",
            tags=["momentum", "bullish"]
        )
        
        result = collection.add(
            id="finding_001",
            document="AAPL shows strong bullish momentum with RSI at 65",
            metadata=metadata.model_dump()
        )
        
        assert result["id"] == "finding_001"
    
    def test_add_invalid_finding(self, collection):
        """Test adding a finding with invalid metadata raises ValidationError."""
        with pytest.raises(ValidationError):
            collection.add(
                id="finding_002",
                document="Invalid finding",
                metadata={
                    "ticker": "AAPL",
                    "type": "invalid_type",  # Invalid type
                    "confidence": 1.5,  # Out of range
                }
            )
    
    def test_get_by_ticker(self, collection):
        """Test retrieving findings by ticker."""
        # Add multiple findings
        for i in range(3):
            metadata = ResearchFindingMetadata(
                ticker="AAPL" if i < 2 else "GOOGL",
                type="technical",
                confidence=0.8,
                agent_id=f"agent_{i}",
                timestamp=datetime.now().isoformat(),
                source="yahoo_finance",
                timeframe="1D",
                tags=[]
            )
            collection.add(
                id=f"finding_{i}",
                document=f"Finding {i}",
                metadata=metadata.model_dump()
            )
        
        # Get AAPL findings
        results = collection.get_by_ticker("AAPL")
        assert len(results) == 2
        assert all(r['metadata']['ticker'] == "AAPL" for r in results)
    
    def test_get_by_type(self, collection):
        """Test retrieving findings by type."""
        # Add findings of different types
        types = ["technical", "fundamental", "sentiment"]
        for i, finding_type in enumerate(types):
            metadata = ResearchFindingMetadata(
                ticker="AAPL",
                type=finding_type,
                confidence=0.8,
                agent_id=f"agent_{i}",
                timestamp=datetime.now().isoformat(),
                source="yahoo_finance",
                timeframe="1D",
                tags=[]
            )
            collection.add(
                id=f"finding_{i}",
                document=f"Finding {i}",
                metadata=metadata.model_dump()
            )
        
        # Get technical findings
        results = collection.get_by_type("technical")
        assert len(results) == 1
        assert results[0]['metadata']['type'] == "technical"
    
    def test_get_high_confidence(self, collection):
        """Test retrieving high-confidence findings."""
        # Add findings with different confidence levels
        confidences = [0.6, 0.85, 0.95]
        for i, conf in enumerate(confidences):
            metadata = ResearchFindingMetadata(
                ticker="AAPL",
                type="technical",
                confidence=conf,
                agent_id=f"agent_{i}",
                timestamp=datetime.now().isoformat(),
                source="yahoo_finance",
                timeframe="1D",
                tags=[]
            )
            collection.add(
                id=f"finding_{i}",
                document=f"Finding {i}",
                metadata=metadata.model_dump()
            )
        
        # Get high-confidence findings (>= 0.8)
        results = collection.get_high_confidence(min_confidence=0.8)
        assert len(results) == 2
        assert all(r['metadata']['confidence'] >= 0.8 for r in results)
    
    def test_semantic_search(self, collection):
        """Test semantic search for findings."""
        # Add findings with different content
        findings = [
            "AAPL shows strong bullish momentum",
            "AAPL technical indicators suggest uptrend",
            "GOOGL earnings beat expectations"
        ]
        
        for i, doc in enumerate(findings):
            metadata = ResearchFindingMetadata(
                ticker="AAPL" if i < 2 else "GOOGL",
                type="technical",
                confidence=0.8,
                agent_id=f"agent_{i}",
                timestamp=datetime.now().isoformat(),
                source="yahoo_finance",
                timeframe="1D",
                tags=[]
            )
            collection.add(
                id=f"finding_{i}",
                document=doc,
                metadata=metadata.model_dump()
            )
        
        # Search for momentum-related findings
        results = collection.search("momentum bullish", n_results=2)
        assert len(results) > 0
        assert "momentum" in results[0]['document'].lower() or "bullish" in results[0]['document'].lower()
```

**Coverage Target**: 90%

---

### 2. StrategyLibraryCollection (6 tests)

**File**: `tests/unit/test_strategy_library_collection.py`

```python
import pytest
from datetime import datetime
from pydantic import ValidationError
from src.memory.collection_wrappers.strategy_library import (
    StrategyLibraryCollection,
    StrategyMetadata,
    PerformanceMetrics
)

class TestStrategyLibraryCollection:
    """Test suite for StrategyLibraryCollection."""
    
    @pytest.fixture
    def collection(self, memory_manager):
        """Fixture for strategy library collection."""
        return StrategyLibraryCollection(
            memory_manager.client,
            "test_strategy_library"
        )
    
    def test_add_valid_strategy(self, collection):
        """Test adding a strategy with valid metadata."""
        metrics = PerformanceMetrics(
            sharpe_ratio=1.8,
            max_drawdown=-0.15,
            total_return=0.25,
            win_rate=0.65
        )
        
        metadata = StrategyMetadata(
            name="Momentum Strategy v1",
            type="momentum",
            tickers=["AAPL", "GOOGL"],
            timeframe="1D",
            parameters={"lookback": 20, "threshold": 0.02},
            performance_metrics=metrics,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        
        result = collection.add(
            id="strategy_001",
            document="Momentum strategy using RSI and MACD",
            metadata=metadata.model_dump()
        )
        
        assert result["id"] == "strategy_001"
    
    def test_add_invalid_strategy(self, collection):
        """Test adding a strategy with invalid metadata raises ValidationError."""
        with pytest.raises(ValidationError):
            collection.add(
                id="strategy_002",
                document="Invalid strategy",
                metadata={
                    "name": "Test",
                    "type": "invalid_type",  # Invalid type
                    "tickers": ["AAPL"],
                    "timeframe": "1D"
                    # Missing required fields
                }
            )
    
    def test_get_by_type(self, collection):
        """Test retrieving strategies by type."""
        # Add strategies of different types
        types = ["momentum", "mean_reversion", "breakout"]
        for i, strategy_type in enumerate(types):
            metrics = PerformanceMetrics(
                sharpe_ratio=1.5,
                max_drawdown=-0.1,
                total_return=0.2,
                win_rate=0.6
            )
            metadata = StrategyMetadata(
                name=f"Strategy {i}",
                type=strategy_type,
                tickers=["AAPL"],
                timeframe="1D",
                parameters={},
                performance_metrics=metrics,
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat()
            )
            collection.add(
                id=f"strategy_{i}",
                document=f"Strategy {i}",
                metadata=metadata.model_dump()
            )
        
        # Get momentum strategies
        results = collection.get_by_type("momentum")
        assert len(results) == 1
        assert results[0]['metadata']['type'] == "momentum"
    
    def test_get_top_performers(self, collection):
        """Test retrieving top-performing strategies."""
        # Add strategies with different Sharpe ratios
        sharpe_ratios = [1.2, 1.8, 2.5]
        for i, sharpe in enumerate(sharpe_ratios):
            metrics = PerformanceMetrics(
                sharpe_ratio=sharpe,
                max_drawdown=-0.1,
                total_return=0.2,
                win_rate=0.6
            )
            metadata = StrategyMetadata(
                name=f"Strategy {i}",
                type="momentum",
                tickers=["AAPL"],
                timeframe="1D",
                parameters={},
                performance_metrics=metrics,
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat()
            )
            collection.add(
                id=f"strategy_{i}",
                document=f"Strategy {i}",
                metadata=metadata.model_dump()
            )
        
        # Get top performers (Sharpe >= 1.5)
        results = collection.get_top_performers(min_sharpe=1.5)
        assert len(results) == 2
        assert all(r['metadata']['performance_metrics']['sharpe_ratio'] >= 1.5 for r in results)
    
    def test_get_for_ticker(self, collection):
        """Test retrieving strategies for a specific ticker."""
        # Add strategies for different tickers
        ticker_lists = [["AAPL", "GOOGL"], ["MSFT"], ["AAPL", "TSLA"]]
        for i, tickers in enumerate(ticker_lists):
            metrics = PerformanceMetrics(
                sharpe_ratio=1.5,
                max_drawdown=-0.1,
                total_return=0.2,
                win_rate=0.6
            )
            metadata = StrategyMetadata(
                name=f"Strategy {i}",
                type="momentum",
                tickers=tickers,
                timeframe="1D",
                parameters={},
                performance_metrics=metrics,
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat()
            )
            collection.add(
                id=f"strategy_{i}",
                document=f"Strategy {i}",
                metadata=metadata.model_dump()
            )
        
        # Get strategies for AAPL
        results = collection.get_for_ticker("AAPL")
        assert len(results) == 2
        assert all("AAPL" in r['metadata']['tickers'] for r in results)
    
    def test_update_performance(self, collection):
        """Test updating performance metrics for a strategy."""
        # Add initial strategy
        metrics = PerformanceMetrics(
            sharpe_ratio=1.5,
            max_drawdown=-0.1,
            total_return=0.2,
            win_rate=0.6
        )
        metadata = StrategyMetadata(
            name="Strategy 1",
            type="momentum",
            tickers=["AAPL"],
            timeframe="1D",
            parameters={},
            performance_metrics=metrics,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        collection.add(
            id="strategy_001",
            document="Strategy 1",
            metadata=metadata.model_dump()
        )
        
        # Update performance
        new_metrics = PerformanceMetrics(
            sharpe_ratio=2.0,
            max_drawdown=-0.08,
            total_return=0.3,
            win_rate=0.7
        )
        collection.update_performance("strategy_001", new_metrics)
        
        # Verify update
        result = collection.get("strategy_001")
        assert result['metadata']['performance_metrics']['sharpe_ratio'] == 2.0
```

**Coverage Target**: 90%

---

### 3. LessonsLearnedCollection (5 tests)

**File**: `tests/unit/test_lessons_learned_collection.py`

```python
import pytest
from datetime import datetime
from src.memory.collection_wrappers.lessons_learned import (
    LessonsLearnedCollection,
    LessonMetadata
)

class TestLessonsLearnedCollection:
    """Test suite for LessonsLearnedCollection."""
    
    @pytest.fixture
    def collection(self, memory_manager):
        """Fixture for lessons learned collection."""
        return LessonsLearnedCollection(
            memory_manager.client,
            "test_lessons_learned"
        )
    
    def test_add_valid_lesson(self, collection):
        """Test adding a lesson with valid metadata."""
        metadata = LessonMetadata(
            type="failure",
            severity="critical",
            context="backtesting",
            timestamp=datetime.now().isoformat(),
            tags=["overfitting", "data_leakage"]
        )
        
        result = collection.add(
            id="lesson_001",
            document="Strategy overfit to training data due to look-ahead bias",
            metadata=metadata.model_dump()
        )
        
        assert result["id"] == "lesson_001"
    
    def test_get_failures(self, collection):
        """Test retrieving failure lessons."""
        # Add lessons of different types
        types = ["failure", "success", "optimization"]
        for i, lesson_type in enumerate(types):
            metadata = LessonMetadata(
                type=lesson_type,
                severity="medium",
                context="backtesting",
                timestamp=datetime.now().isoformat(),
                tags=[]
            )
            collection.add(
                id=f"lesson_{i}",
                document=f"Lesson {i}",
                metadata=metadata.model_dump()
            )
        
        # Get failure lessons
        results = collection.get_failures()
        assert len(results) == 1
        assert results[0]['metadata']['type'] == "failure"
    
    def test_get_critical_lessons(self, collection):
        """Test retrieving critical severity lessons."""
        # Add lessons with different severities
        severities = ["low", "medium", "critical", "critical"]
        for i, severity in enumerate(severities):
            metadata = LessonMetadata(
                type="failure",
                severity=severity,
                context="backtesting",
                timestamp=datetime.now().isoformat(),
                tags=[]
            )
            collection.add(
                id=f"lesson_{i}",
                document=f"Lesson {i}",
                metadata=metadata.model_dump()
            )
        
        # Get critical lessons
        results = collection.get_critical_lessons()
        assert len(results) == 2
        assert all(r['metadata']['severity'] == "critical" for r in results)
    
    def test_get_by_context(self, collection):
        """Test retrieving lessons by context."""
        # Add lessons with different contexts
        contexts = ["backtesting", "live_trading", "backtesting"]
        for i, context in enumerate(contexts):
            metadata = LessonMetadata(
                type="failure",
                severity="medium",
                context=context,
                timestamp=datetime.now().isoformat(),
                tags=[]
            )
            collection.add(
                id=f"lesson_{i}",
                document=f"Lesson {i}",
                metadata=metadata.model_dump()
            )
        
        # Get backtesting lessons
        results = collection.get_by_context("backtesting")
        assert len(results) == 2
        assert all(r['metadata']['context'] == "backtesting" for r in results)
    
    def test_semantic_search(self, collection):
        """Test semantic search for lessons."""
        # Add lessons with different content
        lessons = [
            "Overfitting due to look-ahead bias",
            "Data leakage in feature engineering",
            "Slippage not accounted for in backtest"
        ]
        
        for i, doc in enumerate(lessons):
            metadata = LessonMetadata(
                type="failure",
                severity="critical",
                context="backtesting",
                timestamp=datetime.now().isoformat(),
                tags=[]
            )
            collection.add(
                id=f"lesson_{i}",
                document=doc,
                metadata=metadata.model_dump()
            )
        
        # Search for data-related lessons
        results = collection.search("data leakage overfitting", n_results=2)
        assert len(results) > 0
```

**Coverage Target**: 90%

---

### 4. MarketRegimesCollection (5 tests)

**File**: `tests/unit/test_market_regimes_collection.py`

```python
import pytest
from datetime import datetime
from src.memory.collection_wrappers.market_regimes import (
    MarketRegimesCollection,
    RegimeMetadata
)

class TestMarketRegimesCollection:
    """Test suite for MarketRegimesCollection."""
    
    @pytest.fixture
    def collection(self, memory_manager):
        """Fixture for market regimes collection."""
        return MarketRegimesCollection(
            memory_manager.client,
            "test_market_regimes"
        )
    
    def test_add_valid_regime(self, collection):
        """Test adding a regime with valid metadata."""
        metadata = RegimeMetadata(
            regime_type="bull_low_vol",
            start_date="2024-01-01",
            end_date="",
            volatility=0.15,
            indicators={"vix": 12.5, "spy_return": 0.08}
        )
        
        result = collection.add(
            id="regime_001",
            document="Bull market with low volatility, VIX below 15",
            metadata=metadata.model_dump()
        )
        
        assert result["id"] == "regime_001"
    
    def test_get_current_regime(self, collection):
        """Test retrieving the current regime."""
        # Add historical regime
        metadata1 = RegimeMetadata(
            regime_type="bull_low_vol",
            start_date="2023-01-01",
            end_date="2023-12-31",
            volatility=0.15,
            indicators={}
        )
        collection.add(
            id="regime_001",
            document="Historical regime",
            metadata=metadata1.model_dump()
        )
        
        # Add current regime
        metadata2 = RegimeMetadata(
            regime_type="bear_high_vol",
            start_date="2024-01-01",
            end_date="",  # Current regime
            volatility=0.35,
            indicators={}
        )
        collection.add(
            id="regime_002",
            document="Current regime",
            metadata=metadata2.model_dump()
        )
        
        # Get current regime
        current = collection.get_current_regime()
        assert current is not None
        assert current['metadata']['end_date'] == ""
        assert current['metadata']['regime_type'] == "bear_high_vol"
    
    def test_get_by_regime_type(self, collection):
        """Test retrieving regimes by type."""
        # Add regimes of different types
        types = ["bull_low_vol", "bear_high_vol", "bull_low_vol"]
        for i, regime_type in enumerate(types):
            metadata = RegimeMetadata(
                regime_type=regime_type,
                start_date=f"2024-0{i+1}-01",
                end_date=f"2024-0{i+1}-31",
                volatility=0.2,
                indicators={}
            )
            collection.add(
                id=f"regime_{i}",
                document=f"Regime {i}",
                metadata=metadata.model_dump()
            )
        
        # Get bull_low_vol regimes
        results = collection.get_by_regime_type("bull_low_vol")
        assert len(results) == 2
        assert all(r['metadata']['regime_type'] == "bull_low_vol" for r in results)
    
    def test_get_historical_regimes(self, collection):
        """Test retrieving regimes within a date range."""
        # Add regimes with different date ranges
        dates = [
            ("2023-01-01", "2023-06-30"),
            ("2023-07-01", "2023-12-31"),
            ("2024-01-01", "2024-06-30")
        ]
        
        for i, (start, end) in enumerate(dates):
            metadata = RegimeMetadata(
                regime_type="bull_low_vol",
                start_date=start,
                end_date=end,
                volatility=0.2,
                indicators={}
            )
            collection.add(
                id=f"regime_{i}",
                document=f"Regime {i}",
                metadata=metadata.model_dump()
            )
        
        # Get regimes in 2023
        results = collection.get_historical_regimes("2023-01-01", "2023-12-31")
        assert len(results) == 2
    
    def test_update_regime_end_date(self, collection):
        """Test updating a regime's end date."""
        # Add current regime
        metadata = RegimeMetadata(
            regime_type="bull_low_vol",
            start_date="2024-01-01",
            end_date="",
            volatility=0.15,
            indicators={}
        )
        collection.add(
            id="regime_001",
            document="Current regime",
            metadata=metadata.model_dump()
        )
        
        # Update end date
        collection.update_metadata("regime_001", {"end_date": "2024-06-30"})
        
        # Verify update
        result = collection.get("regime_001")
        assert result['metadata']['end_date'] == "2024-06-30"
```

**Coverage Target**: 90%

---

### 5. ArchiveManager (5 tests)

**File**: `tests/unit/test_archive_manager.py`

```python
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from src.memory.archive_manager import ArchiveManager
from src.memory.collection_wrappers.research_findings import (
    ResearchFindingsCollection,
    ResearchFindingMetadata
)

class TestArchiveManager:
    """Test suite for ArchiveManager."""
    
    @pytest.fixture
    def archive_manager(self, tmp_path):
        """Fixture for archive manager."""
        return ArchiveManager(archive_dir=str(tmp_path / "archives"))
    
    @pytest.fixture
    def collection_with_old_data(self, memory_manager):
        """Fixture for collection with old data."""
        collection = ResearchFindingsCollection(
            memory_manager.client,
            "test_archive_collection"
        )
        
        # Add old findings (100 days old)
        old_timestamp = (datetime.now() - timedelta(days=100)).isoformat()
        for i in range(3):
            metadata = ResearchFindingMetadata(
                ticker="AAPL",
                type="technical",
                confidence=0.8,
                agent_id=f"agent_{i}",
                timestamp=old_timestamp,
                source="yahoo_finance",
                timeframe="1D",
                tags=[]
            )
            collection.add(
                id=f"old_finding_{i}",
                document=f"Old finding {i}",
                metadata=metadata.model_dump()
            )
        
        # Add recent findings (10 days old)
        recent_timestamp = (datetime.now() - timedelta(days=10)).isoformat()
        for i in range(2):
            metadata = ResearchFindingMetadata(
                ticker="GOOGL",
                type="technical",
                confidence=0.8,
                agent_id=f"agent_{i}",
                timestamp=recent_timestamp,
                source="yahoo_finance",
                timeframe="1D",
                tags=[]
            )
            collection.add(
                id=f"recent_finding_{i}",
                document=f"Recent finding {i}",
                metadata=metadata.model_dump()
            )
        
        return collection
    
    def test_archive_old_documents(self, archive_manager, collection_with_old_data):
        """Test archiving old documents from a collection."""
        result = archive_manager.archive_collection(
            collection_with_old_data.collection,
            older_than_days=90,
            compress=False
        )
        
        assert result['archived_count'] == 3
        assert result['archive_path'] is not None
        assert Path(result['archive_path']).exists()
        
        # Verify old documents removed from collection
        remaining = collection_with_old_data.get_all()
        assert len(remaining) == 2  # Only recent findings remain
    
    def test_archive_with_compression(self, archive_manager, collection_with_old_data):
        """Test archiving with compression enabled."""
        result = archive_manager.archive_collection(
            collection_with_old_data.collection,
            older_than_days=90,
            compress=True
        )
        
        assert result['archived_count'] == 3
        assert result['archive_path'].endswith('.gz')
        assert Path(result['archive_path']).exists()
    
    def test_restore_from_archive(self, archive_manager, collection_with_old_data, memory_manager):
        """Test restoring documents from an archive."""
        # Archive old documents
        archive_result = archive_manager.archive_collection(
            collection_with_old_data.collection,
            older_than_days=90,
            compress=False
        )
        
        # Verify documents removed
        assert len(collection_with_old_data.get_all()) == 2
        
        # Restore from archive
        restored_count = archive_manager.restore_archive(
            archive_result['archive_path'],
            collection_with_old_data.collection
        )
        
        assert restored_count == 3
        
        # Verify documents restored
        all_docs = collection_with_old_data.get_all()
        assert len(all_docs) == 5  # 2 recent + 3 restored
    
    def test_list_archives(self, archive_manager, collection_with_old_data):
        """Test listing archives."""
        # Create multiple archives
        archive_manager.archive_collection(
            collection_with_old_data.collection,
            older_than_days=90,
            compress=False
        )
        
        # List all archives
        archives = archive_manager.list_archives()
        assert len(archives) >= 1
        assert 'collection' in archives[0]
        assert 'path' in archives[0]
        assert 'size_bytes' in archives[0]
        
        # List archives for specific collection
        filtered = archive_manager.list_archives(
            collection_name="test_archive_collection"
        )
        assert len(filtered) >= 1
    
    def test_delete_archive(self, archive_manager, collection_with_old_data):
        """Test deleting an archive file."""
        # Create archive
        result = archive_manager.archive_collection(
            collection_with_old_data.collection,
            older_than_days=90,
            compress=False
        )
        
        archive_path = result['archive_path']
        assert Path(archive_path).exists()
        
        # Delete archive
        success = archive_manager.delete_archive(archive_path)
        assert success is True
        assert not Path(archive_path).exists()
```

**Coverage Target**: 85%

---

## Integration Tests

### File: `tests/integration/test_phase2_memory_system.py`

```python
import pytest
from datetime import datetime, timedelta
from src.memory.memory_manager import MemoryManager
from src.memory.lineage_tracker import LineageTracker
from src.memory.archive_manager import ArchiveManager
from src.memory.collection_wrappers.research_findings import ResearchFindingMetadata
from src.memory.collection_wrappers.strategy_library import StrategyMetadata, PerformanceMetrics

class TestPhase2Integration:
    """Integration tests for Phase 2 Memory System."""
    
    @pytest.fixture
    def memory_system(self, tmp_path):
        """Fixture for complete memory system."""
        memory_manager = MemoryManager(persist_directory=str(tmp_path / "chroma"))
        lineage_tracker = LineageTracker(storage_path=str(tmp_path / "lineage.json"))
        archive_manager = ArchiveManager(archive_dir=str(tmp_path / "archives"))
        
        return {
            'memory': memory_manager,
            'lineage': lineage_tracker,
            'archive': archive_manager
        }
    
    def test_complete_research_flow(self, memory_system):
        """
        Test complete research flow:
        1. Add research findings
        2. Track lineage
        3. Search findings
        4. Verify lineage relationships
        """
        memory = memory_system['memory']
        lineage = memory_system['lineage']
        
        # Add research findings
        findings_collection = memory.get_research_findings_collection()
        
        # Add parent finding
        parent_metadata = ResearchFindingMetadata(
            ticker="AAPL",
            type="technical",
            confidence=0.75,
            agent_id="agent_001",
            timestamp=datetime.now().isoformat(),
            source="yahoo_finance",
            timeframe="1D",
            tags=["momentum"]
        )
        findings_collection.add(
            id="finding_parent",
            document="AAPL shows initial momentum signal",
            metadata=parent_metadata.model_dump()
        )
        
        # Add child finding (refined from parent)
        child_metadata = ResearchFindingMetadata(
            ticker="AAPL",
            type="technical",
            confidence=0.90,
            agent_id="agent_002",
            timestamp=datetime.now().isoformat(),
            source="yahoo_finance",
            timeframe="1D",
            tags=["momentum", "confirmed"]
        )
        findings_collection.add(
            id="finding_child",
            document="AAPL momentum confirmed with multiple indicators",
            metadata=child_metadata.model_dump()
        )
        
        # Track lineage
        lineage.add_node("finding_parent", "research_finding", parent_metadata.model_dump())
        lineage.add_node("finding_child", "research_finding", child_metadata.model_dump())
        lineage.add_edge("finding_parent", "finding_child", "refined_by")
        
        # Search findings by ticker
        aapl_findings = findings_collection.get_by_ticker("AAPL")
        assert len(aapl_findings) == 2
        
        # Verify lineage
        descendants = lineage.get_descendants("finding_parent")
        assert "finding_child" in descendants
        
        ancestors = lineage.get_ancestors("finding_child")
        assert "finding_parent" in ancestors
    
    def test_strategy_evolution_flow(self, memory_system):
        """
        Test strategy evolution flow:
        1. Add strategy v1
        2. Add strategy v2 (refined from v1)
        3. Track lineage
        4. Update performance metrics
        5. Verify lineage shows evolution
        """
        memory = memory_system['memory']
        lineage = memory_system['lineage']
        
        # Add strategy v1
        strategies_collection = memory.get_strategy_library_collection()
        
        metrics_v1 = PerformanceMetrics(
            sharpe_ratio=1.5,
            max_drawdown=-0.15,
            total_return=0.20,
            win_rate=0.60
        )
        metadata_v1 = StrategyMetadata(
            name="Momentum Strategy v1",
            type="momentum",
            tickers=["AAPL"],
            timeframe="1D",
            parameters={"lookback": 20, "threshold": 0.02},
            performance_metrics=metrics_v1,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        strategies_collection.add(
            id="strategy_v1",
            document="Basic momentum strategy using RSI",
            metadata=metadata_v1.model_dump()
        )
        
        # Add strategy v2 (refined)
        metrics_v2 = PerformanceMetrics(
            sharpe_ratio=2.0,
            max_drawdown=-0.10,
            total_return=0.30,
            win_rate=0.70
        )
        metadata_v2 = StrategyMetadata(
            name="Momentum Strategy v2",
            type="momentum",
            tickers=["AAPL"],
            timeframe="1D",
            parameters={"lookback": 20, "threshold": 0.02, "stop_loss": 0.05},
            performance_metrics=metrics_v2,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        strategies_collection.add(
            id="strategy_v2",
            document="Enhanced momentum strategy with stop loss",
            metadata=metadata_v2.model_dump()
        )
        
        # Track lineage
        lineage.add_node("strategy_v1", "strategy", metadata_v1.model_dump())
        lineage.add_node("strategy_v2", "strategy", metadata_v2.model_dump())
        lineage.add_edge("strategy_v1", "strategy_v2", "evolved_to")
        
        # Verify evolution path
        evolution_path = lineage.get_descendants("strategy_v1")
        assert "strategy_v2" in evolution_path
        
        # Verify performance improvement
        v1 = strategies_collection.get("strategy_v1")
        v2 = strategies_collection.get("strategy_v2")
        assert v2['metadata']['performance_metrics']['sharpe_ratio'] > v1['metadata']['performance_metrics']['sharpe_ratio']
    
    def test_archive_and_restore_flow(self, memory_system):
        """
        Test archive and restore flow:
        1. Add old research findings (90+ days old)
        2. Archive old findings
        3. Verify findings removed from collection
        4. Restore from archive
        5. Verify findings restored correctly
        """
        memory = memory_system['memory']
        archive = memory_system['archive']
        
        # Add old findings
        findings_collection = memory.get_research_findings_collection()
        
        old_timestamp = (datetime.now() - timedelta(days=100)).isoformat()
        for i in range(5):
            metadata = ResearchFindingMetadata(
                ticker="AAPL",
                type="technical",
                confidence=0.8,
                agent_id=f"agent_{i}",
                timestamp=old_timestamp,
                source="yahoo_finance",
                timeframe="1D",
                tags=[]
            )
            findings_collection.add(
                id=f"old_finding_{i}",
                document=f"Old finding {i}",
                metadata=metadata.model_dump()
            )
        
        # Add recent findings
        recent_timestamp = (datetime.now() - timedelta(days=10)).isoformat()
        for i in range(3):
            metadata = ResearchFindingMetadata(
                ticker="GOOGL",
                type="technical",
                confidence=0.8,
                agent_id=f"agent_{i}",
                timestamp=recent_timestamp,
                source="yahoo_finance",
                timeframe="1D",
                tags=[]
            )
            findings_collection.add(
                id=f"recent_finding_{i}",
                document=f"Recent finding {i}",
                metadata=metadata.model_dump()
            )
        
        # Verify total count
        assert len(findings_collection.get_all()) == 8
        
        # Archive old findings
        archive_result = archive.archive_collection(
            findings_collection.collection,
            older_than_days=90,
            compress=True
        )
        
        assert archive_result['archived_count'] == 5
        
        # Verify old findings removed
        remaining = findings_collection.get_all()
        assert len(remaining) == 3
        assert all("recent" in doc['id'] for doc in remaining)
        
        # Restore from archive
        restored_count = archive.restore_archive(
            archive_result['archive_path'],
            findings_collection.collection
        )
        
        assert restored_count == 5
        
        # Verify all findings restored
        all_findings = findings_collection.get_all()
        assert len(all_findings) == 8
```

**Coverage Target**: End-to-end workflows

---

## Test Execution Plan

### Step 1: Unit Tests (Sequential)

```bash
# Test each collection class as implemented
pytest tests/unit/test_research_findings_collection.py -v
pytest tests/unit/test_strategy_library_collection.py -v
pytest tests/unit/test_lessons_learned_collection.py -v
pytest tests/unit/test_market_regimes_collection.py -v
pytest tests/unit/test_archive_manager.py -v
```

### Step 2: Integration Tests

```bash
# Run after all unit tests pass
pytest tests/integration/test_phase2_memory_system.py -v
```

### Step 3: Full Phase 2 Test Suite

```bash
# Run all Phase 2 tests
pytest tests/unit/test_memory_manager.py tests/unit/test_base_collection.py tests/unit/test_lineage_tracker.py tests/unit/test_research_findings_collection.py tests/unit/test_strategy_library_collection.py tests/unit/test_lessons_learned_collection.py tests/unit/test_market_regimes_collection.py tests/unit/test_archive_manager.py tests/integration/test_phase2_memory_system.py -v --cov=src/memory --cov-report=term-missing
```

---

## Success Criteria

### Phase 2 Complete When:

- ✅ All 43 tests passing (16 existing + 27 new)
- ✅ Code coverage > 85% for memory module
- ✅ No critical bugs or errors
- ✅ All acceptance criteria met
- ✅ Documentation complete

### Performance Benchmarks:

- Similarity search: < 500ms
- Archive operation: < 2s for 1000 documents
- Restore operation: < 1s for 1000 documents
- Memory usage: < 100MB for 10K documents

---

## Next Steps After Phase 2

1. **Phase 3**: Tool Registry & Validation (3 days)
2. **Phase 4**: Research Swarm Implementation (5 days)
3. **Phase 5**: Strategy Development Agent (4 days)
