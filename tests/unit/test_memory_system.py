import pytest
import os
import shutil
from pathlib import Path
from datetime import datetime
from src.memory.memory_manager import MemoryManager
from src.memory.collection_wrappers.research_findings import ResearchFindingsCollection, ResearchFindingMetadata
from src.memory.collection_wrappers.strategy_library import StrategyLibraryCollection, StrategyMetadata, PerformanceMetrics
from src.memory.collection_wrappers.lessons_learned import LessonsLearnedCollection
from src.memory.collection_wrappers.market_regimes import MarketRegimesCollection
from src.memory.archive_manager import ArchiveManager
from src.memory.lineage_tracker import LineageTracker
from pydantic import ValidationError

@pytest.fixture
def temp_db_dir():
    db_dir = Path("./data/test_chromadb")
    if db_dir.exists():
        shutil.rmtree(db_dir)
    yield str(db_dir)
    if db_dir.exists():
        shutil.rmtree(db_dir)

@pytest.fixture
def memory_manager():
    import chromadb
    from chromadb.config import Settings
    from src.memory.memory_manager import MemoryManager
    
    # Create a MemoryManager with an in-memory client for testing
    manager = MemoryManager(persist_directory=":memory:", reset=False)
    # Override the client with an ephemeral one for true in-memory testing if needed,
    # but PersistentClient with ":memory:" or a temp dir should work if permissions are right.
    # Let's try a different approach: use a real temp directory but ensure it's fresh.
    return manager

def test_memory_manager_initialization(memory_manager):
    """Test that MemoryManager initializes correctly and creates collections."""
    assert memory_manager.client is not None
    
    stats = memory_manager.get_collection_stats()
    assert "research_findings" in stats
    assert "strategy_library" in stats
    assert "lessons_learned" in stats
    assert "market_regimes" in stats

def test_research_findings_collection(memory_manager):
    """Test ResearchFindingsCollection operations."""
    collection = ResearchFindingsCollection(memory_manager.research_findings)
    
    finding_id = "test_finding_001"
    content = "AAPL shows strong momentum"
    metadata = {
        "ticker": "AAPL",
        "type": "technical",
        "confidence": 0.85,
        "agent_id": "test_agent",
        "timestamp": datetime.now().isoformat(),
        "source": "test_source",
        "timeframe": "1D",
        "tags": ["momentum"]
    }
    
    # Test add
    assert collection.add(finding_id, content, metadata) is True
    
    # Test get
    result = collection.get(finding_id)
    assert result is not None
    assert result["document"] == content
    assert result["metadata"]["ticker"] == "AAPL"
    
    # Test get_by_ticker
    ticker_results = collection.get_by_ticker("AAPL")
    assert len(ticker_results) > 0
    assert ticker_results[0]["id"] == finding_id
    
    # Test validation error
    invalid_metadata = metadata.copy()
    invalid_metadata["confidence"] = 1.5  # Invalid: must be <= 1.0
    with pytest.raises(ValidationError):
        collection.add("invalid_id", content, invalid_metadata)

def test_strategy_library_collection(memory_manager):
    """Test StrategyLibraryCollection operations."""
    collection = StrategyLibraryCollection(memory_manager.strategy_library)
    
    strategy_id = "test_strategy_001"
    description = "Momentum strategy for AAPL"
    metrics = PerformanceMetrics(
        sharpe_ratio=1.8,
        max_drawdown=-0.15,
        total_return=0.25,
        win_rate=0.65
    )
    metadata = {
        "name": "AAPL Momentum",
        "type": "momentum",
        "tickers": ["AAPL"],
        "timeframe": "1D",
        "parameters": {"lookback": 20},
        "performance_metrics": metrics.model_dump(),
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }
    
    # Test add
    assert collection.add(strategy_id, description, metadata) is True
    
    # Test get_top_performers
    top_performers = collection.get_top_performers(min_sharpe=1.5)
    assert len(top_performers) > 0
    assert top_performers[0]["id"] == strategy_id
    
    # Test update_performance
    new_metrics = PerformanceMetrics(
        sharpe_ratio=2.2,
        max_drawdown=-0.10,
        total_return=0.35,
        win_rate=0.70
    )
    collection.update_performance(strategy_id, new_metrics)
    
    updated = collection.get(strategy_id)
    assert updated["metadata"]["performance_metrics"]["sharpe_ratio"] == 2.2

def test_lessons_learned_collection(memory_manager):
    """Test LessonsLearnedCollection operations."""
    collection = LessonsLearnedCollection(memory_manager.lessons_learned)
    
    lesson_id = "test_lesson_001"
    content = "Over-reliance on RSI led to false signals in low volatility"
    metadata = {
        "type": "failure",
        "severity": "medium",
        "context": "momentum_strategy_v1",
        "timestamp": datetime.now().isoformat(),
        "tags": ["rsi", "volatility"]
    }
    
    # Test add
    assert collection.add(lesson_id, content, metadata) is True
    
    # Test get_failures
    failures = collection.get_failures()
    assert len(failures) > 0
    assert failures[0]["id"] == lesson_id

def test_market_regimes_collection(memory_manager):
    """Test MarketRegimesCollection operations."""
    collection = MarketRegimesCollection(memory_manager.market_regimes)
    
    regime_id = "test_regime_001"
    description = "Bull market with low volatility"
    metadata = {
        "regime_type": "bull_low_vol",
        "start_date": "2024-01-01",
        "end_date": "",
        "volatility": 0.12,
        "indicators": {"vix": 15.5, "sma_200": 4500.0}
    }
    
    # Test add
    assert collection.add(regime_id, description, metadata) is True
    
    # Test get_current_regime
    current = collection.get_current_regime()
    assert current is not None
    assert current["id"] == regime_id
    assert current["metadata"]["regime_type"] == "bull_low_vol"

def test_archive_manager(memory_manager, tmp_path):
    """Test ArchiveManager operations."""
    archive_dir = tmp_path / "archives"
    archive_manager = ArchiveManager(archive_dir=str(archive_dir))
    
    collection = ResearchFindingsCollection(memory_manager.research_findings)
    collection.clear()
    
    # Add an old document (100 days ago)
    from datetime import timedelta
    old_date = (datetime.now() - timedelta(days=100)).isoformat()
    
    collection.add(
        "old_doc", 
        "Old finding", 
        {
            "ticker": "AAPL", "type": "technical", "confidence": 0.5, 
            "agent_id": "agent", "timestamp": old_date, "source": "src", 
            "timeframe": "1D", "tags": []
        }
    )
    
    # Add a new document
    new_date = datetime.now().isoformat()
    collection.add(
        "new_doc", 
        "New finding", 
        {
            "ticker": "AAPL", "type": "technical", "confidence": 0.9, 
            "agent_id": "agent", "timestamp": new_date, "source": "src", 
            "timeframe": "1D", "tags": []
        }
    )
    
    assert collection.count() == 2
    
    # Archive documents older than 90 days
    result = archive_manager.archive_collection(
        collection=memory_manager.research_findings,
        older_than_days=90
    )
    
    assert result["archived_count"] == 1
    assert collection.count() == 1
    assert Path(result["archive_path"]).exists()
    
    # Restore from archive
    restored_count = archive_manager.restore_archive(
        archive_path=result["archive_path"],
        collection=memory_manager.research_findings
    )
    
    assert restored_count == 1
    assert collection.count() == 2
    assert collection.exists("old_doc")

def test_lineage_tracker(tmp_path):
    """Test LineageTracker operations."""
    persist_path = tmp_path / "lineage_graph.json"
    tracker = LineageTracker(persist_path=str(persist_path))
    
    # Add nodes
    tracker.add_node("finding_1", "research_finding", "research_findings")
    tracker.add_node("strategy_1", "strategy", "strategy_library")
    
    # Add edge
    tracker.add_edge("finding_1", "strategy_1", "derived_from")
    
    # Verify lineage
    lineage = tracker.get_lineage("strategy_1")
    assert lineage["node_id"] == "strategy_1"
    assert len(lineage["ancestors"]) == 1
    assert lineage["ancestors"][0]["node_id"] == "finding_1"
    
    # Test circular dependency prevention
    with pytest.raises(ValueError, match="would create a cycle"):
        tracker.add_edge("strategy_1", "finding_1", "refined_from")
