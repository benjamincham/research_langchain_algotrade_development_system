import pytest
import os
import shutil
from pathlib import Path
from datetime import datetime
from src.memory.memory_manager import MemoryManager
from src.memory.collection_wrappers.research_findings import ResearchFindingsCollection, ResearchFindingMetadata
from src.memory.collection_wrappers.strategy_library import StrategyLibraryCollection, StrategyMetadata, PerformanceMetrics
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
