"""
Memory Manager for ChromaDB Integration

This module provides the MemoryManager class that manages all ChromaDB collections
and provides a unified interface for memory operations across the system.

Design Reference: docs/design/MEMORY_ARCHITECTURE.md
Phase: 2 - Memory System
"""

import chromadb
from chromadb.config import Settings
from typing import Optional, Dict, Any, List
from pathlib import Path
from loguru import logger
import json
from datetime import datetime


class MemoryManager:
    """
    Central memory manager that coordinates all ChromaDB collections.
    
    Manages 4 collections:
    - research_findings: Research findings from research swarm
    - strategy_library: Trading strategies with code and metadata
    - lessons_learned: Insights and failures from iterations
    - market_regimes: Market conditions and regime transitions
    """
    
    def __init__(
        self,
        persist_directory: str = "./data/chromadb",
        reset: bool = False
    ):
        """
        Initialize MemoryManager with ChromaDB client.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            reset: If True, delete existing data and start fresh
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client with persistent storage
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        if reset:
            logger.warning("Resetting ChromaDB - all data will be deleted")
            self.client.reset()
        
        # Initialize collections
        self._init_collections()
        
        logger.info(f"MemoryManager initialized with persist_directory={persist_directory}")
    
    def _init_collections(self):
        """Initialize all ChromaDB collections."""
        # Research Findings Collection
        self.research_findings = self.client.get_or_create_collection(
            name="research_findings",
            metadata={
                "description": "Research findings from research swarm agents",
                "created_at": datetime.now().isoformat()
            }
        )
        
        # Strategy Library Collection
        self.strategy_library = self.client.get_or_create_collection(
            name="strategy_library",
            metadata={
                "description": "Trading strategies with code and performance metrics",
                "created_at": datetime.now().isoformat()
            }
        )
        
        # Lessons Learned Collection
        self.lessons_learned = self.client.get_or_create_collection(
            name="lessons_learned",
            metadata={
                "description": "Insights, failures, and optimization decisions",
                "created_at": datetime.now().isoformat()
            }
        )
        
        # Market Regimes Collection
        self.market_regimes = self.client.get_or_create_collection(
            name="market_regimes",
            metadata={
                "description": "Market conditions and regime transitions",
                "created_at": datetime.now().isoformat()
            }
        )
        
        logger.info("All collections initialized successfully")
    
    def get_collection_stats(self) -> Dict[str, int]:
        """
        Get document counts for all collections.
        
        Returns:
            Dictionary with collection names and document counts
        """
        return {
            "research_findings": self.research_findings.count(),
            "strategy_library": self.strategy_library.count(),
            "lessons_learned": self.lessons_learned.count(),
            "market_regimes": self.market_regimes.count()
        }
    
    def add_research_finding(
        self,
        finding_id: str,
        content: str,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Add a research finding to the collection.
        
        Args:
            finding_id: Unique identifier for the finding
            content: Text content of the finding
            metadata: Metadata including ticker, type, confidence, agent_id, etc.
        """
        self.research_findings.add(
            ids=[finding_id],
            documents=[content],
            metadatas=[metadata]
        )
        logger.debug(f"Added research finding: {finding_id}")
    
    def search_research_findings(
        self,
        query: str,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Search research findings by semantic similarity.
        
        Args:
            query: Query text for semantic search
            n_results: Number of results to return
            where: Metadata filter (e.g., {"ticker": "AAPL"})
        
        Returns:
            Dictionary with ids, documents, metadatas, distances
        """
        results = self.research_findings.query(
            query_texts=[query],
            n_results=n_results,
            where=where
        )
        return results
    
    def add_strategy(
        self,
        strategy_id: str,
        description: str,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Add a trading strategy to the library.
        
        Args:
            strategy_id: Unique identifier for the strategy
            description: Text description of the strategy
            metadata: Metadata including name, type, code, parameters, performance, etc.
        """
        self.strategy_library.add(
            ids=[strategy_id],
            documents=[description],
            metadatas=[metadata]
        )
        logger.debug(f"Added strategy: {strategy_id}")
    
    def search_strategies(
        self,
        query: str,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Search strategies by semantic similarity.
        
        Args:
            query: Query text for semantic search
            n_results: Number of results to return
            where: Metadata filter (e.g., {"type": "momentum"})
        
        Returns:
            Dictionary with ids, documents, metadatas, distances
        """
        results = self.strategy_library.query(
            query_texts=[query],
            n_results=n_results,
            where=where
        )
        return results
    
    def add_lesson(
        self,
        lesson_id: str,
        content: str,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Add a lesson learned to the collection.
        
        Args:
            lesson_id: Unique identifier for the lesson
            content: Text content of the lesson
            metadata: Metadata including type, severity, context, etc.
        """
        self.lessons_learned.add(
            ids=[lesson_id],
            documents=[content],
            metadatas=[metadata]
        )
        logger.debug(f"Added lesson: {lesson_id}")
    
    def search_lessons(
        self,
        query: str,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Search lessons learned by semantic similarity.
        
        Args:
            query: Query text for semantic search
            n_results: Number of results to return
            where: Metadata filter (e.g., {"type": "failure"})
        
        Returns:
            Dictionary with ids, documents, metadatas, distances
        """
        results = self.lessons_learned.query(
            query_texts=[query],
            n_results=n_results,
            where=where
        )
        return results
    
    def add_market_regime(
        self,
        regime_id: str,
        description: str,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Add a market regime observation to the collection.
        
        Args:
            regime_id: Unique identifier for the regime
            description: Text description of the market regime
            metadata: Metadata including regime_type, volatility, indicators, etc.
        """
        self.market_regimes.add(
            ids=[regime_id],
            documents=[description],
            metadatas=[metadata]
        )
        logger.debug(f"Added market regime: {regime_id}")
    
    def search_market_regimes(
        self,
        query: str,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Search market regimes by semantic similarity.
        
        Args:
            query: Query text for semantic search
            n_results: Number of results to return
            where: Metadata filter (e.g., {"regime_type": "bull_high_vol"})
        
        Returns:
            Dictionary with ids, documents, metadatas, distances
        """
        results = self.market_regimes.query(
            query_texts=[query],
            n_results=n_results,
            where=where
        )
        return results
    
    def get_by_id(
        self,
        collection_name: str,
        doc_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get a document by ID from any collection.
        
        Args:
            collection_name: Name of the collection
            doc_id: Document ID
        
        Returns:
            Document data or None if not found
        """
        collection = getattr(self, collection_name, None)
        if collection is None:
            logger.error(f"Collection not found: {collection_name}")
            return None
        
        try:
            result = collection.get(ids=[doc_id])
            if result['ids']:
                return {
                    'id': result['ids'][0],
                    'document': result['documents'][0],
                    'metadata': result['metadatas'][0]
                }
            return None
        except Exception as e:
            logger.error(f"Error getting document {doc_id}: {e}")
            return None
    
    def update_metadata(
        self,
        collection_name: str,
        doc_id: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Update metadata for a document.
        
        Args:
            collection_name: Name of the collection
            doc_id: Document ID
            metadata: New metadata (will be merged with existing)
        
        Returns:
            True if successful, False otherwise
        """
        collection = getattr(self, collection_name, None)
        if collection is None:
            logger.error(f"Collection not found: {collection_name}")
            return False
        
        try:
            collection.update(
                ids=[doc_id],
                metadatas=[metadata]
            )
            logger.debug(f"Updated metadata for {doc_id} in {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error updating metadata: {e}")
            return False
    
    def delete_document(
        self,
        collection_name: str,
        doc_id: str
    ) -> bool:
        """
        Delete a document from a collection.
        
        Args:
            collection_name: Name of the collection
            doc_id: Document ID
        
        Returns:
            True if successful, False otherwise
        """
        collection = getattr(self, collection_name, None)
        if collection is None:
            logger.error(f"Collection not found: {collection_name}")
            return False
        
        try:
            collection.delete(ids=[doc_id])
            logger.debug(f"Deleted {doc_id} from {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False
    
    def clear_collection(self, collection_name: str) -> bool:
        """
        Clear all documents from a collection.
        
        Args:
            collection_name: Name of the collection
        
        Returns:
            True if successful, False otherwise
        """
        collection = getattr(self, collection_name, None)
        if collection is None:
            logger.error(f"Collection not found: {collection_name}")
            return False
        
        try:
            # Get all IDs and delete them
            all_docs = collection.get()
            if all_docs['ids']:
                collection.delete(ids=all_docs['ids'])
            logger.info(f"Cleared collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False
    
    def export_collection(
        self,
        collection_name: str,
        output_path: str
    ) -> bool:
        """
        Export a collection to JSON file.
        
        Args:
            collection_name: Name of the collection
            output_path: Path to output JSON file
        
        Returns:
            True if successful, False otherwise
        """
        collection = getattr(self, collection_name, None)
        if collection is None:
            logger.error(f"Collection not found: {collection_name}")
            return False
        
        try:
            all_docs = collection.get()
            export_data = {
                'collection_name': collection_name,
                'exported_at': datetime.now().isoformat(),
                'count': len(all_docs['ids']),
                'documents': [
                    {
                        'id': all_docs['ids'][i],
                        'document': all_docs['documents'][i],
                        'metadata': all_docs['metadatas'][i]
                    }
                    for i in range(len(all_docs['ids']))
                ]
            }
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Exported {collection_name} to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error exporting collection: {e}")
            return False


# Example usage
if __name__ == "__main__":
    # Initialize memory manager
    memory = MemoryManager(persist_directory="./data/chromadb_test", reset=True)
    
    # Add a research finding
    memory.add_research_finding(
        finding_id="finding_001",
        content="AAPL shows strong momentum with RSI at 65 and volume increasing",
        metadata={
            "ticker": "AAPL",
            "type": "technical",
            "confidence": 0.85,
            "agent_id": "price_action_agent",
            "timestamp": datetime.now().isoformat(),
            "source": "yahoo_finance",
            "timeframe": "1d"
        }
    )
    
    # Search research findings
    results = memory.search_research_findings(
        query="momentum indicators for AAPL",
        n_results=5,
        where={"ticker": "AAPL"}
    )
    print(f"Found {len(results['ids'][0])} research findings")
    
    # Get collection stats
    stats = memory.get_collection_stats()
    print(f"Collection stats: {stats}")
    
    print("MemoryManager test completed successfully!")
