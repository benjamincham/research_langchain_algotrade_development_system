import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Any, Optional
from src.config.settings import settings
from src.core.logging import logger

class MemoryManager:
    """Manages vector store collections for the system."""
    
    def __init__(self, persist_directory: Optional[str] = None):
        self.persist_directory = persist_directory or settings.CHROMA_DB_PATH
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collections = {}
        self._init_collections()
        logger.info(f"Initialized MemoryManager with persistence at: {self.persist_directory}")

    def _init_collections(self):
        """Initialize standard collections."""
        collection_names = [
            "research_findings",
            "strategy_library",
            "lessons_learned",
            "market_regimes"
        ]
        for name in collection_names:
            self.collections[name] = self.client.get_or_create_collection(name=name)
            logger.debug(f"Initialized collection: {name}")

    def add_finding(self, collection_name: str, content: str, metadata: Dict[str, Any], id: str):
        """Add a finding to a specific collection."""
        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} does not exist.")
        
        self.collections[collection_name].add(
            documents=[content],
            metadatas=[metadata],
            ids=[id]
        )
        logger.debug(f"Added finding to {collection_name}: {id}")

    def query_findings(self, collection_name: str, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        """Query findings from a specific collection."""
        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} does not exist.")
        
        return self.collections[collection_name].query(
            query_texts=[query_text],
            n_results=n_results
        )

# Singleton instance
_memory_manager = None

def get_memory_manager():
    """Get or initialize the global MemoryManager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager
