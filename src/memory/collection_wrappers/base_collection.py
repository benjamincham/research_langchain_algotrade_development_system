"""
Base Collection Class for ChromaDB Wrappers

This module provides the abstract base class for all collection wrappers.
All concrete collection classes inherit from this base class.

Design Reference: docs/design/MEMORY_ARCHITECTURE.md
Phase: 2 - Memory System
Task: 2.3.1 - BaseCollection
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import json
from pydantic import BaseModel, ValidationError
from chromadb import Collection
from loguru import logger


class BaseCollection(ABC):
    """
    Abstract base class for all collection wrappers.
    
    Provides common functionality:
    - Add, get, update, delete operations
    - Search with metadata filtering
    - Batch operations
    - Pydantic validation for all metadata
    - Error handling and logging
    
    All concrete collection classes must implement get_schema().
    """
    
    def __init__(self, collection: Collection):
        """
        Initialize collection wrapper.
        
        Args:
            collection: ChromaDB collection instance
        """
        self.collection = collection
        self.collection_name = collection.name
        logger.info(f"Initialized {self.__class__.__name__} for collection '{self.collection_name}'")
    
    @abstractmethod
    def get_schema(self) -> type[BaseModel]:
        """
        Get the Pydantic schema for this collection.
        
        Must be implemented by all concrete collection classes.
        
        Returns:
            Pydantic model class for metadata validation
        """
        pass
    
    def add(
        self,
        doc_id: str,
        content: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Add a document to the collection with metadata validation.
        
        Args:
            doc_id: Unique document identifier
            content: Text content for semantic search
            metadata: Metadata dictionary (will be validated against schema)
        
        Returns:
            True if successful, False otherwise
        
        Raises:
            ValidationError: If metadata doesn't match schema
        """
        try:
            # Validate metadata against schema
            schema = self.get_schema()
            validated = schema(**metadata)
            
            # Add to ChromaDB
            # ChromaDB only supports str, int, float, bool for metadata values
            # We need to serialize lists and dicts to JSON strings
            metadata_dump = validated.model_dump()
            processed_metadata = {}
            for k, v in metadata_dump.items():
                if isinstance(v, (list, dict)):
                    processed_metadata[k] = json.dumps(v)
                else:
                    processed_metadata[k] = v

            self.collection.add(
                ids=[doc_id],
                documents=[content],
                metadatas=[processed_metadata]
            )
            
            logger.debug(f"Added document '{doc_id}' to {self.collection_name}")
            return True
            
        except ValidationError as e:
            logger.error(f"Metadata validation failed for '{doc_id}': {e}")
            raise
        except Exception as e:
            logger.error(f"Error adding document '{doc_id}': {e}")
            return False
    
    def get(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a document by ID.
        
        Args:
            doc_id: Document identifier
        
        Returns:
            Dictionary with id, document, metadata or None if not found
        """
        try:
            result = self.collection.get(ids=[doc_id])
            
            if result['ids']:
                metadata = result['metadatas'][0]
                # Deserialize JSON strings back to lists/dicts
                processed_metadata = {}
                for k, v in metadata.items():
                    if isinstance(v, str):
                        try:
                            # Try to parse as JSON if it looks like a list or dict
                            if (v.startswith('[') and v.endswith(']')) or (v.startswith('{') and v.endswith('}')):
                                processed_metadata[k] = json.loads(v)
                            else:
                                processed_metadata[k] = v
                        except (json.JSONDecodeError, TypeError):
                            processed_metadata[k] = v
                    else:
                        processed_metadata[k] = v

                return {
                    'id': result['ids'][0],
                    'document': result['documents'][0],
                    'metadata': processed_metadata
                }
            
            logger.debug(f"Document '{doc_id}' not found in {self.collection_name}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting document '{doc_id}': {e}")
            return None
    
    def search(
        self,
        query: str,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search documents by semantic similarity.
        
        Args:
            query: Query text for semantic search
            n_results: Maximum number of results to return
            where: Metadata filter (e.g., {"ticker": "AAPL"})
        
        Returns:
            List of dictionaries with id, document, metadata, distance
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where
            )
            
            # Convert to list of dicts
            documents = []
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    metadata = results['metadatas'][0][i]
                    processed_metadata = {}
                    for k, v in metadata.items():
                        if isinstance(v, str):
                            try:
                                if (v.startswith('[') and v.endswith(']')) or (v.startswith('{') and v.endswith('}')):
                                    processed_metadata[k] = json.loads(v)
                                else:
                                    processed_metadata[k] = v
                            except (json.JSONDecodeError, TypeError):
                                processed_metadata[k] = v
                        else:
                            processed_metadata[k] = v

                    documents.append({
                        'id': results['ids'][0][i],
                        'document': results['documents'][0][i],
                        'metadata': processed_metadata,
                        'distance': results['distances'][0][i]
                    })
            
            logger.debug(f"Search in {self.collection_name} returned {len(documents)} results")
            return documents
            
        except Exception as e:
            logger.error(f"Error searching {self.collection_name}: {e}")
            return []
    
    def update(
        self,
        doc_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update a document's content and/or metadata.
        
        Args:
            doc_id: Document identifier
            content: New content (optional)
            metadata: New metadata (optional, will be validated)
        
        Returns:
            True if successful, False otherwise
        
        Raises:
            ValidationError: If metadata doesn't match schema
        """
        try:
            update_kwargs = {'ids': [doc_id]}
            
            if content is not None:
                update_kwargs['documents'] = [content]
            
            if metadata is not None:
                # Get current metadata to support partial updates
                current = self.get(doc_id)
                if current:
                    # Merge current metadata with new metadata
                    # Need to deserialize current metadata first
                    current_meta = current['metadata']
                    for k, v in current_meta.items():
                        if isinstance(v, str):
                            try:
                                current_meta[k] = json.loads(v)
                            except (json.JSONDecodeError, TypeError):
                                pass
                    
                    merged_metadata = {**current_meta, **metadata}
                else:
                    merged_metadata = metadata

                # Validate merged metadata
                schema = self.get_schema()
                validated = schema(**merged_metadata).model_dump()
                processed = {}
                for k, v in validated.items():
                    if isinstance(v, (list, dict)):
                        processed[k] = json.dumps(v)
                    else:
                        processed[k] = v
                update_kwargs['metadatas'] = [processed]
            
            self.collection.update(**update_kwargs)
            logger.debug(f"Updated document '{doc_id}' in {self.collection_name}")
            return True
            
        except ValidationError as e:
            logger.error(f"Metadata validation failed for '{doc_id}': {e}")
            raise
        except Exception as e:
            logger.error(f"Error updating document '{doc_id}': {e}")
            return False
    
    def update_metadata(
        self,
        doc_id: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Update only metadata for a document.
        
        Args:
            doc_id: Document identifier
            metadata: New metadata (will be validated)
        
        Returns:
            True if successful, False otherwise
        """
        return self.update(doc_id, metadata=metadata)
    
    def delete(self, doc_id: str) -> bool:
        """
        Delete a document from the collection.
        
        Args:
            doc_id: Document identifier
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.collection.delete(ids=[doc_id])
            logger.debug(f"Deleted document '{doc_id}' from {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting document '{doc_id}': {e}")
            return False
    
    def count(self) -> int:
        """
        Get the total number of documents in the collection.
        
        Returns:
            Document count
        """
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Error getting count for {self.collection_name}: {e}")
            return 0
    
    def batch_add(
        self,
        documents: List[Dict[str, Any]]
    ) -> bool:
        """
        Add multiple documents at once.
        
        Args:
            documents: List of dicts with 'id', 'content', 'metadata' keys
        
        Returns:
            True if successful, False otherwise
        
        Raises:
            ValidationError: If any metadata doesn't match schema
        """
        try:
            ids = [doc['id'] for doc in documents]
            contents = [doc['content'] for doc in documents]
            metadatas = [doc['metadata'] for doc in documents]
            
            # Validate all metadata
            schema = self.get_schema()
            processed_metadatas = []
            for meta in metadatas:
                validated = schema(**meta).model_dump()
                processed = {}
                for k, v in validated.items():
                    if isinstance(v, (list, dict)):
                        processed[k] = json.dumps(v)
                    else:
                        processed[k] = v
                processed_metadatas.append(processed)
            
            # Add to ChromaDB
            self.collection.add(
                ids=ids,
                documents=contents,
                metadatas=processed_metadatas
            )
            
            logger.info(f"Batch added {len(documents)} documents to {self.collection_name}")
            return True
            
        except ValidationError as e:
            logger.error(f"Metadata validation failed in batch add: {e}")
            raise
        except Exception as e:
            logger.error(f"Error in batch add to {self.collection_name}: {e}")
            return False
    
    def get_all(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get all documents in the collection.
        
        Args:
            limit: Maximum number of documents to return (None = all)
        
        Returns:
            List of dictionaries with id, document, metadata
        """
        try:
            result = self.collection.get(limit=limit)
            
            documents = []
            if result['ids']:
                for i in range(len(result['ids'])):
                    metadata = result['metadatas'][i]
                    processed_metadata = {}
                    for k, v in metadata.items():
                        if isinstance(v, str):
                            try:
                                if (v.startswith('[') and v.endswith(']')) or (v.startswith('{') and v.endswith('}')):
                                    processed_metadata[k] = json.loads(v)
                                else:
                                    processed_metadata[k] = v
                            except (json.JSONDecodeError, TypeError):
                                processed_metadata[k] = v
                        else:
                            processed_metadata[k] = v

                    documents.append({
                        'id': result['ids'][i],
                        'document': result['documents'][i],
                        'metadata': processed_metadata
                    })
            
            logger.debug(f"Retrieved {len(documents)} documents from {self.collection_name}")
            return documents
            
        except Exception as e:
            logger.error(f"Error getting all documents from {self.collection_name}: {e}")
            return []
    
    def get_by_metadata(
        self,
        where: Dict[str, Any],
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get documents by metadata filter.
        
        Args:
            where: Metadata filter (e.g., {"ticker": "AAPL"})
            limit: Maximum number of documents to return
        
        Returns:
            List of dictionaries with id, document, metadata
        """
        try:
            result = self.collection.get(
                where=where,
                limit=limit
            )
            
            documents = []
            if result['ids']:
                for i in range(len(result['ids'])):
                    metadata = result['metadatas'][i]
                    processed_metadata = {}
                    for k, v in metadata.items():
                        if isinstance(v, str):
                            try:
                                if (v.startswith('[') and v.endswith(']')) or (v.startswith('{') and v.endswith('}')):
                                    processed_metadata[k] = json.loads(v)
                                else:
                                    processed_metadata[k] = v
                            except (json.JSONDecodeError, TypeError):
                                processed_metadata[k] = v
                        else:
                            processed_metadata[k] = v

                    documents.append({
                        'id': result['ids'][i],
                        'document': result['documents'][i],
                        'metadata': processed_metadata
                    })
            
            logger.debug(f"Retrieved {len(documents)} documents from {self.collection_name} with filter")
            return documents
            
        except Exception as e:
            logger.error(f"Error getting documents by metadata from {self.collection_name}: {e}")
            return []
    
    def clear(self) -> bool:
        """
        Delete all documents from the collection.
        
        Warning: This operation cannot be undone!
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get all IDs and delete them
            all_docs = self.collection.get()
            if all_docs['ids']:
                self.collection.delete(ids=all_docs['ids'])
                logger.warning(f"Cleared all {len(all_docs['ids'])} documents from {self.collection_name}")
            else:
                logger.info(f"Collection {self.collection_name} is already empty")
            return True
        except Exception as e:
            logger.error(f"Error clearing {self.collection_name}: {e}")
            return False
    
    def exists(self, doc_id: str) -> bool:
        """
        Check if a document exists in the collection.
        
        Args:
            doc_id: Document identifier
        
        Returns:
            True if document exists, False otherwise
        """
        result = self.get(doc_id)
        return result is not None
    
    def __repr__(self) -> str:
        """String representation of the collection wrapper."""
        return f"{self.__class__.__name__}(collection='{self.collection_name}', count={self.count()})"


# Example usage
if __name__ == "__main__":
    from pydantic import Field
    import chromadb
    
    # Example concrete implementation
    class ExampleMetadata(BaseModel):
        """Example metadata schema."""
        name: str = Field(..., description="Name")
        value: float = Field(..., ge=0.0, description="Value")
    
    class ExampleCollection(BaseCollection):
        """Example collection implementation."""
        def get_schema(self) -> type[BaseModel]:
            return ExampleMetadata
    
    # Test the implementation
    client = chromadb.Client()
    collection = client.create_collection("test_collection")
    
    example_col = ExampleCollection(collection)
    
    # Test add
    example_col.add(
        doc_id="doc1",
        content="This is a test document",
        metadata={"name": "test", "value": 1.5}
    )
    
    # Test get
    doc = example_col.get("doc1")
    print(f"Retrieved: {doc}")
    
    # Test count
    print(f"Count: {example_col.count()}")
    
    # Test search
    results = example_col.search("test document")
    print(f"Search results: {len(results)}")
    
    print("BaseCollection test completed successfully!")
