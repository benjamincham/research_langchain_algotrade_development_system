"""
ArchiveManager - Manage archiving and restoration of old data.

This module provides functionality to archive old documents from ChromaDB
collections to compressed JSON files, reducing memory usage while maintaining
the ability to restore data when needed.
"""

from typing import Optional, Dict, Any, List
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
    
    Example:
        >>> archive_manager = ArchiveManager(archive_dir="./data/archives")
        >>> 
        >>> # Archive old documents
        >>> result = archive_manager.archive_collection(
        ...     collection=my_collection,
        ...     older_than_days=90,
        ...     compress=True
        ... )
        >>> print(f"Archived {result['archived_count']} documents")
        >>> 
        >>> # List archives
        >>> archives = archive_manager.list_archives()
        >>> 
        >>> # Restore from archive
        >>> count = archive_manager.restore_archive(
        ...     archive_path=result['archive_path'],
        ...     collection=my_collection
        ... )
    """
    
    def __init__(self, archive_dir: str = "./data/archives"):
        """
        Initialize archive manager.
        
        Args:
            archive_dir: Directory to store archive files
        """
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
        
        Documents are filtered by timestamp in metadata. Documents older than
        the specified threshold are removed from the collection and saved to
        an archive file.
        
        Args:
            collection: ChromaDB collection
            older_than_days: Archive documents older than this many days
            compress: Whether to compress the archive file with gzip
        
        Returns:
            Dictionary with archive stats:
                - archived_count: Number of documents archived
                - archive_path: Path to archive file (or None if nothing archived)
                
        Example:
            >>> result = archive_manager.archive_collection(
            ...     collection=findings_collection.collection,
            ...     older_than_days=90,
            ...     compress=True
            ... )
            >>> print(f"Archived {result['archived_count']} documents")
        """
        collection_name = collection.name
        cutoff_timestamp = datetime.now().timestamp() - (older_than_days * 24 * 60 * 60)
        
        logger.info(
            f"Archiving documents from {collection_name} older than {older_than_days} days"
        )
        
        # Get all documents
        try:
            all_docs = collection.get()
        except Exception as e:
            logger.error(f"Error getting documents from {collection_name}: {e}")
            return {"archived_count": 0, "archive_path": None}
        
        if not all_docs or not all_docs.get('ids'):
            logger.info(f"No documents found in {collection_name}")
            return {"archived_count": 0, "archive_path": None}
        
        # Filter old documents
        old_docs = []
        old_ids = []
        
        for i, doc_id in enumerate(all_docs['ids']):
            metadata = all_docs['metadatas'][i] if all_docs.get('metadatas') else {}
            timestamp_str = metadata.get('timestamp', '')
            
            # Parse timestamp and check if old
            if timestamp_str:
                try:
                    # Try ISO format first
                    doc_timestamp = datetime.fromisoformat(timestamp_str).timestamp()
                    
                    if doc_timestamp < cutoff_timestamp:
                        old_docs.append({
                            'id': doc_id,
                            'document': all_docs['documents'][i] if all_docs.get('documents') else '',
                            'metadata': metadata,
                            'embedding': all_docs['embeddings'][i] if all_docs.get('embeddings') else None
                        })
                        old_ids.append(doc_id)
                except (ValueError, TypeError) as e:
                    logger.warning(
                        f"Error parsing timestamp '{timestamp_str}' for {doc_id}: {e}"
                    )
            else:
                logger.debug(f"No timestamp found for document {doc_id}, skipping")
        
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
        
        try:
            if compress:
                with gzip.open(archive_path, 'wt', encoding='utf-8') as f:
                    json.dump(archive_data, f, indent=2)
            else:
                with open(archive_path, 'w', encoding='utf-8') as f:
                    json.dump(archive_data, f, indent=2)
            
            logger.info(f"Created archive file: {archive_path}")
        except Exception as e:
            logger.error(f"Error creating archive file {archive_path}: {e}")
            return {"archived_count": 0, "archive_path": None}
        
        # Delete archived documents from collection
        try:
            collection.delete(ids=old_ids)
            logger.info(
                f"Archived {len(old_docs)} documents from {collection_name} to {archive_path}"
            )
        except Exception as e:
            logger.error(f"Error deleting archived documents from {collection_name}: {e}")
            # Archive file created but deletion failed
            return {"archived_count": 0, "archive_path": str(archive_path)}
        
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
            
        Raises:
            FileNotFoundError: If archive file doesn't exist
            
        Example:
            >>> restored_count = archive_manager.restore_archive(
            ...     archive_path="./data/archives/research_findings_20240119.json.gz",
            ...     collection=findings_collection.collection
            ... )
            >>> print(f"Restored {restored_count} documents")
        """
        archive_path = Path(archive_path)
        
        if not archive_path.exists():
            raise FileNotFoundError(f"Archive file not found: {archive_path}")
        
        logger.info(f"Restoring documents from {archive_path}")
        
        # Read archive
        try:
            if archive_path.suffix == '.gz':
                with gzip.open(archive_path, 'rt', encoding='utf-8') as f:
                    archive_data = json.load(f)
            else:
                with open(archive_path, 'r', encoding='utf-8') as f:
                    archive_data = json.load(f)
        except Exception as e:
            logger.error(f"Error reading archive file {archive_path}: {e}")
            raise
        
        documents = archive_data.get('documents', [])
        
        if not documents:
            logger.warning(f"No documents found in archive {archive_path}")
            return 0
        
        # Restore to collection
        ids = [doc['id'] for doc in documents]
        docs = [doc['document'] for doc in documents]
        metadatas = [doc['metadata'] for doc in documents]
        
        # Embeddings are optional (ChromaDB will regenerate if missing)
        embeddings = [doc.get('embedding') for doc in documents]
        has_embeddings = all(e is not None for e in embeddings)
        
        try:
            if has_embeddings:
                collection.add(
                    ids=ids,
                    documents=docs,
                    metadatas=metadatas,
                    embeddings=embeddings
                )
            else:
                collection.add(
                    ids=ids,
                    documents=docs,
                    metadatas=metadatas
                )
            
            logger.info(
                f"Restored {len(documents)} documents from {archive_path} to {collection.name}"
            )
        except Exception as e:
            logger.error(f"Error restoring documents to {collection.name}: {e}")
            raise
        
        return len(documents)
    
    def list_archives(
        self, 
        collection_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List all archive files.
        
        Args:
            collection_name: Filter by collection name (optional)
        
        Returns:
            List of archive info dictionaries with keys:
                - collection: Collection name
                - path: Full path to archive file
                - size_bytes: File size in bytes
                - created_at: Creation timestamp
                
        Example:
            >>> # List all archives
            >>> all_archives = archive_manager.list_archives()
            >>> 
            >>> # List archives for specific collection
            >>> findings_archives = archive_manager.list_archives(
            ...     collection_name="research_findings"
            ... )
        """
        archives = []
        
        # Find all JSON and JSON.GZ files
        for archive_file in self.archive_dir.glob("*.json*"):
            # Parse filename: collection_name_timestamp.json[.gz]
            stem = archive_file.stem
            if stem.endswith('.json'):
                stem = stem[:-5]  # Remove .json from stem
            
            parts = stem.split('_')
            if len(parts) >= 3:
                # Last 2 parts are timestamp (YYYYMMDD_HHMMSS)
                col_name = '_'.join(parts[:-2])
                
                if collection_name and col_name != collection_name:
                    continue
                
                try:
                    archives.append({
                        "collection": col_name,
                        "path": str(archive_file),
                        "size_bytes": archive_file.stat().st_size,
                        "created_at": datetime.fromtimestamp(
                            archive_file.stat().st_ctime
                        ).isoformat()
                    })
                except Exception as e:
                    logger.warning(f"Error processing archive file {archive_file}: {e}")
        
        # Sort by creation time descending (newest first)
        archives.sort(key=lambda x: x['created_at'], reverse=True)
        
        logger.debug(f"Found {len(archives)} archive files")
        return archives
    
    def delete_archive(self, archive_path: str) -> bool:
        """
        Delete an archive file.
        
        Args:
            archive_path: Path to archive file
        
        Returns:
            True if successful, False otherwise
            
        Example:
            >>> success = archive_manager.delete_archive(
            ...     "./data/archives/old_archive.json.gz"
            ... )
        """
        try:
            archive_file = Path(archive_path)
            if archive_file.exists():
                archive_file.unlink()
                logger.info(f"Deleted archive: {archive_path}")
                return True
            else:
                logger.warning(f"Archive file not found: {archive_path}")
                return False
        except Exception as e:
            logger.error(f"Error deleting archive {archive_path}: {e}")
            return False
    
    def get_archive_info(self, archive_path: str) -> Optional[Dict[str, Any]]:
        """
        Get information about an archive file without loading all documents.
        
        Args:
            archive_path: Path to archive file
            
        Returns:
            Dictionary with archive metadata or None if error
            
        Example:
            >>> info = archive_manager.get_archive_info(
            ...     "./data/archives/research_findings_20240119.json.gz"
            ... )
            >>> print(f"Archive contains {info['document_count']} documents")
        """
        archive_path = Path(archive_path)
        
        if not archive_path.exists():
            logger.error(f"Archive file not found: {archive_path}")
            return None
        
        try:
            if archive_path.suffix == '.gz':
                with gzip.open(archive_path, 'rt', encoding='utf-8') as f:
                    archive_data = json.load(f)
            else:
                with open(archive_path, 'r', encoding='utf-8') as f:
                    archive_data = json.load(f)
            
            # Return metadata only (not documents)
            return {
                "collection": archive_data.get('collection'),
                "archived_at": archive_data.get('archived_at'),
                "older_than_days": archive_data.get('older_than_days'),
                "document_count": archive_data.get('document_count'),
                "file_size_bytes": archive_path.stat().st_size
            }
        except Exception as e:
            logger.error(f"Error reading archive info from {archive_path}: {e}")
            return None
