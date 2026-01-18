from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid
from src.core.logging import logger

class LineageTracker:
    """Tracks the lineage of research findings and strategies."""
    
    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        logger.info("Initialized LineageTracker")

    def add_node(
        self, 
        node_id: str, 
        node_type: str, 
        content_hash: str,
        parent_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add a node to the lineage graph."""
        self.nodes[node_id] = {
            "id": node_id,
            "type": node_type,
            "content_hash": content_hash,
            "parents": parent_ids or [],
            "children": [],
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        # Update children of parents
        if parent_ids:
            for p_id in parent_ids:
                if p_id in self.nodes:
                    self.nodes[p_id]["children"].append(node_id)
        
        logger.debug(f"Added lineage node: {node_id} ({node_type})")

    def get_ancestry(self, node_id: str) -> List[str]:
        """Get all ancestors of a node."""
        ancestors = []
        queue = self.nodes.get(node_id, {}).get("parents", []).copy()
        
        while queue:
            current = queue.pop(0)
            if current not in ancestors:
                ancestors.append(current)
                queue.extend(self.nodes.get(current, {}).get("parents", []))
        
        return ancestors

    def get_descendants(self, node_id: str) -> List[str]:
        """Get all descendants of a node."""
        descendants = []
        queue = self.nodes.get(node_id, {}).get("children", []).copy()
        
        while queue:
            current = queue.pop(0)
            if current not in descendants:
                descendants.append(current)
                queue.extend(self.nodes.get(current, {}).get("children", []))
        
        return descendants

# Singleton instance
_lineage_tracker = None

def get_lineage_tracker():
    """Get or initialize the global LineageTracker instance."""
    global _lineage_tracker
    if _lineage_tracker is None:
        _lineage_tracker = LineageTracker()
    return _lineage_tracker
