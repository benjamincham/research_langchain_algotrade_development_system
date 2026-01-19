"""
Lineage Tracker for Document Relationships

This module provides the LineageTracker class that maintains parent-child
relationships between documents across collections.

Maintains a directed acyclic graph (DAG) to prevent circular dependencies.

Design Reference: docs/design/MEMORY_ARCHITECTURE.md
Phase: 2 - Memory System
Task: 2.4 - Lineage Tracker
"""

from typing import Dict, List, Optional, Literal, Set
from pathlib import Path
import json
from datetime import datetime
from loguru import logger


RelationshipType = Literal[
    "derived_from",      # Strategy derived from finding/strategy
    "based_on",          # Strategy based on research
    "refined_from",      # Strategy refined from previous version
    "informed_by",       # Decision informed by lesson
    "applies_to"         # Strategy applies to regime
]


class LineageTracker:
    """
    Track parent-child relationships between documents.
    
    Maintains a directed acyclic graph (DAG) of relationships to enable:
    - Strategy evolution tracking
    - Research lineage
    - Dependency analysis
    - Impact analysis
    
    The graph structure:
    {
        "nodes": {
            "node_id": {
                "type": "research_finding",
                "collection": "research_findings",
                "created_at": "2026-01-18T10:00:00",
                "metadata": {...}
            }
        },
        "edges": [
            {
                "from": "node_id_1",
                "to": "node_id_2",
                "relationship": "derived_from",
                "timestamp": "2026-01-18T11:00:00"
            }
        ]
    }
    """
    
    def __init__(self, persist_path: str = "./data/lineage_graph.json"):
        """
        Initialize lineage tracker.
        
        Args:
            persist_path: Path to persist the lineage graph
        """
        self.persist_path = Path(persist_path)
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing graph or create new
        if self.persist_path.exists():
            with open(self.persist_path, 'r') as f:
                self.graph = json.load(f)
            logger.info(f"Loaded lineage graph with {len(self.graph['nodes'])} nodes and {len(self.graph['edges'])} edges")
        else:
            self.graph = {"nodes": {}, "edges": []}
            self._save()
            logger.info("Created new lineage graph")
    
    def add_node(
        self,
        node_id: str,
        node_type: str,
        collection: str,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Add a node to the lineage graph.
        
        Args:
            node_id: Unique identifier
            node_type: Type of node (research_finding, strategy, lesson, regime)
            collection: ChromaDB collection name
            metadata: Additional metadata
        
        Returns:
            True if successful, False if node already exists
        """
        if node_id in self.graph['nodes']:
            logger.warning(f"Node '{node_id}' already exists in lineage graph")
            return False
        
        self.graph['nodes'][node_id] = {
            "type": node_type,
            "collection": collection,
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self._save()
        logger.debug(f"Added node '{node_id}' (type={node_type}, collection={collection})")
        return True
    
    def add_edge(
        self,
        from_id: str,
        to_id: str,
        relationship: RelationshipType
    ) -> bool:
        """
        Add an edge (relationship) between two nodes.
        
        Args:
            from_id: Source node ID
            to_id: Target node ID
            relationship: Type of relationship
        
        Returns:
            True if successful, False otherwise
        
        Raises:
            ValueError: If adding edge would create a cycle
            KeyError: If either node doesn't exist
        """
        # Check nodes exist
        if from_id not in self.graph['nodes']:
            raise KeyError(f"Source node '{from_id}' does not exist")
        if to_id not in self.graph['nodes']:
            raise KeyError(f"Target node '{to_id}' does not exist")
        
        # Check for circular dependencies
        if self._would_create_cycle(from_id, to_id):
            raise ValueError(
                f"Adding edge {from_id} -> {to_id} would create a cycle. "
                "Lineage graph must be a DAG (directed acyclic graph)."
            )
        
        # Check if edge already exists
        for edge in self.graph['edges']:
            if edge['from'] == from_id and edge['to'] == to_id and edge['relationship'] == relationship:
                logger.warning(f"Edge {from_id} --[{relationship}]--> {to_id} already exists")
                return False
        
        edge = {
            "from": from_id,
            "to": to_id,
            "relationship": relationship,
            "timestamp": datetime.now().isoformat()
        }
        self.graph['edges'].append(edge)
        self._save()
        logger.debug(f"Added edge: {from_id} --[{relationship}]--> {to_id}")
        return True
    
    def get_node(self, node_id: str) -> Optional[Dict]:
        """
        Get a node by ID.
        
        Args:
            node_id: Node ID
        
        Returns:
            Node data or None if not found
        """
        return self.graph['nodes'].get(node_id)
    
    def get_parents(self, node_id: str) -> List[Dict]:
        """
        Get all parent nodes (nodes that this node derives from).
        
        Args:
            node_id: Node ID
        
        Returns:
            List of parent nodes with relationship info
        """
        parents = []
        for edge in self.graph['edges']:
            if edge['to'] == node_id:
                parent_node = self.graph['nodes'].get(edge['from'])
                if parent_node:
                    parents.append({
                        "node_id": edge['from'],
                        "relationship": edge['relationship'],
                        "timestamp": edge['timestamp'],
                        "node_data": parent_node
                    })
        return parents
    
    def get_children(self, node_id: str) -> List[Dict]:
        """
        Get all child nodes (nodes that derive from this node).
        
        Args:
            node_id: Node ID
        
        Returns:
            List of child nodes with relationship info
        """
        children = []
        for edge in self.graph['edges']:
            if edge['from'] == node_id:
                child_node = self.graph['nodes'].get(edge['to'])
                if child_node:
                    children.append({
                        "node_id": edge['to'],
                        "relationship": edge['relationship'],
                        "timestamp": edge['timestamp'],
                        "node_data": child_node
                    })
        return children
    
    def get_lineage(self, node_id: str, max_depth: int = 10) -> Dict:
        """
        Get complete lineage (ancestors and descendants) for a node.
        
        Args:
            node_id: Node ID
            max_depth: Maximum depth to traverse
        
        Returns:
            Dictionary with node info, ancestors, and descendants
        """
        node = self.get_node(node_id)
        if not node:
            logger.warning(f"Node '{node_id}' not found")
            return {
                "node_id": node_id,
                "node_data": None,
                "ancestors": [],
                "descendants": []
            }
        
        return {
            "node_id": node_id,
            "node_data": node,
            "ancestors": self._get_ancestors(node_id, max_depth),
            "descendants": self._get_descendants(node_id, max_depth)
        }
    
    def _get_ancestors(
        self,
        node_id: str,
        max_depth: int,
        current_depth: int = 0,
        visited: Optional[Set[str]] = None
    ) -> List[Dict]:
        """
        Recursively get all ancestors.
        
        Args:
            node_id: Node ID
            max_depth: Maximum depth to traverse
            current_depth: Current recursion depth
            visited: Set of visited nodes (to prevent infinite loops)
        
        Returns:
            List of ancestor nodes with depth info
        """
        if visited is None:
            visited = set()
        
        if current_depth >= max_depth or node_id in visited:
            return []
        
        visited.add(node_id)
        ancestors = []
        
        for parent in self.get_parents(node_id):
            ancestors.append({
                **parent,
                "depth": current_depth + 1
            })
            # Recursively get parent's ancestors
            ancestors.extend(
                self._get_ancestors(
                    parent['node_id'],
                    max_depth,
                    current_depth + 1,
                    visited
                )
            )
        
        return ancestors
    
    def _get_descendants(
        self,
        node_id: str,
        max_depth: int,
        current_depth: int = 0,
        visited: Optional[Set[str]] = None
    ) -> List[Dict]:
        """
        Recursively get all descendants.
        
        Args:
            node_id: Node ID
            max_depth: Maximum depth to traverse
            current_depth: Current recursion depth
            visited: Set of visited nodes (to prevent infinite loops)
        
        Returns:
            List of descendant nodes with depth info
        """
        if visited is None:
            visited = set()
        
        if current_depth >= max_depth or node_id in visited:
            return []
        
        visited.add(node_id)
        descendants = []
        
        for child in self.get_children(node_id):
            descendants.append({
                **child,
                "depth": current_depth + 1
            })
            # Recursively get child's descendants
            descendants.extend(
                self._get_descendants(
                    child['node_id'],
                    max_depth,
                    current_depth + 1,
                    visited
                )
            )
        
        return descendants
    
    def _would_create_cycle(self, from_id: str, to_id: str) -> bool:
        """
        Check if adding an edge would create a cycle.
        
        If to_id can reach from_id through existing edges,
        then adding from_id -> to_id would create a cycle.
        
        Args:
            from_id: Source node ID
            to_id: Target node ID
        
        Returns:
            True if adding edge would create cycle, False otherwise
        """
        # If to_id can reach from_id, adding edge would create cycle
        descendants = self._get_descendants(to_id, max_depth=100)
        return any(d['node_id'] == from_id for d in descendants)
    
    def delete_node(self, node_id: str) -> bool:
        """
        Delete a node and all its edges.
        
        Args:
            node_id: Node ID
        
        Returns:
            True if successful, False if node doesn't exist
        """
        if node_id not in self.graph['nodes']:
            logger.warning(f"Node '{node_id}' not found")
            return False
        
        # Delete node
        del self.graph['nodes'][node_id]
        
        # Delete all edges involving this node
        self.graph['edges'] = [
            edge for edge in self.graph['edges']
            if edge['from'] != node_id and edge['to'] != node_id
        ]
        
        self._save()
        logger.info(f"Deleted node '{node_id}' and all its edges")
        return True
    
    def delete_edge(
        self,
        from_id: str,
        to_id: str,
        relationship: Optional[RelationshipType] = None
    ) -> bool:
        """
        Delete an edge between two nodes.
        
        Args:
            from_id: Source node ID
            to_id: Target node ID
            relationship: Relationship type (if None, deletes all edges between nodes)
        
        Returns:
            True if at least one edge was deleted, False otherwise
        """
        original_count = len(self.graph['edges'])
        
        if relationship:
            # Delete specific edge
            self.graph['edges'] = [
                edge for edge in self.graph['edges']
                if not (edge['from'] == from_id and edge['to'] == to_id and edge['relationship'] == relationship)
            ]
        else:
            # Delete all edges between nodes
            self.graph['edges'] = [
                edge for edge in self.graph['edges']
                if not (edge['from'] == from_id and edge['to'] == to_id)
            ]
        
        deleted_count = original_count - len(self.graph['edges'])
        
        if deleted_count > 0:
            self._save()
            logger.info(f"Deleted {deleted_count} edge(s) from {from_id} to {to_id}")
            return True
        else:
            logger.warning(f"No edges found from {from_id} to {to_id}")
            return False
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the lineage graph.
        
        Returns:
            Dictionary with node count, edge count, node types, etc.
        """
        node_types = {}
        for node_data in self.graph['nodes'].values():
            node_type = node_data['type']
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        relationship_types = {}
        for edge in self.graph['edges']:
            rel_type = edge['relationship']
            relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
        
        return {
            "total_nodes": len(self.graph['nodes']),
            "total_edges": len(self.graph['edges']),
            "node_types": node_types,
            "relationship_types": relationship_types
        }
    
    def _save(self) -> None:
        """Persist graph to disk."""
        try:
            with open(self.persist_path, 'w') as f:
                json.dump(self.graph, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving lineage graph: {e}")
    
    def export_dot(self, output_path: str) -> bool:
        """
        Export lineage graph to DOT format for visualization with Graphviz.
        
        Args:
            output_path: Path to output .dot file
        
        Returns:
            True if successful, False otherwise
        
        Example:
            tracker.export_dot("lineage.dot")
            # Then: dot -Tpng lineage.dot -o lineage.png
        """
        try:
            lines = ["digraph Lineage {"]
            lines.append("  rankdir=LR;")
            lines.append("  node [shape=box, style=rounded];")
            lines.append("")
            
            # Add nodes with labels
            for node_id, node_data in self.graph['nodes'].items():
                node_type = node_data['type']
                label = f"{node_id}\\n({node_type})"
                
                # Color by type
                color = {
                    "research_finding": "lightblue",
                    "strategy": "lightgreen",
                    "lesson": "lightyellow",
                    "regime": "lightpink"
                }.get(node_type, "lightgray")
                
                lines.append(f'  "{node_id}" [label="{label}", fillcolor="{color}", style="rounded,filled"];')
            
            lines.append("")
            
            # Add edges with labels
            for edge in self.graph['edges']:
                from_id = edge['from']
                to_id = edge['to']
                label = edge['relationship']
                lines.append(f'  "{from_id}" -> "{to_id}" [label="{label}"];')
            
            lines.append("}")
            
            with open(output_path, 'w') as f:
                f.write("\n".join(lines))
            
            logger.info(f"Exported lineage graph to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting lineage graph: {e}")
            return False
    
    def clear(self) -> None:
        """
        Clear all nodes and edges from the graph.
        
        Warning: This operation cannot be undone!
        """
        self.graph = {"nodes": {}, "edges": []}
        self._save()
        logger.warning("Cleared all nodes and edges from lineage graph")
    
    def __repr__(self) -> str:
        """String representation of the lineage tracker."""
        stats = self.get_stats()
        return f"LineageTracker(nodes={stats['total_nodes']}, edges={stats['total_edges']})"


# Example usage
if __name__ == "__main__":
    # Initialize tracker
    tracker = LineageTracker(persist_path="./data/test_lineage.json")
    tracker.clear()  # Start fresh for testing
    
    # Add nodes
    tracker.add_node("finding_001", "research_finding", "research_findings", {"ticker": "AAPL"})
    tracker.add_node("finding_002", "research_finding", "research_findings", {"ticker": "AAPL"})
    tracker.add_node("strategy_001", "strategy", "strategy_library", {"name": "AAPL Momentum"})
    tracker.add_node("strategy_002", "strategy", "strategy_library", {"name": "AAPL Momentum v2"})
    
    # Add edges
    tracker.add_edge("finding_001", "strategy_001", "based_on")
    tracker.add_edge("finding_002", "strategy_001", "based_on")
    tracker.add_edge("strategy_001", "strategy_002", "refined_from")
    
    # Test cycle detection
    try:
        tracker.add_edge("strategy_002", "finding_001", "derived_from")
        print("ERROR: Should have detected cycle!")
    except ValueError as e:
        print(f"✓ Cycle detection working: {e}")
    
    # Get lineage
    lineage = tracker.get_lineage("strategy_002")
    print(f"\nLineage for strategy_002:")
    print(f"  Ancestors: {len(lineage['ancestors'])}")
    print(f"  Descendants: {len(lineage['descendants'])}")
    
    # Get stats
    stats = tracker.get_stats()
    print(f"\nGraph stats: {stats}")
    
    # Export to DOT
    tracker.export_dot("./data/test_lineage.dot")
    print("\n✓ Exported to DOT format")
    
    print(f"\n{tracker}")
    print("LineageTracker test completed successfully!")
