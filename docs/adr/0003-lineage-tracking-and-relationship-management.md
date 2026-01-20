# ADR 0003: Lineage Tracking and Relationship Management

## Status
Accepted

## Context
In an agentic research workflow, it is crucial to understand how trading strategies are derived from research findings, how they evolve over time, and how they are informed by lessons learned. ChromaDB provides vector search but does not natively support complex graph-based relationships between documents across different collections.

## Decision
1.  **Dedicated Lineage Tracker**: Implemented a `LineageTracker` class that maintains a Directed Acyclic Graph (DAG) of relationships between documents.
2.  **JSON Persistence**: The lineage graph is persisted as a JSON file (`lineage_graph.json`), allowing it to be easily loaded and shared across different components of the system.
3.  **Cycle Detection**: Implemented strict cycle detection using a Depth-First Search (DFS) algorithm to ensure the lineage remains a DAG, preventing circular dependencies that would break reasoning.
4.  **Relationship Types**: Defined a set of standard relationship types (`derived_from`, `based_on`, `refined_from`, `informed_by`, `applies_to`) to categorize how data points are connected.
5.  **Lineage Traversal**: Provided methods to retrieve full ancestors and descendants for any given node, enabling impact analysis and research provenance.

## Consequences
- **Pros**:
    - Clear visibility into the "why" behind every strategy.
    - Enables automated impact analysis (e.g., "if this research finding is invalidated, which strategies are affected?").
    - Prevents logical errors through cycle detection.
    - Lightweight and easy to inspect via JSON.
- **Cons**:
    - Requires manual management of nodes and edges alongside ChromaDB operations.
    - The JSON file could grow large over time, though it only stores metadata and relationships, not the full document content.
