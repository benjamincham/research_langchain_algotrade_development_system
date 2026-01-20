# ADR 0001: Memory System Serialization and Partial Updates

## Status
Accepted

## Context
The memory system uses ChromaDB for persistent storage of research findings, strategies, and other data. ChromaDB's metadata storage only supports primitive types (str, int, float, bool). However, our data models (defined via Pydantic) often include complex types like lists (e.g., `tickers`, `tags`) and nested dictionaries (e.g., `performance_metrics`, `parameters`).

Additionally, the system needs to support partial updates to metadata (e.g., updating only the performance metrics of a strategy without re-submitting the entire metadata object).

## Decision
1.  **Automatic Serialization**: The `BaseCollection` class will automatically serialize complex metadata fields (lists and dictionaries) to JSON strings before storing them in ChromaDB.
2.  **Automatic Deserialization**: The `BaseCollection` class will automatically deserialize JSON strings back into Python lists or dictionaries when retrieving documents from ChromaDB.
3.  **Partial Updates Support**: The `update` method in `BaseCollection` will fetch the current metadata, merge it with the new partial metadata, and then validate the complete merged object against the Pydantic schema before saving.
4.  **In-Memory Testing**: Added support for `EphemeralClient` in `MemoryManager` when `persist_directory=":memory:"` is used, facilitating faster and more reliable unit testing.

## Consequences
- **Pros**:
    - Transparent handling of complex data structures.
    - Strong validation via Pydantic remains intact.
    - Simplified API for updating specific metadata fields.
    - Improved testability.
- **Cons**:
    - Slight performance overhead due to JSON serialization/deserialization.
    - ChromaDB cannot natively filter based on the contents of serialized JSON strings (though we can still filter on primitive fields).
