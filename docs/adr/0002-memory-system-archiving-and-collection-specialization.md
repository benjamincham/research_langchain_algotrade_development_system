# ADR 0002: Memory System Archiving and Collection Specialization

## Status
Accepted

## Context
As the algorithmic trading research system runs, it generates a large volume of research findings, strategy variants, and market regime observations. Storing all historical data in the active ChromaDB collections can lead to increased memory usage and slower query performance over time.

Furthermore, different types of data (lessons learned, market regimes) require specialized query methods and metadata validation to be useful for the agentic workflow.

## Decision
1.  **Specialized Collection Wrappers**: Implemented `LessonsLearnedCollection` and `MarketRegimesCollection` with domain-specific schemas and helper methods (e.g., `get_current_regime`, `get_failures`).
2.  **Compressed Archiving**: Implemented `ArchiveManager` to move old documents (based on a configurable `older_than_days` threshold) from active ChromaDB collections to compressed GZIP JSON files.
3.  **Data Restoration**: Provided a mechanism to restore archived data back into active collections when historical analysis is required.
4.  **JSON Serialization for Complex Metadata**: Extended the serialization logic to handle nested dictionaries in `MarketRegimesCollection` (e.g., `indicators`) and lists in `LessonsLearnedCollection` (e.g., `tags`).

## Consequences
- **Pros**:
    - Maintains high performance for active research by offloading old data.
    - Reduces memory footprint of the vector database.
    - Provides specialized APIs for agents to query specific types of memory.
    - Ensures data durability through file-based backups.
- **Cons**:
    - Restoring data from archives is slower than querying active memory.
    - Requires management of archive files on disk.
