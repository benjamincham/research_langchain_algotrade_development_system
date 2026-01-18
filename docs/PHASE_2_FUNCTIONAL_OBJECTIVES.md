# Phase 2: Memory System - Functional Objectives & Passing Criteria

**Phase**: Memory System Integration  
**Status**: Ready to Start (Phase 1 Complete)  
**Estimated Effort**: 3.5 days  
**Dependencies**: ✅ Phase 1 (Core Infrastructure)  
**Target Completion**: 2026-01-23

---

## Executive Summary

Phase 2 implements a comprehensive persistent memory system using ChromaDB that enables the algorithmic trading research system to store, retrieve, and track relationships between research findings, trading strategies, lessons learned, and market regimes. The memory system provides semantic search capabilities, lineage tracking for parent-child relationships, archiving for data management, and session state persistence for workflow continuity.

---

## System Architecture

### Core Components

The Memory System consists of eight integrated components working together to provide comprehensive data persistence and retrieval capabilities.

**MemoryManager** serves as the central orchestrator, managing ChromaDB connections and coordinating operations across all collections. It provides a unified interface for database operations while handling connection pooling, error recovery, and logging.

**Collection Classes** implement specialized storage for different entity types. The ResearchFindingsCollection stores market research with semantic embeddings for similarity search. The StrategyLibraryCollection manages trading strategy code and metadata with performance tracking. The LessonsLearnedCollection captures system insights and optimization decisions. The MarketRegimesCollection tracks market conditions and regime transitions over time.

**LineageTracker** maintains parent-child relationships between entities, enabling the system to trace how strategies evolved from research findings, how backtests relate to strategies, and how lessons learned influenced subsequent decisions. This creates an auditable trail of the research process.

**ArchiveManager** handles data lifecycle management by archiving old data to reduce active database size while maintaining the ability to restore historical information when needed. It includes compression and automatic scheduling capabilities.

**SessionStateManager** persists workflow state across system restarts, ensuring that long-running research processes can resume from checkpoints without losing progress.

### Data Flow

Research findings flow into the system through the ResearchFindingsCollection, where they are embedded and indexed for semantic search. When the system generates trading strategies from these findings, the LineageTracker creates parent-child links. As strategies are backtested and optimized, performance metrics are updated in the StrategyLibraryCollection, and lessons learned are captured in the LessonsLearnedCollection. Throughout this process, the SessionStateManager maintains workflow state, while the ArchiveManager periodically moves old data to compressed storage.

---

## Functional Objectives

### Objective 2.1: ChromaDB Integration

**Purpose**: Establish reliable connection to ChromaDB with persistent storage and implement core CRUD operations for all collections.

**Implementation Requirements**:

The MemoryManager class must initialize a ChromaDB client with persistent storage configured to a dedicated directory within the project structure. The client should support both local development and production deployment scenarios. Collection management must include creation, retrieval, listing, and deletion operations with proper error handling.

CRUD operations must support adding documents with automatic embedding generation, querying by similarity using semantic search, filtering by metadata attributes, updating existing documents, and deleting documents by ID. All operations must include comprehensive error handling with user-friendly error messages and detailed logging for debugging.

**Acceptance Criteria**:

The ChromaDB client initializes successfully on first run and connects to existing database on subsequent runs. Persistent storage is configured to a project-specific directory that survives system restarts. Collections can be created programmatically with custom metadata schemas. Documents can be added with text content and metadata, with embeddings generated automatically. Similarity search returns relevant results ranked by semantic similarity. Metadata filtering supports complex queries with multiple conditions. Update operations modify existing documents without creating duplicates. Delete operations remove documents completely and handle non-existent IDs gracefully. All database operations are logged with appropriate detail levels. Errors are caught and wrapped in custom exception classes with clear messages.

**Performance Requirements**:

ChromaDB initialization must complete in under one second. Adding a single document must complete in under 100 milliseconds. Querying by ID must complete in under 50 milliseconds. Similarity search for top 10 results must complete in under 500 milliseconds. Metadata filter queries must complete in under 200 milliseconds. The system must support at least 100 concurrent write operations and 1000 concurrent read operations. Storage capacity must accommodate at least 100,000 documents per collection.

---

### Objective 2.2: Research Findings Storage

**Purpose**: Store research findings with semantic embeddings to enable similarity-based retrieval and support the research swarm's knowledge sharing.

**Implementation Requirements**:

The ResearchFindingsCollection class must implement a validated metadata schema that captures all relevant attributes of research findings. Required metadata includes ticker symbol, finding type (technical, fundamental, sentiment, or pattern), confidence score (0.0 to 1.0), and agent ID. Optional metadata includes data source, timeframe, and custom tags.

The add_finding method must validate all input data, generate a unique finding ID, create semantic embeddings from the finding text, and store everything in ChromaDB. The get_finding method retrieves findings by ID with full metadata. The search_findings method performs semantic similarity search to find related findings. The filter_findings method supports complex metadata queries to find findings matching specific criteria.

**Acceptance Criteria**:

Research findings can be stored with text content and complete metadata. The metadata schema enforces required fields and validates data types. Confidence scores are validated to be between 0.0 and 1.0. Ticker symbols are validated against a known list or pattern. Finding types are restricted to the defined set. Embeddings are generated automatically without manual intervention. Findings can be retrieved by ID with all original metadata intact. Semantic search returns findings ranked by relevance to the query. Metadata filtering supports multiple conditions with AND/OR logic. Related findings can be discovered based on semantic similarity. Invalid data raises ValidationError with clear messages indicating the problem.

**Use Cases**:

A technical analysis agent discovers that AAPL shows strong momentum based on RSI and MACD indicators. It stores this finding with high confidence. Later, when the strategy development agent searches for "Apple stock momentum opportunities," it retrieves this finding along with related findings from other agents. The system can also filter to show only high-confidence technical findings for AAPL from the past week.

---

### Objective 2.3: Strategy Library

**Purpose**: Store trading strategy code, descriptions, and performance metrics to build a searchable library of strategies that can be reused and evolved.

**Implementation Requirements**:

The StrategyLibraryCollection class must store complete strategy implementations including Python code, natural language descriptions, and comprehensive metadata. The metadata schema captures strategy name, type (momentum, mean_reversion, breakout, arbitrage, etc.), target tickers, timeframe, parameters, and performance metrics.

The add_strategy method validates strategy data, stores the code securely, generates embeddings from the description, and assigns a unique strategy ID. The get_strategy method retrieves strategies with code and metadata. The search_strategies method finds strategies based on description similarity. The filter_strategies method supports queries like "all momentum strategies for AAPL" or "strategies with Sharpe ratio > 1.5". The update_performance method updates metrics after backtesting without modifying the strategy code.

**Acceptance Criteria**:

Strategies can be stored with complete code, description, and metadata. Code is stored as text and can be retrieved for execution. Descriptions are embedded for semantic search. Metadata includes all required fields with validation. Performance metrics can be updated independently of strategy code. Strategies can be retrieved by ID with all components intact. Semantic search finds strategies based on description similarity. Metadata filtering supports performance-based queries. Strategies with similar characteristics can be discovered. Invalid data raises ValidationError with specific field information.

**Use Cases**:

The system generates a momentum strategy for AAPL using RSI and MACD. It stores the strategy with code, description, and initial parameters. After backtesting, it updates the performance metrics (Sharpe ratio, max drawdown, win rate). Later, when researching MSFT, it searches for "momentum strategies" and finds this strategy, which can be adapted for the new ticker. The system can also query for "all strategies with Sharpe ratio > 1.5 and max drawdown < 15%" to find high-quality strategies.

---

### Objective 2.4: Lessons Learned

**Purpose**: Capture insights, failures, and optimization decisions to prevent repeating mistakes and accelerate learning across research iterations.

**Implementation Requirements**:

The LessonsLearnedCollection class stores lessons with categorization, context, and actionable recommendations. Each lesson includes a description of what happened, why it happened, what was learned, and what action should be taken in the future. Metadata captures the lesson category (strategy_failure, optimization_insight, data_quality_issue, etc.), severity (low, medium, high, critical), and related entity IDs.

The add_lesson method validates lesson data and creates embeddings for semantic search. The get_lessons_by_category method retrieves all lessons of a specific type. The search_lessons method finds relevant lessons based on current context. The get_related_lessons method finds lessons related to a specific entity (strategy, finding, etc.).

**Acceptance Criteria**:

Lessons can be stored with description, category, severity, and recommendations. Metadata links lessons to related entities (strategies, findings, etc.). Lessons can be retrieved by category for systematic review. Semantic search finds relevant lessons based on current situation. Lessons related to specific entities can be queried. Severity levels are validated against the defined set. Invalid data raises ValidationError with clear messages.

**Use Cases**:

A strategy fails backtesting due to overfitting on historical data. The system stores a lesson: "Strategy overfit to 2020-2021 bull market, failed in 2022 bear market. Learned: Always test across multiple market regimes. Action: Require regime-diverse backtesting before deployment." Later, when developing a new strategy, the system searches for relevant lessons and finds this one, preventing the same mistake.

---

### Objective 2.5: Market Regimes

**Purpose**: Track market conditions and regime transitions to enable regime-aware strategy development and evaluation.

**Implementation Requirements**:

The MarketRegimesCollection class stores market regime data with characteristics and time periods. Each regime includes a regime type (bull_high_vol, bull_low_vol, bear_high_vol, bear_low_vol, sideways, crisis), quantitative characteristics (volatility, trend, momentum), and time boundaries (start_date, end_date).

The add_regime method validates regime data and stores it with embeddings. The get_current_regime method retrieves the most recent regime for a ticker. The get_regime_history method returns all regimes for a ticker within a date range. The detect_regime method analyzes market data to classify the current regime.

**Acceptance Criteria**:

Market regimes can be stored with type, characteristics, and time period. Regime types are validated against the defined set. Characteristics include quantitative metrics (volatility, trend, momentum). Current regime can be queried for any ticker. Regime history can be retrieved for date ranges. Regime detection analyzes market data and returns classification with confidence. Invalid data raises ValidationError with specific field information.

**Use Cases**:

The system analyzes AAPL market data and detects a transition from "bull_low_vol" to "bull_high_vol" regime. It stores this regime with start date, characteristics (volatility=0.35, trend=upward, momentum=0.75), and confidence=0.90. When developing strategies, the system queries regime history to ensure strategies are tested across different regimes. Quality gates check that strategies perform well in the current regime.

---

### Objective 2.6: Lineage Tracking

**Purpose**: Maintain parent-child relationships between entities to create an auditable trail of the research process and enable impact analysis.

**Implementation Requirements**:

The LineageTracker class manages relationships between entities across collections. It supports linking parents to children (e.g., finding → strategy → backtest), querying ancestors (all entities that led to this one), querying descendants (all entities derived from this one), and querying siblings (entities with the same parent).

The link_parent_child method creates a relationship with validation to prevent circular references. The get_ancestors method returns all ancestors in order from immediate parent to root. The get_descendants method returns all descendants in breadth-first order. The get_siblings method returns all entities with the same parent. The get_lineage_tree method returns a hierarchical tree structure showing all relationships.

**Acceptance Criteria**:

Parent-child relationships can be created between any entity types. Circular references are detected and prevented. Ancestors can be queried to any depth. Descendants can be queried to any depth. Siblings can be queried efficiently. Lineage trees can be generated for visualization. Relationships include metadata (relationship type, timestamp). Invalid operations raise ValidationError with clear messages.

**Use Cases**:

A research finding "AAPL shows momentum" leads to strategy "AAPL Momentum v1". This strategy is backtested, creating backtest "BT-001". The strategy is then optimized, creating "AAPL Momentum v2", which is backtested as "BT-002". The lineage tracker maintains these relationships. When analyzing BT-002, the system can query ancestors to see it came from "AAPL Momentum v2" → "AAPL Momentum v1" → "AAPL shows momentum". This enables impact analysis: if the original finding is invalidated, all descendant strategies can be flagged for review.

---

### Objective 2.7: Archiving

**Purpose**: Manage data lifecycle by archiving old data to reduce active database size while maintaining the ability to restore historical information.

**Implementation Requirements**:

The ArchiveManager class implements automatic and manual archiving of old data. It supports archiving data older than a specified number of days, compressing archived data to save storage space, restoring archived data when needed, and listing all available archives.

The archive_old_data method identifies documents older than the threshold, moves them to compressed archive files, and removes them from the active database. The restore_archived_data method decompresses an archive and restores documents to the active database. The list_archives method returns metadata about all archives. An automatic scheduler runs archiving on a configurable schedule.

**Acceptance Criteria**:

Data older than N days can be archived automatically. Archived data is compressed to reduce storage usage. Archived data can be restored completely and accurately. Archives can be listed with metadata (date, collection, count). Archiving does not affect active data or ongoing operations. Automatic archiving runs on schedule without manual intervention. Archive files are stored securely with checksums for integrity. Invalid operations raise appropriate errors with clear messages.

**Use Cases**:

The system runs for months, accumulating thousands of research findings. Many findings are from old research sessions and no longer relevant. The ArchiveManager automatically archives findings older than 90 days, reducing the active database size by 70%. When a user wants to review historical research from 6 months ago, they restore that archive, making the old findings available for analysis.

---

### Objective 2.8: Session State Management

**Purpose**: Persist workflow state across system restarts to enable long-running research processes to resume from checkpoints.

**Implementation Requirements**:

The SessionStateManager class manages workflow state persistence. It supports saving current workflow state (active agents, pending tasks, intermediate results), loading workflow state on restart, clearing completed workflow state, and listing all active sessions.

The save_session method serializes workflow state and stores it with a session ID. The load_session method retrieves and deserializes workflow state. The clear_session method removes completed session state. The list_sessions method returns all active sessions with metadata.

**Acceptance Criteria**:

Workflow state can be saved at any point during execution. Saved state includes all necessary information to resume. State can be loaded on system restart and workflow resumes correctly. Completed sessions can be cleared to free storage. Active sessions can be listed with metadata (start time, status, progress). State is stored securely and validated on load. Invalid state raises appropriate errors with recovery options.

**Use Cases**:

A research workflow is analyzing 50 stocks, generating strategies for each. After processing 30 stocks, the system crashes. On restart, the SessionStateManager loads the saved state, showing that 30 stocks are complete and 20 remain. The workflow resumes from stock 31, avoiding redundant work. All intermediate results (findings, strategies) are preserved.

---

## Passing Criteria

### Must Pass (10 Critical Requirements)

**ChromaDB Connection**: The MemoryManager must successfully initialize a ChromaDB client with persistent storage. The client must connect to an existing database on subsequent runs without data loss. Connection errors must be caught and logged with clear recovery instructions.

**Collection Management**: All four collection classes (ResearchFindings, StrategyLibrary, LessonsLearned, MarketRegimes) must be created and accessible. Collections must persist across system restarts. Collection schemas must be enforced with validation.

**CRUD Operations**: All collections must support Create, Read, Update, and Delete operations. Operations must complete within performance requirements. Errors must be handled gracefully with rollback where appropriate.

**Semantic Search**: Similarity search must return relevant results ranked by semantic similarity. Search must work across all collections. Performance must meet requirements (< 500ms for top 10 results).

**Metadata Filtering**: Complex metadata queries must work with AND/OR logic. Filtering must support all defined metadata fields. Performance must meet requirements (< 200ms).

**Lineage Tracking**: Parent-child relationships must be created and queried correctly. Circular references must be prevented. Lineage trees must be generated accurately.

**Data Validation**: All input data must be validated before storage. Invalid data must raise ValidationError with specific field information. Validation must enforce all schema requirements.

**Error Handling**: All database errors must be caught and wrapped in custom exceptions. Error messages must be user-friendly and actionable. Errors must be logged with full context for debugging.

**Performance Requirements**: All operations must meet specified performance targets. The system must support specified concurrent operation limits. Storage capacity must meet requirements.

**Integration Test**: A complete end-to-end test must pass, demonstrating: storing a research finding, generating a strategy from it, linking them with lineage, storing the strategy, searching for related findings, filtering by metadata, and retrieving the lineage tree.

### Should Pass (5 Important Requirements)

**Cross-Collection Queries**: Queries that span multiple collections must work correctly. For example, finding all strategies derived from findings about a specific ticker.

**Archive and Restore**: The archiving system must successfully archive old data and restore it accurately. Compression must reduce storage size significantly. Automatic archiving must run on schedule.

**Session State Persistence**: Workflow state must be saved and restored correctly. Resumed workflows must continue from the exact checkpoint. All intermediate results must be preserved.

**Performance Benchmarks**: All operations must meet or exceed performance requirements under load. The system must handle specified concurrent operations without degradation.

**Documentation**: All classes and methods must have comprehensive docstrings. Usage examples must be provided for all major features. A user guide must be created for the memory system.

### Nice to Have (3 Enhancement Features)

**Visualization**: Lineage trees should be visualizable as graphs. Regime transitions should be plotable as timelines. Strategy performance should be visualizable with charts.

**Advanced Search**: Hybrid search combining semantic similarity and metadata filtering should be supported. Search should support relevance boosting based on recency or confidence.

**Automatic Optimization**: The system should automatically optimize database performance. Indexes should be created for frequently queried fields. Query plans should be analyzed and optimized.

---

## Test Cases Summary

### Unit Tests (40+ test cases)

**ChromaDB Integration (8 tests)**: Initialize client, create collection, add document, query by similarity, query by metadata, update document, delete document, error handling.

**Research Findings (6 tests)**: Add finding, get finding, search findings, filter findings, validation, get related findings.

**Strategy Library (6 tests)**: Add strategy, get strategy, search strategies, filter strategies, update performance, validation.

**Lessons Learned (5 tests)**: Add lesson, get by category, search lessons, get related lessons, validation.

**Market Regimes (4 tests)**: Add regime, get current regime, get regime history, detect regime.

**Lineage Tracking (6 tests)**: Link parent-child, get ancestors, get descendants, get siblings, prevent circular references, get lineage tree.

**Archiving (5 tests)**: Archive old data, restore archived data, list archives, compression, automatic scheduling.

**Session State (4 tests)**: Save session, load session, clear session, list sessions.

### Integration Tests (3 test cases)

**End-to-End Research Flow**: Store finding → generate strategy → link lineage → backtest → store lesson → query lineage tree.

**Cross-Collection Query**: Find all strategies derived from findings about AAPL with Sharpe ratio > 1.5.

**Archive and Restore**: Archive data → verify removal → restore data → verify restoration.

---

## Implementation Checklist

### Phase 2.1: ChromaDB Integration (Day 1)
- [ ] Install ChromaDB and dependencies
- [ ] Implement MemoryManager class
- [ ] Implement collection management methods
- [ ] Implement CRUD operations
- [ ] Add error handling and logging
- [ ] Write 8 unit tests
- [ ] All tests passing
- [ ] Commit and push

### Phase 2.2: Collection Classes (Day 2)
- [ ] Implement ResearchFindingsCollection
- [ ] Implement StrategyLibraryCollection
- [ ] Implement LessonsLearnedCollection
- [ ] Implement MarketRegimesCollection
- [ ] Define all metadata schemas
- [ ] Add validation for all collections
- [ ] Write 21 unit tests
- [ ] All tests passing
- [ ] Commit and push

### Phase 2.3: Lineage and Archiving (Day 3)
- [ ] Implement LineageTracker class
- [ ] Implement ArchiveManager class
- [ ] Implement SessionStateManager class
- [ ] Add circular reference prevention
- [ ] Add compression for archives
- [ ] Write 15 unit tests
- [ ] All tests passing
- [ ] Commit and push

### Phase 2.4: Integration and Testing (Day 4)
- [ ] Write 3 integration tests
- [ ] Run all tests (44 unit + 3 integration)
- [ ] Fix any failing tests
- [ ] Performance benchmarking
- [ ] Update documentation
- [ ] Create Phase 2 completion report
- [ ] Final commit and push

---

## Success Metrics

**Test Coverage**: 100% of implemented features have unit tests. All critical paths have integration tests. Test-to-code ratio > 1:1.

**Performance**: All operations meet performance requirements. System handles specified concurrent operations. No performance degradation under load.

**Reliability**: Zero data loss during normal operations. All errors are caught and handled. System recovers gracefully from failures.

**Usability**: API is intuitive and well-documented. Error messages are clear and actionable. Examples are provided for all major features.

---

**End of Phase 2 Functional Objectives**
