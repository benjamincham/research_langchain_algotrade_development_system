# Agent Instructions for Research LangChain AlgoTrade Development System

## Purpose

This document provides instructions for any AI agent (including Manus.ai sessions) to continue development of this project. Read this document first when starting a new session.

## Quick Start

1. **Read the conversation history**: `.manus/CONVERSATION_HISTORY.md`
2. **Check the decision log**: `.manus/DECISION_LOG.md`
3. **Review the current status**: `.manus/PROJECT_STATUS.md`
4. **Understand the design**: `docs/design/SYSTEM_DESIGN.md`
5. **Check the roadmap**: `docs/ROADMAP.md`

## Project Overview

This is a sophisticated LangChain-based agentic workflow pipeline for algorithmic trading research and development. The system uses:

- **LangChain/LangGraph**: Agent orchestration and workflow management
- **ChromaDB**: Vector store for semantic memory
- **yfinance**: Market data source
- **Backtrader**: Trading strategy backtesting framework
- **OpenAI-compatible API**: LLM provider (gpt-4.1-mini)

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        MASTER ORCHESTRATOR                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │ Research    │  │ Strategy    │  │ Backtest    │  │ Quality     │   │
│  │ Swarm       │  │ Agent       │  │ Agent       │  │ Gate Agent  │   │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    MEMORY LAYER (ChromaDB)                              │
└─────────────────────────────────────────────────────────────────────────┘
```

## Key Design Decisions (Approved by User)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| LLM Provider | OpenAI-compatible API | Available in environment |
| Vector Store | ChromaDB | Lightweight, local |
| Data Source | yfinance | Free, comprehensive |
| Trading Framework | Backtrader | Mature, well-documented |
| Strategy Types | Unrestricted | Maximum flexibility |
| Human-in-the-Loop | At start only | Define criteria and direction |

## Critical Design Patterns

### 1. Research Swarm Pattern

The Research Agent is NOT a single agent but an **orchestrated swarm** with:
- **Leader Agent**: Coordinates research strategy, spawns subagents, synthesizes results
- **Subagents**: Specialized workers (Market Research, Technical Analysis, Fundamental Analysis, Sentiment, Pattern Mining)
- **Conflict Resolution**: Weighted confidence voting for contradictory findings

See: `docs/design/RESEARCH_SWARM.md`

### 2. Dynamic Quality Gates

Quality gates are NOT static boolean checks but:
- **Fuzzy Logic Scoring**: Continuous 0-1 scores
- **Statistical Validation**: Confidence intervals, significance tests
- **Adaptive Thresholds**: Adjust based on market regime
- **Detailed Feedback**: Actionable improvement suggestions

See: `docs/design/QUALITY_GATES.md`

### 3. Tool Development Meta-System

Before R&D starts, there's a **Tool Development Phase** that:
- Generates custom metric calculation tools from user specifications
- Validates tools through multi-stage pipeline
- Registers tools with lifecycle management (Draft → Active → Deprecated)
- Validates entire toolchain for integration

See: `docs/design/TOOL_DEVELOPMENT.md`

### 4. Memory with Lineage Tracking

Memory is NOT just storage but includes:
- **UUID-based identification** for all items
- **Parent-child relationships** for provenance
- **Automatic archiving** after 90 days of inactivity
- **Semantic search** via ChromaDB embeddings

See: `docs/design/MEMORY_ARCHITECTURE.md`

## Current Implementation Status

**Phase: Documentation and Design (Complete)**

- ✅ Repository structure created
- ✅ Design documentation complete
- ✅ Agent instructions created
- ⏳ Implementation not started

See `.manus/PROJECT_STATUS.md` for detailed status.

## User Preferences

The user has specified:
1. **Approval required** for every design decision
2. **Quality gates must be dynamic** and unique per project
3. **Research agent must be a swarm** with leader orchestration
4. **Tool development phase** must precede R&D
5. **Systematic toolchain validation** as tools expand
6. **Conversation tracking** for session continuity

## How to Continue Development

### Starting a New Session

1. Read `.manus/CONVERSATION_HISTORY.md` to understand context
2. Check `.manus/PROJECT_STATUS.md` for current phase
3. Review any pending decisions in `.manus/DECISION_LOG.md`
4. Continue from where the last session left off

### Making Changes

1. **Design Changes**: Update relevant `docs/design/*.md` files
2. **Implementation**: Follow the roadmap in `docs/ROADMAP.md`
3. **Decisions**: Log all decisions in `.manus/DECISION_LOG.md`
4. **Progress**: Update `.manus/PROJECT_STATUS.md`

### Seeking User Approval

Before implementing any of the following, ask for user approval:
- New agents or tools
- Changes to quality gate criteria
- Modifications to the workflow pipeline
- Integration with new data sources
- Changes to memory architecture

### Updating Conversation History

After each session, append a summary to `.manus/CONVERSATION_HISTORY.md`:

```markdown
## Session [Date]

### Topics Discussed
- ...

### Decisions Made
- ...

### Actions Taken
- ...

### Next Steps
- ...
```

## File Structure

```
research_langchain_algotrade_development_system/
├── .manus/                          # Agent steering files
│   ├── AGENT_INSTRUCTIONS.md        # This file
│   ├── CONVERSATION_HISTORY.md      # Full conversation log
│   ├── DECISION_LOG.md              # All design decisions
│   └── PROJECT_STATUS.md            # Current status
├── docs/
│   ├── design/                      # Design documentation
│   │   ├── SYSTEM_DESIGN.md         # Overall architecture
│   │   ├── RESEARCH_SWARM.md        # Research swarm design
│   │   ├── QUALITY_GATES.md         # Quality gate system
│   │   ├── TOOL_DEVELOPMENT.md      # Tool meta-system
│   │   ├── MEMORY_ARCHITECTURE.md   # Memory system
│   │   ├── ERROR_HANDLING.md        # Error recovery
│   │   └── DESIGN_REVIEW.md         # Gap analysis
│   ├── ROADMAP.md                   # Implementation roadmap
│   └── API.md                       # API documentation (TBD)
├── src/                             # Source code (TBD)
│   ├── agents/
│   ├── tools/
│   ├── memory/
│   ├── quality_gates/
│   └── workflows/
├── tests/                           # Test suite (TBD)
├── config/                          # Configuration files (TBD)
└── README.md                        # Project overview
```

## Important Notes

1. **No code implementation yet** - User requested documentation first
2. **All design decisions are approved** - See DECISION_LOG.md
3. **Deep review completed** - Using LLM analysis for gap identification
4. **Gaps addressed** - Critical gaps have been addressed in design docs

## Contact

For questions about this project, the user should be consulted. All decisions require user approval.
