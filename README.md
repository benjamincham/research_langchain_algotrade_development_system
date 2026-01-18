# Research LangChain AlgoTrade Development System

[![Status](https://img.shields.io/badge/Status-Design%20Phase-yellow)](docs/design/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A sophisticated LangChain-based agentic workflow pipeline for systematic algorithmic trading research and development. This system uses state-of-the-art multi-agent orchestration patterns, dynamic quality gates, and vector store memory to iteratively develop and validate profitable trading algorithms.

## Project Status

**Current Phase: Design & Documentation Complete - Ready for Implementation**

See [Implementation Roadmap](docs/IMPLEMENTATION_ROADMAP.md) for next steps.

## Key Features

- **Research Swarm Architecture**: Leader agent orchestrating parallel specialized subagents for comprehensive market research
- **Dynamic Quality Gates**: User-defined, evolving criteria with fuzzy logic scoring
- **Tool Development Meta-System**: Generate and validate metric-calculating tools before R&D
- **Vector Store Memory**: ChromaDB-based persistent memory with lineage tracking
- **Systematic Toolchain Validation**: Comprehensive testing framework for tool reliability
- **Human-in-the-Loop**: Initial criteria definition with optional ongoing interaction

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    HUMAN-IN-THE-LOOP PHASE                      │
│  Define: Passing Criteria, Alpha Direction, Risk Tolerance      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   TOOL DEVELOPMENT PHASE                        │
│  Metric Tool Generator → Validation → Registry                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   RESEARCH SWARM PHASE                          │
│  Leader Agent → Parallel Subagents → Conflict Resolution        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                STRATEGY DEVELOPMENT PHASE                       │
│  Formulation → Code Generation → Validation Pipeline            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              BACKTESTING & OPTIMIZATION PHASE                   │
│  Backtest → Walk-Forward → Monte Carlo                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               QUALITY GATE VALIDATION PHASE                     │
│  Dynamic Criteria → Fuzzy Scoring → Iterate or Approve          │
└─────────────────────────────────────────────────────────────────┘
```

## Technology Stack

| Component | Technology |
|-----------|------------|
| Agent Framework | LangChain + LangGraph |
| LLM Provider | OpenAI-compatible API (gpt-4.1-mini) |
| Vector Store | ChromaDB |
| Market Data | yfinance |
| Backtesting | Backtrader |
| Language | Python 3.11+ |

## Repository Structure

```
research_langchain_algotrade_development_system/
├── .manus/                    # Manus AI session continuity files
│   ├── AGENT_INSTRUCTIONS.md  # Instructions for AI agents
│   ├── CONVERSATION_LOG.md    # Full conversation history
│   └── SESSION_STATE.json     # Current state for session resumption
├── docs/
│   ├── design/               # Design documents
│   ├── architecture/         # Architecture diagrams and specs
│   ├── api/                  # API documentation
│   └── guides/               # User and developer guides
├── src/
│   ├── agents/               # Agent implementations
│   ├── tools/                # Custom tools
│   ├── memory/               # Vector store and memory management
│   ├── quality_gates/        # Quality gate system
│   ├── workflows/            # LangGraph workflows
│   └── utils/                # Utility functions
├── tests/
│   ├── unit/                 # Unit tests
│   ├── integration/          # Integration tests
│   └── e2e/                  # End-to-end tests
├── config/                   # Configuration files
├── examples/                 # Example usage
└── scripts/                  # Utility scripts
```

## Quick Start

> **Note**: Implementation not yet complete. See [Implementation Roadmap](docs/IMPLEMENTATION_ROADMAP.md).

```bash
# Clone the repository
git clone https://github.com/benjamincham/research_langchain_algotrade_development_system.git
cd research_langchain_algotrade_development_system

# Install dependencies (when available)
pip install -r requirements.txt

# Run the system (when implemented)
python -m src.main
```

## Documentation

- [Design Documentation](docs/design/README.md) - Comprehensive system design
- [Architecture](docs/architecture/README.md) - Technical architecture details
- [Implementation Roadmap](docs/IMPLEMENTATION_ROADMAP.md) - Development phases and tasks
- [Decision Log](docs/DECISION_LOG.md) - All design decisions and rationale
- [Conversation History](.manus/CONVERSATION_LOG.md) - Full project conversation

## For AI Agents (Manus Continuity)

This project is designed for seamless continuation across Manus AI sessions. 

**To continue this project in a new session:**

1. Read `.manus/AGENT_INSTRUCTIONS.md` for context and instructions
2. Review `.manus/SESSION_STATE.json` for current state
3. Check `.manus/CONVERSATION_LOG.md` for full history
4. Consult `docs/IMPLEMENTATION_ROADMAP.md` for next tasks

## Contributing

This project is in active development. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Anthropic's multi-agent research system architecture
- OpenAI Swarm framework patterns
- LangChain/LangGraph documentation and best practices
