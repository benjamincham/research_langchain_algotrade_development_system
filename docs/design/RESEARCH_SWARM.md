# Research Swarm Architecture Design

## Overview

The Research Swarm is a multi-agent system following the orchestrator-worker pattern, inspired by Anthropic's multi-agent research system. A leader agent coordinates specialized subagents that operate in parallel, each with their own context windows and specialized tools.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        RESEARCH LEADER AGENT                            │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  • Analyze research objective                                      │ │
│  │  • Develop research strategy                                       │ │
│  │  • Spawn subagents with clear task descriptions                   │ │
│  │  • Synthesize results                                              │ │
│  │  • Resolve conflicts                                               │ │
│  │  • Decide if more research needed                                  │ │
│  └───────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐
│  MARKET RESEARCH    │ │ TECHNICAL ANALYSIS  │ │ FUNDAMENTAL ANALYSIS│
│  SUBAGENT           │ │ SUBAGENT            │ │ SUBAGENT            │
│  ─────────────────  │ │ ─────────────────   │ │ ─────────────────   │
│  • Market trends    │ │ • Price patterns    │ │ • Financial metrics │
│  • Sector analysis  │ │ • Indicator signals │ │ • Valuations        │
│  • Economic factors │ │ • Chart patterns    │ │ • Earnings analysis │
└─────────────────────┘ └─────────────────────┘ └─────────────────────┘
                    │               │               │
                    ▼               ▼               ▼
┌─────────────────────┐ ┌─────────────────────┐
│  SENTIMENT ANALYSIS │ │ PATTERN MINING      │
│  SUBAGENT           │ │ SUBAGENT            │
│  ─────────────────  │ │ ─────────────────   │
│  • News sentiment   │ │ • Historical patterns│
│  • Social media     │ │ • Anomaly detection │
│  • Analyst ratings  │ │ • Seasonality       │
└─────────────────────┘ └─────────────────────┘
                    │               │
                    └───────┬───────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     CONFLICT RESOLUTION MODULE                          │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  • Weighted confidence voting                                      │ │
│  │  • Contradiction detection                                         │ │
│  │  • Human escalation for high-conflict scenarios                   │ │
│  └───────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        SYNTHESIS & MEMORY                               │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  • Combine findings into coherent research output                  │ │
│  │  • Store to ChromaDB with embeddings                              │ │
│  │  • Generate hypotheses for strategy development                   │ │
│  └───────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

## Leader Agent Design

### Responsibilities

1. **Strategy Development**: Analyze the research objective and create a plan
2. **Task Decomposition**: Break down research into parallelizable subtasks
3. **Subagent Spawning**: Create subagents with clear, non-overlapping responsibilities
4. **Result Synthesis**: Combine findings from all subagents
5. **Conflict Resolution**: Handle contradictory findings
6. **Iteration Decision**: Determine if more research is needed

### Leader Agent Prompt Structure

```python
LEADER_SYSTEM_PROMPT = """
You are the Research Leader Agent for an algorithmic trading research system.

Your responsibilities:
1. Analyze the research objective provided by the user
2. Develop a comprehensive research strategy
3. Spawn specialized subagents with clear, non-overlapping tasks
4. Synthesize results from subagents into coherent findings
5. Resolve conflicts between contradictory findings
6. Decide if additional research is needed

Guidelines for spawning subagents:
- Each subagent needs: objective, output format, tools to use, clear boundaries
- Avoid task overlap between subagents
- Scale effort to complexity:
  - Simple queries: 1-2 subagents, 3-10 tool calls each
  - Moderate queries: 3-4 subagents, 10-15 tool calls each
  - Complex queries: 5+ subagents, clearly divided responsibilities

For conflict resolution:
- Each subagent reports confidence scores (0-1)
- Use weighted voting for contradictions
- Escalate to human review if confidence spread > 0.5

Always save your research plan to memory before spawning subagents.
"""
```

## Subagent Specifications

### Market Research Subagent

| Attribute | Value |
|-----------|-------|
| **Focus** | Market conditions, trends, sectors |
| **Tools** | web_search, yfinance_market_data, sector_analysis |
| **Output** | Market analysis report with confidence scores |

```python
MARKET_RESEARCH_PROMPT = """
You are a Market Research Subagent specializing in market conditions and trends.

Your task: {task_description}

Focus areas:
- Overall market conditions (bull/bear/sideways)
- Sector performance and rotation
- Economic indicators and their impact
- Market breadth and internals

Output format:
{
    "findings": [...],
    "confidence": 0.0-1.0,
    "sources": [...],
    "contradictions": [...],
    "recommendations": [...]
}

Start with broad searches, then narrow down based on findings.
"""
```

### Technical Analysis Subagent

| Attribute | Value |
|-----------|-------|
| **Focus** | Price patterns, indicators, signals |
| **Tools** | yfinance_data, indicator_calculator, pattern_detector |
| **Output** | Technical signals with confidence scores |

### Fundamental Analysis Subagent

| Attribute | Value |
|-----------|-------|
| **Focus** | Financial metrics, valuations, earnings |
| **Tools** | yfinance_fundamentals, ratio_calculator, earnings_analyzer |
| **Output** | Fundamental scores with confidence |

### Sentiment Analysis Subagent

| Attribute | Value |
|-----------|-------|
| **Focus** | News, social media, analyst ratings |
| **Tools** | web_search, sentiment_analyzer, news_aggregator |
| **Output** | Sentiment indicators with confidence |

### Pattern Mining Subagent

| Attribute | Value |
|-----------|-------|
| **Focus** | Historical patterns, anomalies, seasonality |
| **Tools** | yfinance_historical, pattern_miner, seasonality_detector |
| **Output** | Pattern library with confidence |

## Conflict Resolution

### Weighted Confidence Voting

When subagents produce contradictory findings:

```python
def resolve_conflict(findings: list[SubagentFinding]) -> ResolvedFinding:
    """
    Resolve conflicts using weighted confidence voting.
    
    Args:
        findings: List of findings from subagents
        
    Returns:
        Resolved finding with aggregated confidence
    """
    # Group by topic
    topics = group_by_topic(findings)
    
    resolved = []
    for topic, topic_findings in topics.items():
        if has_contradiction(topic_findings):
            # Calculate weighted vote
            weights = [f.confidence * f.subagent_weight for f in topic_findings]
            total_weight = sum(weights)
            
            # Check if conflict is too high
            confidence_spread = max(f.confidence for f in topic_findings) - \
                               min(f.confidence for f in topic_findings)
            
            if confidence_spread > 0.5:
                # Escalate to human
                resolved.append(HumanEscalation(topic, topic_findings))
            else:
                # Use weighted average
                weighted_finding = weighted_average(topic_findings, weights)
                resolved.append(weighted_finding)
        else:
            # No contradiction, merge findings
            resolved.append(merge_findings(topic_findings))
    
    return ResolvedFinding(resolved)
```

### Contradiction Detection

```python
def has_contradiction(findings: list[Finding]) -> bool:
    """
    Detect if findings contain contradictions.
    
    Examples of contradictions:
    - One says bullish, another says bearish
    - Conflicting price targets
    - Opposing recommendations
    """
    # Extract claims from findings
    claims = [extract_claims(f) for f in findings]
    
    # Check for semantic opposition
    for i, claim_set_1 in enumerate(claims):
        for j, claim_set_2 in enumerate(claims[i+1:], i+1):
            if semantic_opposition(claim_set_1, claim_set_2):
                return True
    
    return False
```

## Swarm Execution Flow

```python
class ResearchSwarm:
    """Research swarm with leader and subagents."""
    
    def __init__(self, llm, memory: ChromaDB):
        self.leader = ResearchLeaderAgent(llm)
        self.memory = memory
        self.conflict_resolver = ConflictResolver()
    
    async def execute_research(
        self, 
        objective: str,
        user_config: UserConfiguration
    ) -> ResearchFindings:
        """
        Execute research swarm for given objective.
        
        Args:
            objective: Research objective from user
            user_config: User configuration with preferences
            
        Returns:
            Synthesized research findings
        """
        # Step 1: Leader develops strategy
        strategy = await self.leader.develop_strategy(
            objective=objective,
            config=user_config,
            memory=self.memory
        )
        
        # Step 2: Save plan to memory
        await self.memory.store_plan(strategy)
        
        # Step 3: Spawn subagents in parallel
        subagent_tasks = []
        for spec in strategy.subagent_specs:
            subagent = self.create_subagent(spec)
            task = subagent.execute(spec.task_description)
            subagent_tasks.append(task)
        
        # Step 4: Gather results
        results = await asyncio.gather(*subagent_tasks, return_exceptions=True)
        
        # Step 5: Handle failures
        successful_results = []
        failed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_results.append((strategy.subagent_specs[i], result))
            else:
                successful_results.append(result)
        
        # Check if enough succeeded
        success_rate = len(successful_results) / len(results)
        if success_rate < 0.5:
            # Retry failed subagents
            retry_results = await self.retry_failed(failed_results)
            successful_results.extend(retry_results)
        
        # Step 6: Resolve conflicts
        resolved = self.conflict_resolver.resolve(successful_results)
        
        # Step 7: Leader synthesizes
        synthesis = await self.leader.synthesize(
            results=resolved,
            objective=objective
        )
        
        # Step 8: Check if more research needed
        if synthesis.needs_more_research:
            # Store partial findings
            await self.memory.store_findings(synthesis.partial_findings)
            
            # Recurse with refined objective
            return await self.execute_research(
                objective=synthesis.refined_objective,
                user_config=user_config
            )
        
        # Step 9: Store final findings
        await self.memory.store_findings(synthesis.findings)
        
        return synthesis.findings
    
    def create_subagent(self, spec: SubagentSpec) -> ResearchSubagent:
        """Create a subagent based on specification."""
        return ResearchSubagent(
            name=spec.name,
            prompt=spec.prompt,
            tools=spec.tools,
            llm=self.llm
        )
```

## Performance Optimization

### Parallel Execution

- Subagents run in parallel using `asyncio.gather`
- Each subagent can make parallel tool calls
- Target: 3-5 subagents with 3+ parallel tool calls each

### Token Efficiency

- Subagents return compressed findings, not raw data
- Leader only receives summaries, not full context
- Memory stores embeddings for efficient retrieval

### Context Management

- Each subagent has independent context window
- Leader saves plan to memory before spawning
- Long research sessions use checkpointing

## Integration with Pipeline

The Research Swarm integrates with the main pipeline:

```python
# In main workflow
async def research_phase(state: PipelineState) -> PipelineState:
    """Execute research phase of pipeline."""
    swarm = ResearchSwarm(llm=get_llm(), memory=get_memory())
    
    findings = await swarm.execute_research(
        objective=state["research_objective"],
        user_config=state["user_config"]
    )
    
    # Generate hypotheses from findings
    hypotheses = await generate_hypotheses(findings)
    
    return {
        **state,
        "research_findings": findings.to_dict(),
        "hypotheses": hypotheses
    }
```
