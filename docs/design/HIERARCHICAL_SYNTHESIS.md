# Hierarchical Synthesis Solution Design

## Problem Statement

The current Research Swarm architecture relies on a single Leader Agent to synthesize findings from 5+ specialized subagents operating in parallel. This creates several critical issues:

1. **Cognitive Overload**: The Leader must process and reconcile complex, potentially contradictory findings from multiple domains (technical, fundamental, sentiment, etc.)
2. **Information Loss**: Critical nuances from individual subagents may be lost during synthesis
3. **Hallucination Risk**: The LLM may "fill in gaps" or create connections that don't exist when overwhelmed with information
4. **Scalability Bottleneck**: Adding more subagents exacerbates the synthesis complexity

## Proposed Solution: Three-Tier Hierarchical Synthesis

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        TIER 3: LEADER AGENT                             │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  • Receives pre-synthesized Fact Sheets from Tier 2               │ │
│  │  • Performs high-level strategic synthesis                        │ │
│  │  • Identifies cross-domain patterns                               │ │
│  │  • Generates research hypotheses                                  │ │
│  └───────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                    ▲
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
┌─────────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐
│  TIER 2: DOMAIN         │ │  TIER 2: DOMAIN     │ │  TIER 2: DOMAIN     │
│  SYNTHESIZER            │ │  SYNTHESIZER        │ │  SYNTHESIZER        │
│  (Technical)            │ │  (Fundamental)      │ │  (Sentiment)        │
│  ─────────────────      │ │  ─────────────────  │ │  ─────────────────  │
│  • Aggregates findings  │ │  • Aggregates       │ │  • Aggregates       │
│  • Resolves conflicts   │ │    findings         │ │    findings         │
│  • Creates Fact Sheet   │ │  • Resolves         │ │  • Resolves         │
│                         │ │    conflicts        │ │    conflicts        │
│                         │ │  • Creates Fact     │ │  • Creates Fact     │
│                         │ │    Sheet            │ │    Sheet            │
└─────────────────────────┘ └─────────────────────┘ └─────────────────────┘
          ▲                           ▲                         ▲
          │                           │                         │
    ┌─────┴─────┐             ┌──────┴──────┐          ┌──────┴──────┐
    │           │             │             │          │             │
┌─────────┐ ┌─────────┐  ┌─────────┐ ┌─────────┐  ┌─────────┐ ┌─────────┐
│ TIER 1: │ │ TIER 1: │  │ TIER 1: │ │ TIER 1: │  │ TIER 1: │ │ TIER 1: │
│ Tech    │ │ Tech    │  │ Fund    │ │ Fund    │  │ Sent    │ │ Pattern │
│ Sub 1   │ │ Sub 2   │  │ Sub 1   │ │ Sub 2   │  │ Sub 1   │ │ Mining  │
└─────────┘ └─────────┘  └─────────┘ └─────────┘  └─────────┘ └─────────┘
```

### Tier 1: Specialized Subagents (Unchanged)

These remain as currently designed, but with enhanced output structure:

```python
class SubagentFinding(BaseModel):
    """Standardized output from Tier 1 subagents."""
    
    # Identification
    subagent_id: str
    subagent_type: str  # "technical", "fundamental", "sentiment", etc.
    timestamp: datetime
    
    # Core Finding
    finding: str  # Natural language description
    finding_type: Literal["signal", "pattern", "metric", "insight"]
    
    # Confidence and Evidence
    confidence: float  # 0.0 to 1.0
    evidence: list[Evidence]  # Supporting data points
    sources: list[str]  # Data sources used
    
    # Context
    asset_class: str
    time_horizon: str
    market_regime: Optional[str]
    
    # Relationships
    supports: list[str]  # IDs of findings this supports
    contradicts: list[str]  # IDs of findings this contradicts
    
    # Metadata
    tool_calls: int
    execution_time: float

class Evidence(BaseModel):
    """A single piece of supporting evidence."""
    data_point: str
    value: Any
    source: str
    timestamp: datetime
    reliability: float
```

### Tier 2: Domain Synthesizers (NEW)

**Purpose**: Aggregate findings within a single domain (e.g., all technical analysis findings) before passing to the Leader.

#### Domain Synthesizer Agent Specification

```python
class DomainSynthesizer:
    """Synthesizes findings within a single domain."""
    
    def __init__(self, domain: str, llm):
        self.domain = domain  # "technical", "fundamental", "sentiment"
        self.llm = llm
        self.conflict_resolver = IntraDomainConflictResolver()
    
    async def synthesize(
        self, 
        subagent_findings: list[SubagentFinding]
    ) -> FactSheet:
        """
        Synthesize multiple subagent findings into a single Fact Sheet.
        
        Steps:
        1. Group findings by topic/asset
        2. Detect and resolve intra-domain conflicts
        3. Calculate aggregate confidence scores
        4. Extract key insights
        5. Generate standardized Fact Sheet
        """
        # Step 1: Group findings
        grouped = self._group_findings(subagent_findings)
        
        # Step 2: Resolve conflicts within domain
        resolved = await self.conflict_resolver.resolve(grouped)
        
        # Step 3: Calculate aggregate metrics
        aggregate_confidence = self._calculate_aggregate_confidence(resolved)
        
        # Step 4: Extract key insights
        key_insights = await self._extract_insights(resolved)
        
        # Step 5: Generate Fact Sheet
        return FactSheet(
            domain=self.domain,
            timestamp=datetime.now(),
            key_insights=key_insights,
            confidence=aggregate_confidence,
            supporting_findings=resolved,
            conflicts_resolved=len(grouped) - len(resolved),
            evidence_count=sum(len(f.evidence) for f in subagent_findings),
            recommendation=await self._generate_recommendation(key_insights)
        )
```

#### Fact Sheet Schema

```python
class FactSheet(BaseModel):
    """Standardized output from Domain Synthesizers."""
    
    # Identification
    domain: str
    timestamp: datetime
    synthesizer_id: str
    
    # Core Content
    key_insights: list[KeyInsight]  # Top 3-5 insights
    recommendation: str  # Domain-specific recommendation
    
    # Confidence and Quality
    confidence: float  # Aggregate confidence
    evidence_strength: float  # 0.0 to 1.0
    conflicts_resolved: int
    
    # Supporting Data
    supporting_findings: list[SubagentFinding]
    evidence_count: int
    
    # Metadata
    subagents_used: list[str]
    execution_time: float

class KeyInsight(BaseModel):
    """A single key insight from domain synthesis."""
    
    insight: str  # Natural language
    insight_type: Literal["bullish", "bearish", "neutral", "conditional"]
    confidence: float
    supporting_findings: list[str]  # Finding IDs
    conditions: Optional[list[str]]  # Conditions for conditional insights
```

#### Intra-Domain Conflict Resolution

```python
class IntraDomainConflictResolver:
    """Resolves conflicts within a single domain."""
    
    async def resolve(
        self, 
        grouped_findings: dict[str, list[SubagentFinding]]
    ) -> list[SubagentFinding]:
        """
        Resolve conflicts within grouped findings.
        
        Strategy:
        1. For each group, check for contradictions
        2. If contradictions exist:
           a. Compare confidence scores
           b. Compare evidence strength
           c. Check temporal relevance
           d. Apply weighted voting
        3. Keep highest-confidence finding or create merged finding
        """
        resolved = []
        
        for topic, findings in grouped_findings.items():
            if self._has_contradiction(findings):
                # Resolve using weighted confidence
                resolved_finding = self._weighted_resolution(findings)
                resolved.append(resolved_finding)
            else:
                # No conflict, merge complementary findings
                merged = self._merge_complementary(findings)
                resolved.append(merged)
        
        return resolved
    
    def _weighted_resolution(
        self, 
        findings: list[SubagentFinding]
    ) -> SubagentFinding:
        """Resolve contradictory findings using weighted voting."""
        # Calculate weights based on confidence and evidence
        weights = []
        for f in findings:
            weight = f.confidence * (1 + len(f.evidence) * 0.1)
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Select finding with highest weight
        max_idx = normalized_weights.index(max(normalized_weights))
        winner = findings[max_idx]
        
        # Adjust confidence based on disagreement
        disagreement_penalty = 1 - (max(normalized_weights) - min(normalized_weights))
        winner.confidence *= disagreement_penalty
        
        # Add metadata about resolution
        winner.metadata = {
            "resolved_from": [f.subagent_id for f in findings],
            "disagreement_level": 1 - disagreement_penalty
        }
        
        return winner
```

### Tier 3: Leader Agent (Enhanced)

The Leader Agent now receives pre-synthesized Fact Sheets instead of raw findings:

```python
class EnhancedLeaderAgent:
    """Leader agent with hierarchical synthesis."""
    
    async def synthesize_research(
        self, 
        fact_sheets: list[FactSheet]
    ) -> ResearchSynthesis:
        """
        Perform high-level synthesis across domains.
        
        The Leader now focuses on:
        1. Cross-domain pattern identification
        2. Strategic hypothesis generation
        3. Risk-opportunity assessment
        4. Research iteration decisions
        """
        # Step 1: Cross-domain analysis
        cross_domain_patterns = await self._identify_cross_domain_patterns(
            fact_sheets
        )
        
        # Step 2: Generate hypotheses
        hypotheses = await self._generate_hypotheses(
            fact_sheets, 
            cross_domain_patterns
        )
        
        # Step 3: Assess confidence
        overall_confidence = self._calculate_overall_confidence(fact_sheets)
        
        # Step 4: Decide if more research needed
        needs_more_research = self._evaluate_research_completeness(
            fact_sheets, 
            overall_confidence
        )
        
        return ResearchSynthesis(
            fact_sheets=fact_sheets,
            cross_domain_patterns=cross_domain_patterns,
            hypotheses=hypotheses,
            overall_confidence=overall_confidence,
            needs_more_research=needs_more_research,
            recommended_next_steps=self._generate_next_steps(
                needs_more_research, 
                hypotheses
            )
        )
```

### Implementation Workflow

```python
async def execute_hierarchical_research(
    objective: str,
    user_config: UserConfiguration
) -> ResearchSynthesis:
    """Execute research with hierarchical synthesis."""
    
    # Step 1: Leader develops strategy (unchanged)
    strategy = await leader.develop_strategy(objective, user_config)
    
    # Step 2: Spawn Tier 1 subagents by domain
    technical_subagents = [spawn_subagent(spec) for spec in strategy.technical_specs]
    fundamental_subagents = [spawn_subagent(spec) for spec in strategy.fundamental_specs]
    sentiment_subagents = [spawn_subagent(spec) for spec in strategy.sentiment_specs]
    
    # Step 3: Execute Tier 1 in parallel
    technical_findings = await asyncio.gather(*technical_subagents)
    fundamental_findings = await asyncio.gather(*fundamental_subagents)
    sentiment_findings = await asyncio.gather(*sentiment_subagents)
    
    # Step 4: Tier 2 domain synthesis (parallel)
    technical_synthesizer = DomainSynthesizer("technical", llm)
    fundamental_synthesizer = DomainSynthesizer("fundamental", llm)
    sentiment_synthesizer = DomainSynthesizer("sentiment", llm)
    
    fact_sheets = await asyncio.gather(
        technical_synthesizer.synthesize(technical_findings),
        fundamental_synthesizer.synthesize(fundamental_findings),
        sentiment_synthesizer.synthesize(sentiment_findings)
    )
    
    # Step 5: Tier 3 leader synthesis
    final_synthesis = await leader.synthesize_research(fact_sheets)
    
    # Step 6: Store to memory
    await memory.store_research_synthesis(final_synthesis)
    
    return final_synthesis
```

## Benefits

1. **Reduced Cognitive Load**: Leader processes 3-5 Fact Sheets instead of 15-30 raw findings
2. **Improved Accuracy**: Domain-specific conflicts resolved by domain experts
3. **Better Traceability**: Clear lineage from raw findings → Fact Sheets → Final synthesis
4. **Scalability**: Can add more Tier 1 subagents without overwhelming the Leader
5. **Parallelization**: Tier 2 synthesis happens in parallel across domains

## Implementation Checklist

- [ ] Define `FactSheet` and `KeyInsight` schemas
- [ ] Implement `DomainSynthesizer` class
- [ ] Implement `IntraDomainConflictResolver`
- [ ] Update `ResearchLeaderAgent` to consume Fact Sheets
- [ ] Update memory schema to store Fact Sheets
- [ ] Update Research Swarm workflow to use 3-tier architecture
- [ ] Add unit tests for domain synthesis
- [ ] Add integration tests for full hierarchical flow
