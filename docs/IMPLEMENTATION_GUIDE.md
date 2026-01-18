# Implementation Guide: Hierarchical Synthesis

This guide provides a practical roadmap for implementing the Hierarchical Synthesis design improvement identified in the design critique.

## Overview

The Hierarchical Synthesis solution addresses the critical weakness of synthesis bottleneck in the Research Swarm by introducing Domain Synthesizers as an intermediate processing layer between specialized subagents and the Leader Agent.

## Implementation Priority

### Phase 1: Core Infrastructure (Already Started)
- ✅ Basic agent framework
- ✅ LLM client
- ✅ Memory manager
- ⏳ Tool registry

### Phase 2: Enhanced Memory for Fact Sheets
**Before implementing Hierarchical Synthesis, enhance the memory system:**

1. Add `FactSheet` schema to memory
2. Add `KeyInsight` schema to memory
3. Update memory manager to store and retrieve Fact Sheets

**Estimated Effort**: 1-2 days

### Phase 3: Hierarchical Synthesis Implementation
**Implement in this order:**

1. **Domain Synthesizer Base Class** (1 day)
   - Create `DomainSynthesizer` abstract class
   - Implement `FactSheet` and `KeyInsight` schemas
   - Add unit tests

2. **Intra-Domain Conflict Resolution** (1 day)
   - Implement `IntraDomainConflictResolver`
   - Add weighted voting logic
   - Add tests for conflict scenarios

3. **Domain-Specific Synthesizers** (2 days)
   - Implement `TechnicalSynthesizer`
   - Implement `FundamentalSynthesizer`
   - Implement `SentimentSynthesizer`
   - Add integration tests

4. **Enhanced Leader Agent** (1 day)
   - Update Leader to consume Fact Sheets
   - Implement cross-domain pattern detection
   - Update synthesis logic

5. **Workflow Integration** (1 day)
   - Update Research Swarm workflow
   - Add Tier 2 execution layer
   - End-to-end testing

**Total Estimated Effort**: 6 days

## Detailed Implementation Steps

### Step 1: Enhance Memory Schema

```python
# Add to src/memory/schemas.py

class KeyInsight(BaseModel):
    """A single key insight from domain synthesis."""
    
    insight: str  # Natural language
    insight_type: Literal["bullish", "bearish", "neutral", "conditional"]
    confidence: float
    supporting_findings: list[str]  # Finding IDs
    conditions: Optional[list[str]]  # Conditions for conditional insights

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
```

### Step 2: Implement Domain Synthesizer

```python
# Create src/agents/research/domain_synthesizer.py

from abc import ABC, abstractmethod
from src.core.base_agent import BaseAgent

class DomainSynthesizer(BaseAgent):
    """Base class for domain-specific synthesizers."""
    
    def __init__(self, domain: str, llm):
        super().__init__(
            name=f"{domain.capitalize()}Synthesizer",
            role=f"Domain Synthesizer for {domain}",
            llm=llm
        )
        self.domain = domain
        self.conflict_resolver = IntraDomainConflictResolver()
    
    async def run(self, input_data: dict) -> dict:
        """Synthesize findings from subagents."""
        subagent_findings = input_data['findings']
        
        # Group findings by topic
        grouped = self._group_findings(subagent_findings)
        
        # Resolve conflicts
        resolved = await self.conflict_resolver.resolve(grouped)
        
        # Extract insights
        insights = await self._extract_insights(resolved)
        
        # Generate fact sheet
        fact_sheet = FactSheet(
            domain=self.domain,
            timestamp=datetime.now(),
            synthesizer_id=self.name,
            key_insights=insights,
            recommendation=await self._generate_recommendation(insights),
            confidence=self._calculate_confidence(resolved),
            evidence_strength=self._calculate_evidence_strength(resolved),
            conflicts_resolved=len(grouped) - len(resolved),
            supporting_findings=resolved,
            evidence_count=sum(len(f.evidence) for f in subagent_findings),
            subagents_used=[f.subagent_id for f in subagent_findings],
            execution_time=0.0  # Set by caller
        )
        
        return {"fact_sheet": fact_sheet}
    
    @abstractmethod
    async def _extract_insights(self, findings: list) -> list[KeyInsight]:
        """Extract key insights from findings."""
        pass
    
    @abstractmethod
    async def _generate_recommendation(self, insights: list[KeyInsight]) -> str:
        """Generate domain-specific recommendation."""
        pass
```

### Step 3: Implement Conflict Resolver

```python
# Create src/agents/research/conflict_resolver.py

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

### Step 4: Update Research Swarm Workflow

```python
# Update src/workflows/research_swarm_workflow.py

async def execute_hierarchical_research(
    objective: str,
    user_config: UserConfiguration
) -> ResearchSynthesis:
    """Execute research with hierarchical synthesis."""
    
    # Step 1: Leader develops strategy
    leader = ResearchLeaderAgent(llm)
    strategy = await leader.develop_strategy(objective, user_config)
    
    # Step 2: Spawn Tier 1 subagents by domain
    technical_tasks = [
        spawn_subagent(spec).execute() 
        for spec in strategy.technical_specs
    ]
    fundamental_tasks = [
        spawn_subagent(spec).execute() 
        for spec in strategy.fundamental_specs
    ]
    sentiment_tasks = [
        spawn_subagent(spec).execute() 
        for spec in strategy.sentiment_specs
    ]
    
    # Step 3: Execute Tier 1 in parallel
    technical_findings = await asyncio.gather(*technical_tasks)
    fundamental_findings = await asyncio.gather(*fundamental_tasks)
    sentiment_findings = await asyncio.gather(*sentiment_tasks)
    
    # Step 4: Tier 2 domain synthesis (parallel)
    technical_synthesizer = TechnicalSynthesizer(llm)
    fundamental_synthesizer = FundamentalSynthesizer(llm)
    sentiment_synthesizer = SentimentSynthesizer(llm)
    
    fact_sheets = await asyncio.gather(
        technical_synthesizer.run({'findings': technical_findings}),
        fundamental_synthesizer.run({'findings': fundamental_findings}),
        sentiment_synthesizer.run({'findings': sentiment_findings})
    )
    
    # Step 5: Tier 3 leader synthesis
    final_synthesis = await leader.synthesize_research(
        [fs['fact_sheet'] for fs in fact_sheets]
    )
    
    # Step 6: Store to memory
    await memory.store_research_synthesis(final_synthesis)
    
    return final_synthesis
```

## Testing Strategy

### Unit Tests
- Test each component in isolation
- Mock dependencies
- Cover edge cases

### Integration Tests
- Test Tier 1 → Tier 2 → Tier 3 flow
- Test conflict resolution with various scenarios
- Verify memory persistence

### End-to-End Tests
- Full research swarm execution
- Verify synthesis quality
- Performance benchmarks

## Rollout Plan

1. **Week 1**: Implement Domain Synthesizer base class and schemas
2. **Week 2**: Implement conflict resolution and domain-specific synthesizers
3. **Week 3**: Update Leader Agent and integrate workflow
4. **Week 4**: Testing and documentation

## Success Metrics

### Hierarchical Synthesis
- Leader synthesis time reduced by 50%
- Synthesis accuracy improved (measured by human review)
- Reduced "hallucination" rate in synthesis
- Scales to 15-20 subagents (vs 5-6 before)
- Context window usage reduced by 50%

## Next Steps

1. Review this implementation guide
2. Set up development environment
3. Begin with Phase 2: Enhanced Memory
4. Proceed through Phase 3 sequentially
5. Document learnings in DECISION_LOG.md
