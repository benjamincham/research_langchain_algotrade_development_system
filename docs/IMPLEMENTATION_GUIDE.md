# Implementation Guide: Hierarchical Synthesis & Regime-Aware Quality Gates

This guide provides a practical roadmap for implementing the two major design improvements identified in the design critique.

## Overview

Both solutions address critical weaknesses in the current design:

1. **Hierarchical Synthesis**: Prevents the Leader Agent from becoming a synthesis bottleneck by introducing Domain Synthesizers
2. **Regime-Aware Quality Gates**: Makes quality evaluation context-aware by adjusting thresholds based on market conditions

## Implementation Priority

These features should be implemented in the following order:

### Phase 1: Core Infrastructure (Already Started)
- ✅ Basic agent framework
- ✅ LLM client
- ✅ Memory manager
- ⏳ Tool registry

### Phase 2: Enhanced Memory for Regime Tracking
**Before implementing either feature, enhance the memory system:**

1. Add `RegimeCharacteristics` schema to memory
2. Implement `RegimeDetector` as a standalone utility
3. Populate historical regime data for backtesting periods
4. Add `FactSheet` schema to memory

**Estimated Effort**: 2-3 days

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

### Phase 4: Regime-Aware Quality Gates Implementation
**Implement in this order:**

1. **Regime Detection** (2 days)
   - Implement `RegimeDetector`
   - Add trend and volatility calculations
   - Test on historical data

2. **Threshold Adjustment** (1 day)
   - Implement `ThresholdAdjuster`
   - Define `REGIME_ADJUSTMENTS` table
   - Add adjustment logic

3. **Enhanced Fuzzy Evaluator** (1 day)
   - Update `FuzzyEvaluator` to accept regime
   - Add regime-normalized scoring
   - Update tests

4. **Benchmark Comparator** (1 day)
   - Implement `BenchmarkComparator`
   - Add percentile calculations
   - Add relative performance logic

5. **Integration** (1 day)
   - Update Quality Gate workflow
   - Add regime detection step
   - End-to-end testing

**Total Estimated Effort**: 6 days

## Detailed Implementation Steps

### Step 1: Enhance Memory Schema

```python
# Add to src/memory/schemas.py

class RegimeCharacteristics(BaseModel):
    regime: MarketRegime
    trend_direction: float
    trend_strength: float
    volatility: float
    volatility_percentile: float
    correlation_regime: float
    liquidity_score: float
    start_date: date
    end_date: Optional[date]
    historical_sharpe: float
    historical_drawdown: float

class FactSheet(BaseModel):
    domain: str
    timestamp: datetime
    synthesizer_id: str
    key_insights: list[KeyInsight]
    recommendation: str
    confidence: float
    evidence_strength: float
    conflicts_resolved: int
    supporting_findings: list[SubagentFinding]
    evidence_count: int
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

### Step 3: Implement Regime Detector

```python
# Create src/quality_gates/regime_detector.py

import pandas as pd
import numpy as np
from src.memory.memory_manager import get_memory_manager

class RegimeDetector:
    """Detects and classifies market regimes."""
    
    def __init__(self, lookback_days: int = 252):
        self.lookback_days = lookback_days
        self.memory = get_memory_manager()
    
    async def detect_current_regime(
        self, 
        market_data: pd.DataFrame,
        asset_class: str = "equities"
    ) -> RegimeCharacteristics:
        """Detect current market regime."""
        
        # Calculate trend
        trend_direction, trend_strength = self._calculate_trend(market_data)
        
        # Calculate volatility
        volatility = self._calculate_volatility(market_data)
        volatility_percentile = self._calculate_volatility_percentile(
            volatility, 
            market_data
        )
        
        # Classify regime
        regime = self._classify_regime(
            trend_direction, 
            trend_strength, 
            volatility_percentile
        )
        
        # Get historical characteristics
        historical_data = await self._get_historical_regime_data(
            regime, 
            asset_class
        )
        
        return RegimeCharacteristics(
            regime=regime,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            volatility=volatility,
            volatility_percentile=volatility_percentile,
            correlation_regime=self._calculate_correlation(market_data),
            liquidity_score=self._calculate_liquidity(market_data),
            start_date=market_data.index[-self.lookback_days],
            end_date=market_data.index[-1],
            historical_sharpe=historical_data['avg_sharpe'],
            historical_drawdown=historical_data['avg_drawdown']
        )
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

### Step 5: Update Quality Gate Evaluation

```python
# Update src/quality_gates/evaluator.py

async def evaluate_strategy_with_regime_awareness(
    strategy: Strategy,
    backtest_results: dict,
    user_config: UserConfiguration
) -> RegimeAwareGateResult:
    """Evaluate strategy with regime awareness."""
    
    # Detect regime
    regime_detector = RegimeDetector()
    regime = await regime_detector.detect_current_regime(
        market_data=backtest_results['benchmark_data'],
        asset_class=strategy.asset_class
    )
    
    # Adjust thresholds
    threshold_adjuster = ThresholdAdjuster()
    adjusted_criteria = threshold_adjuster.adjust_thresholds(
        user_config.quality_criteria,
        regime
    )
    
    # Evaluate with adjusted thresholds
    evaluator = RegimeAwareFuzzyEvaluator(threshold_adjuster)
    gate_result = await evaluator.evaluate_with_regime(
        strategy_metrics=backtest_results['metrics'],
        base_criteria=user_config.quality_criteria,
        regime=regime
    )
    
    return gate_result
```

## Testing Strategy

### Unit Tests
- Test each component in isolation
- Mock dependencies
- Cover edge cases

### Integration Tests
- Test Tier 1 → Tier 2 → Tier 3 flow
- Test regime detection with historical data
- Test threshold adjustment logic

### End-to-End Tests
- Full research swarm execution
- Full quality gate evaluation
- Verify memory persistence

## Rollout Plan

1. **Week 1-2**: Implement Hierarchical Synthesis
2. **Week 3**: Test and debug Hierarchical Synthesis
3. **Week 4-5**: Implement Regime-Aware Quality Gates
4. **Week 6**: Integration testing and documentation
5. **Week 7**: User acceptance testing

## Success Metrics

### Hierarchical Synthesis
- Leader synthesis time reduced by 50%
- Synthesis accuracy improved (measured by human review)
- Reduced "hallucination" rate in synthesis

### Regime-Aware Quality Gates
- False negative rate reduced by 30%
- False positive rate reduced by 30%
- User satisfaction with quality gate feedback improved

## Next Steps

1. Review this implementation guide
2. Set up development environment
3. Begin with Phase 2: Enhanced Memory
4. Proceed through phases sequentially
5. Document learnings in DECISION_LOG.md
