# Intelligent Failure Analysis System

## Problem Statement

The Quality Gate Validation Phase needs to make sophisticated decisions about **why** a strategy failed and **what** to do next. This cannot be done with simple rule-based logic. The system needs:

1. **Deep reasoning** about failure causes (parameter issue vs. algorithm bug vs. fundamental research flaw)
2. **Statistical assessment** of whether improvements are possible
3. **Context awareness** of iteration history and failure patterns
4. **Actionable recommendations** for the next step

**Key Question**: How does the system know whether to:
- **Tune parameters** (Tier 1: Strategy Refinement)
- **Fix algorithm bugs** (e.g., incorrect RSI calculation)
- **Refine algorithm approach** (e.g., add regime awareness)
- **Go back to research** (Tier 2: Research Refinement)
- **Abandon** (Tier 3: Futility detected)

---

## Solution: LLM-Powered Failure Analysis Agent

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    FAILURE ANALYSIS AGENT                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    STEP 1: DATA COLLECTION                       │   │
│  │  • Strategy code                                                 │   │
│  │  • Backtest metrics                                              │   │
│  │  • Quality gate results                                          │   │
│  │  • Iteration history                                             │   │
│  │  • Research findings                                             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    STEP 2: FAILURE CLASSIFICATION                │   │
│  │  LLM analyzes failure and classifies into:                       │   │
│  │  • Parameter Issue                                               │   │
│  │  • Algorithm Bug                                                 │   │
│  │  • Design Flaw                                                   │   │
│  │  • Research Gap                                                  │   │
│  │  • Fundamental Impossibility                                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    STEP 3: ROOT CAUSE ANALYSIS                   │   │
│  │  LLM performs deep reasoning:                                    │   │
│  │  • Why did the strategy fail?                                    │   │
│  │  • Is the failure fixable?                                       │   │
│  │  • What is the likelihood of success if we iterate?              │   │
│  │  • Are there signs of bugs or calculation errors?               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    STEP 4: STATISTICAL ASSESSMENT                │   │
│  │  LLM evaluates:                                                  │   │
│  │  • Distance from threshold (how close to passing?)               │   │
│  │  • Improvement trajectory (getting better or worse?)             │   │
│  │  • Variance across iterations (consistent or erratic?)           │   │
│  │  • Comparison to research expectations                           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    STEP 5: DECISION RECOMMENDATION               │   │
│  │  LLM recommends next action:                                     │   │
│  │  • TUNE_PARAMETERS (with specific suggestions)                   │   │
│  │  • FIX_BUG (with suspected bug location)                         │   │
│  │  • REFINE_ALGORITHM (with design suggestions)                    │   │
│  │  • REFINE_RESEARCH (with new research directive)                 │   │
│  │  • ABANDON (with futility reasoning)                             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    STEP 6: CONFIDENCE SCORING                    │   │
│  │  LLM assigns confidence to recommendation:                       │   │
│  │  • High (0.8-1.0): Very confident in diagnosis                   │   │
│  │  • Medium (0.5-0.8): Reasonable confidence                       │   │
│  │  • Low (0.0-0.5): Uncertain, may need human review               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Failure Classification Taxonomy

### 1. Parameter Issue

**Definition**: The strategy logic is sound, but parameters need tuning.

**Indicators**:
- Metrics are close to thresholds (within 20%)
- Performance is in the right direction but not strong enough
- No obvious bugs or logic errors
- Research findings support the strategy approach

**Examples**:
- Sharpe ratio is 0.85 (target: 1.0) → Need larger position sizes
- Max drawdown is 22% (target: 20%) → Need tighter stop losses
- Win rate is 48% (target: 50%) → Need better entry timing

**Next Action**: TUNE_PARAMETERS (Tier 1)

---

### 2. Algorithm Bug

**Definition**: The strategy has implementation errors (incorrect calculations, logic bugs).

**Indicators**:
- Metrics are far worse than research expectations
- Suspicious patterns (e.g., all trades losing, extreme drawdown)
- Code review reveals potential errors
- Metrics don't align with strategy logic

**Examples**:
- RSI calculation using wrong period (14 instead of 20)
- Stop loss logic inverted (selling winners, holding losers)
- Position sizing calculation error (using wrong account value)
- Date/time handling bug (trading at wrong times)

**Next Action**: FIX_BUG → then retry (Tier 1)

**LLM Reasoning**:
```
"The strategy shows 100% losing trades despite bullish market conditions.
This is statistically improbable. Code review reveals the entry condition
uses 'RSI < 30' but the research specified 'RSI > 70' for momentum.
This appears to be a logic inversion bug."

Recommendation: FIX_BUG
Confidence: 0.95
Specific Fix: "Change line 45 from 'if rsi < 30' to 'if rsi > 70'"
```

---

### 3. Design Flaw

**Definition**: The strategy logic has fundamental design issues that prevent it from working.

**Indicators**:
- Strategy fails consistently across parameter variations
- Logic doesn't account for important factors (regime, volatility, etc.)
- Strategy is too simple or too complex
- Missing risk management or position sizing logic

**Examples**:
- Momentum strategy doesn't account for market regime (fails in sideways markets)
- Mean reversion strategy lacks volatility filter (fails in trending markets)
- Strategy uses single indicator without confirmation
- No stop loss or take profit logic

**Next Action**: REFINE_ALGORITHM (Tier 1 with major changes) or REFINE_RESEARCH (Tier 2)

**LLM Reasoning**:
```
"The momentum strategy achieves Sharpe 1.2 in bull markets but -0.5 in
sideways markets. The strategy lacks regime awareness. Research findings
mentioned market regime importance, but the strategy doesn't implement it.

This is a design flaw, not a parameter issue. The strategy needs regime
detection and adaptive logic."

Recommendation: REFINE_ALGORITHM
Confidence: 0.85
Specific Fix: "Add regime detection using volatility and trend indicators.
Apply momentum logic only in trending regimes."
```

---

### 4. Research Gap

**Definition**: The research findings are insufficient or incorrect, leading to strategies that cannot work.

**Indicators**:
- Multiple strategy variants fail despite different approaches
- All strategies fail on fundamental metrics (negative returns)
- Research findings conflict with backtest results
- Missing critical research dimensions (regime, sentiment, etc.)

**Examples**:
- Research says "AAPL shows strong momentum" but all momentum strategies fail
- Research lacks regime analysis, strategies fail in certain market conditions
- Research didn't consider transaction costs, strategies fail after costs
- Research based on too short time period, strategies don't generalize

**Next Action**: REFINE_RESEARCH (Tier 2)

**LLM Reasoning**:
```
"Three different momentum strategies (RSI, MACD, Breakout) all failed with
negative Sharpe ratios. Research findings claimed 'AAPL shows strong momentum
in 2023', but backtests show AAPL was range-bound in 2023.

This suggests the research findings are incorrect or incomplete. The research
swarm may have analyzed too short a time period or missed regime transitions."

Recommendation: REFINE_RESEARCH
Confidence: 0.80
New Research Directive: "Re-analyze AAPL for 2023 with focus on regime
detection. Include volatility analysis and trend strength metrics."
```

---

### 5. Fundamental Impossibility

**Definition**: The alpha opportunity doesn't exist; no strategy can succeed in this context.

**Indicators**:
- All strategies have negative returns
- Market conditions make the approach impossible (e.g., long-only in bear market)
- Transaction costs exceed potential profits
- No improvement across multiple research iterations

**Examples**:
- Long-only strategies in strong bear market
- High-frequency strategies with high transaction costs
- Arbitrage opportunities that no longer exist
- Overfitted historical patterns that don't persist

**Next Action**: ABANDON (Tier 3)

**LLM Reasoning**:
```
"After 3 research iterations and 15 strategy attempts, all strategies for
'AAPL long-only in 2022' have negative returns. Market analysis shows AAPL
declined 27% in 2022 in a strong bear market.

Long-only strategies cannot generate positive returns in a strong downtrend.
This is a fundamental impossibility, not a fixable issue."

Recommendation: ABANDON
Confidence: 0.95
Lesson Learned: "Always check market regime before developing directional
strategies. Long-only strategies require bullish or neutral markets."
```

---

## LLM Reasoning Chain

### Prompt Template for Failure Analysis

```python
FAILURE_ANALYSIS_PROMPT = """
You are an expert quantitative trading analyst tasked with diagnosing why a trading strategy failed quality gates.

## Context

**Strategy Code:**
{strategy_code}

**Research Findings:**
{research_findings}

**Backtest Metrics:**
{backtest_metrics}

**Quality Gate Results:**
{quality_gate_results}

**Iteration History:**
{iteration_history}

**Current Iteration:** {current_iteration} / {max_iterations}

## Your Task

Perform a deep analysis to determine:

1. **Failure Classification**: Classify the failure into one of these categories:
   - PARAMETER_ISSUE: Strategy logic is sound, parameters need tuning
   - ALGORITHM_BUG: Implementation error (calculation bug, logic error)
   - DESIGN_FLAW: Strategy design has fundamental issues
   - RESEARCH_GAP: Research findings are insufficient or incorrect
   - FUNDAMENTAL_IMPOSSIBILITY: Alpha opportunity doesn't exist

2. **Root Cause Analysis**: Explain WHY the strategy failed. Be specific.

3. **Statistical Assessment**: Evaluate:
   - How close are metrics to thresholds?
   - Is the strategy improving across iterations?
   - Are metrics consistent or erratic?
   - Do results align with research expectations?

4. **Bug Detection**: Look for potential bugs:
   - Incorrect indicator calculations
   - Logic inversions
   - Off-by-one errors
   - Date/time handling issues
   - Position sizing errors

5. **Improvement Likelihood**: Estimate the probability that iteration will succeed:
   - HIGH (>70%): Very likely to pass with iteration
   - MEDIUM (30-70%): Possible but uncertain
   - LOW (<30%): Unlikely to pass even with iteration

6. **Recommendation**: Choose ONE action:
   - TUNE_PARAMETERS: Adjust parameters (provide specific suggestions)
   - FIX_BUG: Fix implementation bug (identify suspected bug location)
   - REFINE_ALGORITHM: Redesign strategy logic (provide design suggestions)
   - REFINE_RESEARCH: Go back to research (provide new research directive)
   - ABANDON: Give up on this direction (explain why)

7. **Confidence**: Rate your confidence in this diagnosis (0.0 to 1.0)

## Output Format

Provide your analysis in JSON format:

```json
{
  "failure_classification": "PARAMETER_ISSUE | ALGORITHM_BUG | DESIGN_FLAW | RESEARCH_GAP | FUNDAMENTAL_IMPOSSIBILITY",
  "root_cause": "Detailed explanation of why the strategy failed",
  "statistical_assessment": {
    "distance_from_threshold": "How close to passing? (e.g., 'Within 15%')",
    "improvement_trajectory": "Getting better, worse, or flat?",
    "consistency": "Consistent or erratic performance?",
    "alignment_with_research": "Do results match research expectations?"
  },
  "bug_detection": {
    "suspected_bugs": ["List of potential bugs found"],
    "bug_locations": ["Specific code locations or logic areas"]
  },
  "improvement_likelihood": "HIGH | MEDIUM | LOW",
  "recommendation": "TUNE_PARAMETERS | FIX_BUG | REFINE_ALGORITHM | REFINE_RESEARCH | ABANDON",
  "specific_actions": ["Detailed list of specific actions to take"],
  "confidence": 0.85,
  "reasoning": "Step-by-step explanation of your diagnosis"
}
```

## Guidelines

- Be thorough and analytical
- Look for patterns across iterations
- Consider both statistical and logical evidence
- Be honest about uncertainty
- Provide actionable recommendations
- If confidence is low (<0.5), recommend human review
"""
```

---

## Implementation

### FailureAnalysisAgent Class

```python
from pydantic import BaseModel
from typing import Literal, List
from langchain_core.messages import HumanMessage, SystemMessage
from src.core.llm_client import create_powerful_llm

class FailureAnalysis(BaseModel):
    """Result of failure analysis."""
    
    failure_classification: Literal[
        "PARAMETER_ISSUE",
        "ALGORITHM_BUG",
        "DESIGN_FLAW",
        "RESEARCH_GAP",
        "FUNDAMENTAL_IMPOSSIBILITY"
    ]
    root_cause: str
    statistical_assessment: dict
    bug_detection: dict
    improvement_likelihood: Literal["HIGH", "MEDIUM", "LOW"]
    recommendation: Literal[
        "TUNE_PARAMETERS",
        "FIX_BUG",
        "REFINE_ALGORITHM",
        "REFINE_RESEARCH",
        "ABANDON"
    ]
    specific_actions: List[str]
    confidence: float
    reasoning: str


class FailureAnalysisAgent:
    """
    LLM-powered agent that analyzes strategy failures and recommends next actions.
    """
    
    def __init__(self):
        self.llm = create_powerful_llm()  # Use GPT-4 or Claude for deep reasoning
        self.logger = get_logger(__name__)
    
    async def analyze_failure(
        self,
        strategy_code: str,
        research_findings: List[dict],
        backtest_metrics: dict,
        quality_gate_results: dict,
        iteration_history: List[dict],
        current_iteration: int,
        max_iterations: int
    ) -> FailureAnalysis:
        """
        Analyze why a strategy failed and recommend next action.
        
        Args:
            strategy_code: The strategy's Python code
            research_findings: Research findings that led to this strategy
            backtest_metrics: Metrics from backtest
            quality_gate_results: Quality gate evaluation results
            iteration_history: History of previous iterations
            current_iteration: Current iteration number
            max_iterations: Maximum allowed iterations
        
        Returns:
            FailureAnalysis with classification and recommendations
        """
        self.logger.info(f"Analyzing failure for iteration {current_iteration}/{max_iterations}")
        
        # Build prompt
        prompt = FAILURE_ANALYSIS_PROMPT.format(
            strategy_code=strategy_code,
            research_findings=self._format_research_findings(research_findings),
            backtest_metrics=self._format_metrics(backtest_metrics),
            quality_gate_results=self._format_gate_results(quality_gate_results),
            iteration_history=self._format_iteration_history(iteration_history),
            current_iteration=current_iteration,
            max_iterations=max_iterations
        )
        
        # Call LLM with structured output
        messages = [
            SystemMessage(content="You are an expert quantitative trading analyst."),
            HumanMessage(content=prompt)
        ]
        
        response = await self.llm.ainvoke(messages)
        
        # Parse JSON response
        analysis_dict = self._parse_json_response(response.content)
        
        # Validate and create FailureAnalysis object
        analysis = FailureAnalysis(**analysis_dict)
        
        # Log analysis
        self.logger.info(f"Failure classified as: {analysis.failure_classification}")
        self.logger.info(f"Recommendation: {analysis.recommendation}")
        self.logger.info(f"Confidence: {analysis.confidence}")
        
        # If confidence is low, flag for human review
        if analysis.confidence < 0.5:
            self.logger.warning("Low confidence in analysis. Consider human review.")
        
        return analysis
    
    def _format_research_findings(self, findings: List[dict]) -> str:
        """Format research findings for prompt."""
        formatted = []
        for f in findings:
            formatted.append(f"- {f['type']}: {f['content']} (confidence: {f['confidence']})")
        return "\n".join(formatted)
    
    def _format_metrics(self, metrics: dict) -> str:
        """Format backtest metrics for prompt."""
        return "\n".join([f"- {k}: {v}" for k, v in metrics.items()])
    
    def _format_gate_results(self, results: dict) -> str:
        """Format quality gate results for prompt."""
        formatted = [f"Overall Score: {results['overall_score']:.2f}"]
        formatted.append("\nFailed Criteria:")
        for criterion in results['failed_criteria']:
            formatted.append(
                f"- {criterion['name']}: {criterion['actual']:.2f} "
                f"(threshold: {criterion['threshold']:.2f})"
            )
        return "\n".join(formatted)
    
    def _format_iteration_history(self, history: List[dict]) -> str:
        """Format iteration history for prompt."""
        if not history:
            return "No previous iterations"
        
        formatted = []
        for i, iteration in enumerate(history, 1):
            formatted.append(f"\nIteration {i}:")
            formatted.append(f"  Sharpe: {iteration['metrics'].get('sharpe_ratio', 'N/A')}")
            formatted.append(f"  Max DD: {iteration['metrics'].get('max_drawdown', 'N/A')}")
            formatted.append(f"  Action Taken: {iteration.get('action', 'N/A')}")
        return "\n".join(formatted)
    
    def _parse_json_response(self, response: str) -> dict:
        """Parse JSON from LLM response."""
        import json
        import re
        
        # Extract JSON from markdown code blocks if present
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON object directly
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                raise ValueError("Could not find JSON in LLM response")
        
        return json.loads(json_str)
```

---

## Integration with Quality Gate System

### Updated Quality Gate Workflow

```python
async def quality_gate_with_intelligent_analysis(
    strategy: TradingStrategy,
    research_findings: List[dict],
    gate: QualityGate,
    max_iterations: int = 5
) -> tuple[bool, TradingStrategy, List[dict]]:
    """
    Quality gate loop with intelligent failure analysis.
    """
    failure_agent = FailureAnalysisAgent()
    iteration_history = []
    current_strategy = strategy
    
    for iteration in range(max_iterations):
        # Run backtest
        metrics = await run_backtest(current_strategy)
        
        # Evaluate quality gate
        result = await evaluate_quality_gate(metrics, gate)
        
        if result.passed:
            # SUCCESS!
            return True, current_strategy, iteration_history
        
        # FAILED - Perform intelligent analysis
        analysis = await failure_agent.analyze_failure(
            strategy_code=current_strategy.code,
            research_findings=research_findings,
            backtest_metrics=metrics,
            quality_gate_results=result.to_dict(),
            iteration_history=iteration_history,
            current_iteration=iteration + 1,
            max_iterations=max_iterations
        )
        
        # Store iteration
        iteration_history.append({
            "iteration": iteration + 1,
            "metrics": metrics,
            "analysis": analysis.dict(),
            "action": analysis.recommendation
        })
        
        # Route based on recommendation
        if analysis.recommendation == "TUNE_PARAMETERS":
            # Tier 1: Refine parameters
            current_strategy = await refine_strategy_parameters(
                current_strategy,
                analysis.specific_actions
            )
        
        elif analysis.recommendation == "FIX_BUG":
            # Tier 1: Fix bug
            current_strategy = await fix_strategy_bug(
                current_strategy,
                analysis.bug_detection
            )
        
        elif analysis.recommendation == "REFINE_ALGORITHM":
            # Tier 1 (major) or Tier 2: Redesign strategy
            current_strategy = await refine_strategy_algorithm(
                current_strategy,
                analysis.specific_actions
            )
        
        elif analysis.recommendation == "REFINE_RESEARCH":
            # Tier 2: Go back to research
            return False, current_strategy, iteration_history
        
        elif analysis.recommendation == "ABANDON":
            # Tier 3: Give up
            return False, current_strategy, iteration_history
    
    # Max iterations reached
    return False, current_strategy, iteration_history
```

---

## Example Scenarios

### Scenario 1: Parameter Issue

**Input:**
- Strategy: RSI momentum strategy
- Sharpe Ratio: 0.85 (target: 1.0)
- Max Drawdown: 18% (target: 20%)
- Iteration: 2/5

**LLM Analysis:**
```json
{
  "failure_classification": "PARAMETER_ISSUE",
  "root_cause": "Strategy is close to passing. Sharpe ratio is 15% below target, but max drawdown is within limits. The strategy logic appears sound based on research findings.",
  "statistical_assessment": {
    "distance_from_threshold": "Within 15% for Sharpe ratio",
    "improvement_trajectory": "Improved from 0.75 to 0.85 (13% improvement)",
    "consistency": "Consistent performance across iterations",
    "alignment_with_research": "Matches research expectations for momentum strategy"
  },
  "bug_detection": {
    "suspected_bugs": [],
    "bug_locations": []
  },
  "improvement_likelihood": "HIGH",
  "recommendation": "TUNE_PARAMETERS",
  "specific_actions": [
    "Increase position size from 0.1 to 0.15 to boost returns",
    "Tighten RSI entry threshold from 70 to 75 to improve win rate",
    "Add trailing stop at 15% to capture more profits"
  ],
  "confidence": 0.90,
  "reasoning": "The strategy is performing well and showing improvement. Metrics are close to thresholds. Small parameter adjustments should push it over the line."
}
```

**Next Action:** TUNE_PARAMETERS (Tier 1)

---

### Scenario 2: Algorithm Bug

**Input:**
- Strategy: RSI momentum strategy
- Sharpe Ratio: -1.2 (target: 1.0)
- Win Rate: 5% (target: 50%)
- All trades are losses

**LLM Analysis:**
```json
{
  "failure_classification": "ALGORITHM_BUG",
  "root_cause": "Strategy has extremely poor performance with 95% losing trades. This is statistically improbable and suggests a logic error.",
  "statistical_assessment": {
    "distance_from_threshold": "Far from threshold (negative Sharpe)",
    "improvement_trajectory": "N/A (first iteration)",
    "consistency": "N/A",
    "alignment_with_research": "Completely contradicts research expectations"
  },
  "bug_detection": {
    "suspected_bugs": [
      "Entry condition may be inverted (buying when should sell)",
      "RSI calculation may be using wrong period or formula"
    ],
    "bug_locations": [
      "Line 45: 'if rsi < 30' should likely be 'if rsi > 70' for momentum",
      "Line 23: RSI period is 14, research specified 20"
    ]
  },
  "improvement_likelihood": "HIGH",
  "recommendation": "FIX_BUG",
  "specific_actions": [
    "Review entry logic: Change 'if rsi < 30' to 'if rsi > 70' on line 45",
    "Fix RSI period: Change from 14 to 20 on line 23",
    "Add unit test to verify RSI calculation matches expected values"
  ],
  "confidence": 0.95,
  "reasoning": "The 95% loss rate is a clear indicator of a logic bug. Code review reveals entry condition is inverted and RSI period is wrong."
}
```

**Next Action:** FIX_BUG (Tier 1)

---

### Scenario 3: Research Gap

**Input:**
- Strategy: Momentum strategy (3 variants tried)
- All variants have negative Sharpe ratios
- Research claimed "strong momentum in AAPL"
- Iteration: 5/5 (exhausted Tier 1)

**LLM Analysis:**
```json
{
  "failure_classification": "RESEARCH_GAP",
  "root_cause": "Three different momentum strategies all failed despite research claiming 'strong momentum'. Backtest shows AAPL was range-bound in 2023, not trending. Research findings appear incorrect or incomplete.",
  "statistical_assessment": {
    "distance_from_threshold": "Far from threshold (all negative Sharpe)",
    "improvement_trajectory": "No improvement across 5 iterations",
    "consistency": "Consistently poor across all variants",
    "alignment_with_research": "Contradicts research findings"
  },
  "bug_detection": {
    "suspected_bugs": [],
    "bug_locations": []
  },
  "improvement_likelihood": "LOW",
  "recommendation": "REFINE_RESEARCH",
  "specific_actions": [
    "Re-analyze AAPL for 2023 with focus on regime detection",
    "Include volatility analysis to identify range-bound periods",
    "Extend research period to 2020-2023 for better context",
    "Add sentiment analysis to understand why momentum failed"
  ],
  "confidence": 0.80,
  "reasoning": "Multiple strategy variants failed despite different implementations. This suggests the underlying research hypothesis is wrong. Need to go back to research phase."
}
```

**Next Action:** REFINE_RESEARCH (Tier 2)

---

## Summary

### How the System Makes Assessments

1. **LLM Deep Reasoning**: Uses powerful LLM (GPT-4, Claude) to analyze failures with multi-step reasoning

2. **Comprehensive Context**: Provides strategy code, research findings, metrics, iteration history to LLM

3. **Structured Output**: LLM returns structured JSON with classification, root cause, and recommendations

4. **Confidence Scoring**: LLM assigns confidence to its diagnosis; low confidence triggers human review

5. **Actionable Recommendations**: LLM provides specific actions (e.g., "Change line 45 from X to Y")

### Decision Flow

```
Quality Gate FAIL
       ↓
Failure Analysis Agent (LLM)
       ↓
Analyzes: Code + Metrics + History + Research
       ↓
Classifies: Parameter | Bug | Design | Research | Impossible
       ↓
Recommends: Tune | Fix | Refine | Research | Abandon
       ↓
Routes to: Tier 1 | Tier 2 | Tier 3
```

### Key Advantages

- ✅ **Intelligent**: Uses LLM reasoning, not just rules
- ✅ **Context-aware**: Considers full history and research
- ✅ **Bug detection**: Can identify implementation errors
- ✅ **Statistical**: Evaluates improvement likelihood
- ✅ **Actionable**: Provides specific next steps
- ✅ **Confident**: Scores its own confidence
- ✅ **Transparent**: Explains its reasoning

---

**End of Failure Analysis System Design**
