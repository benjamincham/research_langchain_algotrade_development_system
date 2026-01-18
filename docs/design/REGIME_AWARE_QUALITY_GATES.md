# Regime-Aware Quality Gates Solution Design

## Problem Statement

The current Quality Gate system uses static thresholds for performance metrics (e.g., Sharpe Ratio ≥ 1.0, Max Drawdown ≤ 20%). This approach has critical flaws:

1. **Context Blindness**: A Sharpe Ratio of 1.0 during a low-volatility bull market is very different from 1.0 during a high-volatility bear market
2. **False Negatives**: Excellent strategies may fail gates simply because market conditions are unfavorable
3. **False Positives**: Mediocre strategies may pass gates due to favorable market tailwinds
4. **Lack of Adaptability**: Thresholds remain fixed regardless of the trading environment

## Proposed Solution: Dynamic Regime-Aware Thresholds

### Core Concept

Quality Gate thresholds should **adapt** based on the current market regime. A strategy's performance is evaluated relative to what is **achievable** in the current regime, not against universal constants.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   REGIME-AWARE QUALITY GATE SYSTEM                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                  REGIME DETECTOR                                 │   │
│  │  • Classify current market regime                                │   │
│  │  • Calculate regime characteristics                              │   │
│  │  • Query historical regime data                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                  THRESHOLD ADJUSTER                              │   │
│  │  • Load base thresholds from user config                         │   │
│  │  • Apply regime-specific adjustments                             │   │
│  │  • Calculate adaptive soft thresholds                            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                  FUZZY EVALUATOR (Enhanced)                      │   │
│  │  • Evaluate metrics against adjusted thresholds                  │   │
│  │  • Apply regime-aware penalty curves                             │   │
│  │  • Calculate regime-normalized scores                            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                  BENCHMARK COMPARATOR                            │   │
│  │  • Compare strategy to regime-appropriate benchmarks             │   │
│  │  • Calculate relative performance                                │   │
│  │  • Assess alpha generation                                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Component 1: Regime Detector

### Market Regime Classification

```python
from enum import Enum
from typing import Optional
import pandas as pd
import numpy as np

class MarketRegime(str, Enum):
    """Market regime classifications."""
    BULL_LOW_VOL = "bull_low_vol"        # Trending up, low volatility
    BULL_HIGH_VOL = "bull_high_vol"      # Trending up, high volatility
    BEAR_LOW_VOL = "bear_low_vol"        # Trending down, low volatility
    BEAR_HIGH_VOL = "bear_high_vol"      # Trending down, high volatility
    SIDEWAYS_LOW_VOL = "sideways_low_vol"  # Range-bound, low volatility
    SIDEWAYS_HIGH_VOL = "sideways_high_vol"  # Range-bound, high volatility
    CRISIS = "crisis"                    # Extreme volatility, sharp moves

class RegimeCharacteristics(BaseModel):
    """Characteristics of a market regime."""
    
    regime: MarketRegime
    
    # Trend
    trend_direction: float  # -1.0 (bear) to +1.0 (bull)
    trend_strength: float   # 0.0 (no trend) to 1.0 (strong trend)
    
    # Volatility
    volatility: float       # Annualized volatility
    volatility_percentile: float  # Historical percentile (0-100)
    
    # Other characteristics
    correlation_regime: float  # Average asset correlation
    liquidity_score: float     # Market liquidity indicator
    
    # Time period
    start_date: date
    end_date: Optional[date]
    
    # Historical context
    historical_sharpe: float   # Typical Sharpe in this regime
    historical_drawdown: float # Typical max drawdown in this regime

class RegimeDetector:
    """Detects and classifies market regimes."""
    
    def __init__(self, lookback_days: int = 252):
        self.lookback_days = lookback_days
    
    async def detect_current_regime(
        self, 
        market_data: pd.DataFrame,
        asset_class: str = "equities"
    ) -> RegimeCharacteristics:
        """
        Detect the current market regime.
        
        Args:
            market_data: OHLCV data for benchmark (e.g., SPY)
            asset_class: Asset class being traded
            
        Returns:
            RegimeCharacteristics for current regime
        """
        # Calculate trend
        trend_direction, trend_strength = self._calculate_trend(market_data)
        
        # Calculate volatility
        volatility = self._calculate_volatility(market_data)
        volatility_percentile = self._calculate_percentile(
            volatility, 
            market_data['returns'].rolling(self.lookback_days).std() * np.sqrt(252)
        )
        
        # Classify regime
        regime = self._classify_regime(
            trend_direction, 
            trend_strength, 
            volatility_percentile
        )
        
        # Get historical characteristics for this regime
        historical_data = await self._get_historical_regime_data(regime, asset_class)
        
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
    
    def _calculate_trend(self, data: pd.DataFrame) -> tuple[float, float]:
        """Calculate trend direction and strength."""
        # Use linear regression on log prices
        log_prices = np.log(data['close'])
        x = np.arange(len(log_prices))
        slope, intercept = np.polyfit(x, log_prices, 1)
        
        # Direction: sign of slope
        direction = np.tanh(slope * 100)  # Normalize to [-1, 1]
        
        # Strength: R-squared of regression
        y_pred = slope * x + intercept
        ss_res = np.sum((log_prices - y_pred) ** 2)
        ss_tot = np.sum((log_prices - np.mean(log_prices)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        strength = r_squared
        
        return direction, strength
    
    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """Calculate annualized volatility."""
        returns = data['close'].pct_change().dropna()
        return returns.std() * np.sqrt(252)
    
    def _classify_regime(
        self, 
        trend_direction: float, 
        trend_strength: float,
        volatility_percentile: float
    ) -> MarketRegime:
        """Classify regime based on trend and volatility."""
        # Determine volatility level
        high_vol = volatility_percentile > 70
        
        # Determine trend
        if trend_strength < 0.3:
            # Sideways
            return MarketRegime.SIDEWAYS_HIGH_VOL if high_vol else MarketRegime.SIDEWAYS_LOW_VOL
        elif trend_direction > 0.2:
            # Bull
            return MarketRegime.BULL_HIGH_VOL if high_vol else MarketRegime.BULL_LOW_VOL
        elif trend_direction < -0.2:
            # Bear
            return MarketRegime.BEAR_HIGH_VOL if high_vol else MarketRegime.BEAR_LOW_VOL
        else:
            # Weak trend, treat as sideways
            return MarketRegime.SIDEWAYS_HIGH_VOL if high_vol else MarketRegime.SIDEWAYS_LOW_VOL
```

## Component 2: Threshold Adjuster

### Regime-Based Threshold Adjustment

```python
class ThresholdAdjuster:
    """Adjusts quality gate thresholds based on market regime."""
    
    # Adjustment factors for each regime (multipliers)
    REGIME_ADJUSTMENTS = {
        MarketRegime.BULL_LOW_VOL: {
            "sharpe_ratio": 1.2,      # Expect higher Sharpe in favorable conditions
            "max_drawdown": 0.8,      # Stricter drawdown limits
            "win_rate": 1.1,
            "profit_factor": 1.1
        },
        MarketRegime.BULL_HIGH_VOL: {
            "sharpe_ratio": 1.0,      # Baseline expectations
            "max_drawdown": 1.0,
            "win_rate": 1.0,
            "profit_factor": 1.0
        },
        MarketRegime.BEAR_LOW_VOL: {
            "sharpe_ratio": 0.7,      # Lower expectations in bear markets
            "max_drawdown": 1.3,      # More lenient drawdown
            "win_rate": 0.9,
            "profit_factor": 0.9
        },
        MarketRegime.BEAR_HIGH_VOL: {
            "sharpe_ratio": 0.5,      # Significantly lower expectations
            "max_drawdown": 1.5,      # Much more lenient
            "win_rate": 0.8,
            "profit_factor": 0.8
        },
        MarketRegime.SIDEWAYS_LOW_VOL: {
            "sharpe_ratio": 0.8,
            "max_drawdown": 1.0,
            "win_rate": 1.0,
            "profit_factor": 1.0
        },
        MarketRegime.SIDEWAYS_HIGH_VOL: {
            "sharpe_ratio": 0.6,
            "max_drawdown": 1.2,
            "win_rate": 0.9,
            "profit_factor": 0.9
        },
        MarketRegime.CRISIS: {
            "sharpe_ratio": 0.3,      # Just survive
            "max_drawdown": 2.0,      # Very lenient
            "win_rate": 0.7,
            "profit_factor": 0.7
        }
    }
    
    def adjust_thresholds(
        self,
        base_criteria: list[Criterion],
        regime: RegimeCharacteristics
    ) -> list[Criterion]:
        """
        Adjust thresholds based on regime.
        
        Args:
            base_criteria: User-defined base criteria
            regime: Current market regime characteristics
            
        Returns:
            Adjusted criteria with regime-aware thresholds
        """
        adjusted = []
        adjustments = self.REGIME_ADJUSTMENTS[regime.regime]
        
        for criterion in base_criteria:
            adjusted_criterion = criterion.copy(deep=True)
            
            # Apply adjustment if available
            if criterion.metric in adjustments:
                adjustment_factor = adjustments[criterion.metric]
                
                # Adjust threshold
                if criterion.operator in [">=", ">"]:
                    # For "greater than" criteria, multiply threshold
                    adjusted_criterion.threshold *= adjustment_factor
                elif criterion.operator in ["<=", "<"]:
                    # For "less than" criteria, multiply threshold
                    adjusted_criterion.threshold *= adjustment_factor
                
                # Adjust soft threshold proportionally
                if criterion.soft_threshold:
                    adjusted_criterion.soft_threshold *= adjustment_factor
                
                # Add regime metadata
                adjusted_criterion.metadata = {
                    "regime": regime.regime,
                    "base_threshold": criterion.threshold,
                    "adjustment_factor": adjustment_factor,
                    "regime_adjusted": True
                }
            
            adjusted.append(adjusted_criterion)
        
        return adjusted
    
    def calculate_regime_normalized_score(
        self,
        raw_score: float,
        metric: str,
        regime: RegimeCharacteristics
    ) -> float:
        """
        Normalize score relative to regime expectations.
        
        A score of 0.7 in a crisis regime might be equivalent to 0.9 in a bull regime.
        """
        adjustments = self.REGIME_ADJUSTMENTS[regime.regime]
        
        if metric not in adjustments:
            return raw_score
        
        adjustment_factor = adjustments[metric]
        
        # Inverse adjustment for normalization
        # If we lowered expectations (factor < 1), boost the score
        # If we raised expectations (factor > 1), reduce the score
        normalized_score = raw_score / adjustment_factor
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, normalized_score))
```

## Component 3: Enhanced Fuzzy Evaluator

```python
class RegimeAwareFuzzyEvaluator:
    """Fuzzy evaluator with regime awareness."""
    
    def __init__(self, threshold_adjuster: ThresholdAdjuster):
        self.threshold_adjuster = threshold_adjuster
    
    async def evaluate_with_regime(
        self,
        strategy_metrics: dict[str, float],
        base_criteria: list[Criterion],
        regime: RegimeCharacteristics
    ) -> RegimeAwareGateResult:
        """
        Evaluate strategy against regime-adjusted criteria.
        """
        # Step 1: Adjust thresholds for regime
        adjusted_criteria = self.threshold_adjuster.adjust_thresholds(
            base_criteria, 
            regime
        )
        
        # Step 2: Evaluate each criterion
        criterion_results = []
        for criterion in adjusted_criteria:
            metric_value = strategy_metrics.get(criterion.metric)
            
            if metric_value is None:
                continue
            
            # Calculate fuzzy score with adjusted thresholds
            score = self._fuzzy_score(
                value=metric_value,
                threshold=criterion.threshold,
                soft_threshold=criterion.soft_threshold,
                operator=criterion.operator,
                penalty_curve=criterion.penalty_curve
            )
            
            # Calculate regime-normalized score
            normalized_score = self.threshold_adjuster.calculate_regime_normalized_score(
                score, 
                criterion.metric, 
                regime
            )
            
            criterion_results.append(CriterionResult(
                criterion=criterion,
                raw_value=metric_value,
                score=score,
                normalized_score=normalized_score,
                passed=score >= 0.5
            ))
        
        # Step 3: Aggregate scores
        overall_score = self._aggregate_scores(criterion_results)
        normalized_overall = self._aggregate_scores(
            criterion_results, 
            use_normalized=True
        )
        
        return RegimeAwareGateResult(
            regime=regime,
            criterion_results=criterion_results,
            overall_score=overall_score,
            normalized_overall_score=normalized_overall,
            passed=overall_score >= 0.7,
            regime_adjusted=True
        )
```

## Component 4: Benchmark Comparator

```python
class BenchmarkComparator:
    """Compare strategy to regime-appropriate benchmarks."""
    
    async def compare_to_regime_benchmark(
        self,
        strategy_metrics: dict[str, float],
        regime: RegimeCharacteristics,
        asset_class: str
    ) -> BenchmarkComparison:
        """
        Compare strategy performance to what's typical in this regime.
        """
        # Get historical benchmark data for this regime
        benchmark_data = await self._get_regime_benchmark(regime, asset_class)
        
        # Calculate relative performance
        relative_sharpe = (
            strategy_metrics['sharpe_ratio'] / benchmark_data['median_sharpe']
        )
        relative_drawdown = (
            strategy_metrics['max_drawdown'] / benchmark_data['median_drawdown']
        )
        
        # Calculate percentile rankings
        sharpe_percentile = self._calculate_percentile(
            strategy_metrics['sharpe_ratio'],
            benchmark_data['sharpe_distribution']
        )
        
        return BenchmarkComparison(
            regime=regime,
            strategy_sharpe=strategy_metrics['sharpe_ratio'],
            benchmark_median_sharpe=benchmark_data['median_sharpe'],
            relative_sharpe=relative_sharpe,
            sharpe_percentile=sharpe_percentile,
            outperforms_benchmark=relative_sharpe > 1.0,
            assessment=self._generate_assessment(relative_sharpe, sharpe_percentile)
        )
```

## Integration with Quality Gate System

```python
async def evaluate_strategy_with_regime_awareness(
    strategy: Strategy,
    backtest_results: dict,
    user_config: UserConfiguration
) -> RegimeAwareGateResult:
    """Full regime-aware quality gate evaluation."""
    
    # Step 1: Detect current regime
    regime_detector = RegimeDetector()
    regime = await regime_detector.detect_current_regime(
        market_data=backtest_results['benchmark_data'],
        asset_class=strategy.asset_class
    )
    
    # Step 2: Evaluate with regime-adjusted thresholds
    evaluator = RegimeAwareFuzzyEvaluator(ThresholdAdjuster())
    gate_result = await evaluator.evaluate_with_regime(
        strategy_metrics=backtest_results['metrics'],
        base_criteria=user_config.quality_criteria,
        regime=regime
    )
    
    # Step 3: Compare to regime benchmark
    comparator = BenchmarkComparator()
    benchmark_comparison = await comparator.compare_to_regime_benchmark(
        strategy_metrics=backtest_results['metrics'],
        regime=regime,
        asset_class=strategy.asset_class
    )
    
    # Step 4: Generate regime-aware feedback
    feedback = generate_regime_aware_feedback(
        gate_result, 
        benchmark_comparison, 
        regime
    )
    
    return gate_result
```

## Benefits

1. **Context-Aware Evaluation**: Strategies are judged relative to market conditions
2. **Reduced False Negatives**: Good strategies aren't penalized for bad market conditions
3. **Reduced False Positives**: Lucky strategies in favorable conditions are held to higher standards
4. **Better Feedback**: Users understand *why* a strategy passed or failed relative to the regime
5. **Historical Anchoring**: Thresholds are grounded in historical regime performance

## Implementation Checklist

- [ ] Implement `RegimeDetector` with trend and volatility analysis
- [ ] Define `REGIME_ADJUSTMENTS` table with empirically-derived factors
- [ ] Implement `ThresholdAdjuster` with regime-based logic
- [ ] Update `FuzzyEvaluator` to support regime-adjusted thresholds
- [ ] Implement `BenchmarkComparator` for relative performance
- [ ] Update `market_regimes` ChromaDB collection schema
- [ ] Populate historical regime data for backtesting
- [ ] Add unit tests for regime detection
- [ ] Add integration tests for full regime-aware evaluation
