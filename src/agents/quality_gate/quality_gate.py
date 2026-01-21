"""
Quality Gate Agent - Validates strategies against quality criteria.

This module implements the QualityGateAgent responsible for evaluating
backtest results against user-defined quality criteria.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any
from datetime import datetime
from enum import Enum
import math


class CriterionCategory(str, Enum):
    """Category of quality criterion."""
    PERFORMANCE = "performance"
    RISK = "risk"
    STATISTICAL = "statistical"
    CUSTOM = "custom"


class CriterionOperator(str, Enum):
    """Operator for criterion evaluation."""
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    EQUAL = "=="
    NOT_EQUAL = "!="
    GREATER = ">"
    LESS = "<"


class Criterion(BaseModel):
    """A single quality gate criterion."""
    name: str
    description: str
    category: CriterionCategory
    metric: str  # Name of metric to evaluate
    operator: CriterionOperator
    threshold: float

    # Weighting
    weight: float = 1.0
    is_required: bool = False

    # Soft threshold for fuzzy scoring
    soft_threshold: Optional[float] = None


class CriterionResult(BaseModel):
    """Result of evaluating a single criterion."""
    criterion: Criterion
    value: float
    score: float  # 0-1 fuzzy score
    passed: bool
    details: str


class GateResult(BaseModel):
    """Result of evaluating a complete quality gate."""
    passed: bool
    overall_score: float
    criterion_results: List[CriterionResult]
    next_action: Literal["success", "tune", "fix", "refine", "research", "abandon"]
    failure_analysis: Optional["FailureAnalysis"] = None
    trajectory_analysis: Optional["TrajectoryAnalysis"] = None
    feedback: str


class FailureClassification(str, Enum):
    """Classification of failure type."""
    PARAMETER_ISSUE = "PARAMETER_ISSUE"
    ALGORITHM_BUG = "ALGORITHM_BUG"
    DESIGN_FLAW = "DESIGN_FLAW"
    RESEARCH_GAP = "RESEARCH_GAP"
    FUNDAMENTAL_IMPOSSIBILITY = "FUNDAMENTAL_IMPOSSIBILITY"


class FailureAnalysis(BaseModel):
    """Analysis of why a strategy failed."""
    classification: FailureClassification
    reasoning: str
    suggested_action: str
    confidence: float


class TrajectoryStatus(str, Enum):
    """Status of experiment trajectory."""
    CONVERGING = "CONVERGING"
    DIVERGING = "DIVERGING"
    OSCILLATING = "OSCILLATING"
    STAGNANT = "STAGNANT"


class TrajectoryAnalysis(BaseModel):
    """Analysis of experiment trajectory over iterations."""
    status: TrajectoryStatus
    reasoning: str
    recommendation: Literal["CONTINUE", "PIVOT", "ABANDON"]
    confidence: float


class QualityGateAgent:
    """Agent responsible for evaluating strategies against quality criteria."""

    # Standard criteria library
    STANDARD_CRITERIA = {
        "sharpe_ratio": Criterion(
            name="Sharpe Ratio",
            description="Risk-adjusted return measure",
            category=CriterionCategory.PERFORMANCE,
            metric="sharpe_ratio",
            operator=CriterionOperator.GREATER_EQUAL,
            threshold=1.0,
            weight=2.0,
            is_required=True,
            soft_threshold=0.8
        ),
        "max_drawdown": Criterion(
            name="Maximum Drawdown",
            description="Maximum peak-to-trough decline",
            category=CriterionCategory.RISK,
            metric="max_drawdown",
            operator=CriterionOperator.LESS_EQUAL,
            threshold=0.15,
            weight=2.0,
            is_required=True,
            soft_threshold=0.20
        ),
        "win_rate": Criterion(
            name="Win Rate",
            description="Percentage of winning trades",
            category=CriterionCategory.PERFORMANCE,
            metric="win_rate",
            operator=CriterionOperator.GREATER_EQUAL,
            threshold=0.50,
            weight=1.0,
            is_required=False,
            soft_threshold=0.40
        ),
        "profit_factor": Criterion(
            name="Profit Factor",
            description="Gross profit / Gross loss",
            category=CriterionCategory.PERFORMANCE,
            metric="profit_factor",
            operator=CriterionOperator.GREATER_EQUAL,
            threshold=1.5,
            weight=1.5,
            is_required=False,
            soft_threshold=1.2
        ),
        "trade_count": Criterion(
            name="Minimum Trades",
            description="Minimum number of trades for statistical significance",
            category=CriterionCategory.STATISTICAL,
            metric="total_trades",
            operator=CriterionOperator.GREATER_EQUAL,
            threshold=30,
            weight=0.5,
            is_required=False
        )
    }

    def __init__(
        self,
        llm=None,
        memory_manager=None,
        tool_registry=None,
        custom_criteria: Optional[List[Criterion]] = None
    ):
        self.llm = llm
        self.memory_manager = memory_manager
        self.tool_registry = tool_registry

        # Build criteria dictionary
        self.criteria = self.STANDARD_CRITERIA.copy()
        if custom_criteria:
            for criterion in custom_criteria:
                self.criteria[criterion.name] = criterion

    async def run(
        self,
        backtest_results: List[Dict[str, Any]],
        quality_criteria: Dict[str, float],
        experiment_history: List[Dict[str, Any]]
    ) -> GateResult:
        """
        Evaluate backtest results against quality criteria.

        Args:
            backtest_results: List of backtest result dictionaries
            quality_criteria: User-defined quality criteria overrides
            experiment_history: History of previous experiments

        Returns:
            GateResult with evaluation and next action
        """
        # Find best result
        best_result = self._find_best_result(backtest_results)

        if not best_result:
            return self._create_failure_result(
                "No valid backtest results",
                "RESEARCH_GAP"
            )

        # Update criteria with user overrides
        criteria = self._apply_user_criteria(quality_criteria)

        # Evaluate criteria
        criterion_results = self._evaluate_criteria(best_result, criteria)

        # Calculate overall score
        overall_score = self._calculate_overall_score(criterion_results)

        # Determine if passed
        passed = self._determine_passed(criterion_results, overall_score)

        # Generate failure analysis if failed
        failure_analysis = None
        if not passed:
            failure_analysis = await self._analyze_failure(
                best_result,
                criterion_results,
                experiment_history
            )

        # Analyze trajectory
        trajectory_analysis = self._analyze_trajectory(
            experiment_history,
            best_result
        )

        # Determine next action
        next_action = self._determine_next_action(
            passed,
            failure_analysis,
            trajectory_analysis
        )

        # Generate feedback
        feedback = self._generate_feedback(criterion_results, overall_score)

        return GateResult(
            passed=passed,
            overall_score=overall_score,
            criterion_results=criterion_results,
            next_action=next_action,
            failure_analysis=failure_analysis,
            trajectory_analysis=trajectory_analysis,
            feedback=feedback
        )

    def _find_best_result(self, results: List[Dict]) -> Optional[Dict]:
        """Find the best backtest result based on Sharpe ratio."""
        if not results:
            return None

        valid_results = [r for r in results if r.get("sharpe_ratio", 0) > 0]
        if not valid_results:
            return results[0]  # Return first if all failed

        return max(valid_results, key=lambda r: r.get("sharpe_ratio", 0))

    def _apply_user_criteria(self, user_criteria: Dict[str, float]) -> Dict[str, Criterion]:
        """Apply user-defined criteria overrides."""
        criteria = self.criteria.copy()

        for name, threshold in user_criteria.items():
            if name in criteria:
                # Update threshold
                criterion = criteria[name]
                criteria[name] = Criterion(
                    name=criterion.name,
                    description=criterion.description,
                    category=criterion.category,
                    metric=criterion.metric,
                    operator=criterion.operator,
                    threshold=threshold,
                    weight=criterion.weight,
                    is_required=criterion.is_required,
                    soft_threshold=criterion.soft_threshold
                )

        return criteria

    def _evaluate_criteria(
        self,
        result: Dict,
        criteria: Dict[str, Criterion]
    ) -> List[CriterionResult]:
        """Evaluate all criteria against a result."""
        results = []

        for name, criterion in criteria.items():
            # Get metric value from result
            value = result.get(criterion.metric, 0)

            # Evaluate
            passed, score, details = self._evaluate_single(
                value, criterion
            )

            results.append(CriterionResult(
                criterion=criterion,
                value=value,
                score=score,
                passed=passed,
                details=details
            ))

        return results

    def _evaluate_single(
        self,
        value: float,
        criterion: Criterion
    ) -> tuple[bool, float, str]:
        """Evaluate a single criterion."""
        # Apply fuzzy scoring
        score = self._fuzzy_score(value, criterion)

        # Determine passed
        if criterion.operator in [">=", ">"]:
            passed = value >= criterion.threshold
        elif criterion.operator in ["<=", "<"]:
            passed = value <= criterion.threshold
        elif criterion.operator == "==":
            passed = abs(value - criterion.threshold) < 0.01
        else:
            passed = value != criterion.threshold

        # Generate details
        details = f"{criterion.name}: {value:.3f} {'>=' if criterion.operator in ['>=', '>'] else '<=' if criterion.operator in ['<=', '<'] else '=='} {criterion.threshold:.3f} (score: {score:.2f})"

        return passed, score, details

    def _fuzzy_score(self, value: float, criterion: Criterion) -> float:
        """Calculate fuzzy score (0-1) for a criterion."""
        threshold = criterion.threshold
        soft = criterion.soft_threshold

        if criterion.operator in [">=", ">"]:
            if value >= threshold:
                return 1.0
            elif soft and value >= soft:
                # Gradual penalty zone
                range_size = threshold - soft
                distance = threshold - value
                penalty = distance / range_size
                return max(0.5, 1.0 - (penalty * 0.5))
            else:
                # Hard failure zone
                if soft:
                    base = soft
                else:
                    base = threshold * 0.5
                return max(0.0, value / base * 0.5)

        elif criterion.operator in ["<=", "<"]:
            if value <= threshold:
                return 1.0
            elif soft and value <= soft:
                range_size = soft - threshold
                distance = value - threshold
                penalty = distance / range_size
                return max(0.5, 1.0 - (penalty * 0.5))
            else:
                base = threshold * 1.5
                return max(0.0, 1.0 - (value - threshold) / threshold)

        return 1.0 if value == threshold else 0.5

    def _calculate_overall_score(self, results: List[CriterionResult]) -> float:
        """Calculate weighted overall score."""
        if not results:
            return 0.0

        total_weight = sum(r.criterion.weight for r in results)
        if total_weight == 0:
            return 0.0

        weighted_sum = sum(r.score * r.criterion.weight for r in results)
        return weighted_sum / total_weight

    def _determine_passed(
        self,
        results: List[CriterionResult],
        overall_score: float
    ) -> bool:
        """Determine if quality gate is passed."""
        # Check required criteria
        required_failures = sum(
            1 for r in results
            if r.criterion.is_required and not r.passed
        )

        if required_failures > 0:
            return False

        # Check overall score
        if overall_score < 0.7:
            return False

        return True

    async def _analyze_failure(
        self,
        result: Dict,
        criterion_results: List[CriterionResult],
        history: List[Dict]
    ) -> FailureAnalysis:
        """Analyze why the strategy failed."""
        # Find worst criteria
        worst = sorted(
            criterion_results,
            key=lambda r: r.score
        )[:3]

        # Classify failure
        if any(r.score > 0.7 for r in worst):
            classification = FailureClassification.PARAMETER_ISSUE
            reasoning = "Strategy close to passing, needs parameter tuning"
            action = "tune"
        elif any(r.score < 0.3 for r in worst):
            classification = FailureClassification.DESIGN_FLAW
            reasoning = "Strategy fundamentally flawed, needs redesign"
            action = "fix"
        else:
            classification = FailureClassification.RESEARCH_GAP
            reasoning = "Research may have missed key factors"
            action = "research"

        return FailureAnalysis(
            classification=classification,
            reasoning=reasoning,
            suggested_action=action,
            confidence=0.8
        )

    def _analyze_trajectory(
        self,
        history: List[Dict],
        current_result: Dict
    ) -> Optional[TrajectoryAnalysis]:
        """Analyze experiment trajectory."""
        if len(history) < 2:
            return None

        # Extract Sharpe ratios
        sharpe_values = [
            h.get("metrics", {}).get("sharpe_ratio", 0)
            for h in history
        ]
        sharpe_values.append(current_result.get("sharpe_ratio", 0))

        if len(sharpe_values) < 3:
            return None

        # Calculate trend
        recent = sharpe_values[-3:]
        improving = all(recent[i] <= recent[i+1] for i in range(len(recent)-1))
        declining = all(recent[i] >= recent[i+1] for i in range(len(recent)-1))

        if improving:
            status = TrajectoryStatus.CONVERGING
            reasoning = "Metrics improving over recent iterations"
            recommendation = "CONTINUE"
            confidence = 0.85
        elif declining:
            status = TrajectoryStatus.DIVERGING
            reasoning = "Metrics declining, may need different approach"
            recommendation = "PIVOT"
            confidence = 0.80
        elif max(sharpe_values) == sharpe_values[-1] or min(sharpe_values) == sharpe_values[-1]:
            status = TrajectoryStatus.OSCILLATING
            reasoning = "Metrics oscillating around same values"
            recommendation = "CONTINUE"
            confidence = 0.70
        else:
            status = TrajectoryStatus.STAGNANT
            reasoning = "Metrics not improving significantly"
            recommendation = "PIVOT"
            confidence = 0.75

        return TrajectoryAnalysis(
            status=status,
            reasoning=reasoning,
            recommendation=recommendation,
            confidence=confidence
        )

    def _determine_next_action(
        self,
        passed: bool,
        failure_analysis: Optional[FailureAnalysis],
        trajectory_analysis: Optional[TrajectoryAnalysis]
    ) -> str:
        """Determine the next action based on analysis."""
        if passed:
            return "success"

        if not failure_analysis:
            return "research"

        # Use failure classification to determine action
        if failure_analysis.classification == FailureClassification.PARAMETER_ISSUE:
            return "tune"
        elif failure_analysis.classification == FailureClassification.ALGORITHM_BUG:
            return "fix"
        elif failure_analysis.classification == FailureClassification.DESIGN_FLAW:
            return "refine"
        elif failure_analysis.classification == FailureClassification.RESEARCH_GAP:
            return "research"
        else:
            return "abandon"

    def _generate_feedback(
        self,
        results: List[CriterionResult],
        overall_score: float
    ) -> str:
        """Generate human-readable feedback."""
        strengths = [r.criterion.name for r in results if r.score >= 0.8]
        weaknesses = [r.criterion.name for r in results if r.score < 0.7]

        feedback = f"Overall Score: {overall_score:.2%}\n"
        feedback += f"Strengths: {', '.join(strengths) if strengths else 'None'}\n"
        feedback += f"Needs Improvement: {', '.join(weaknesses) if weaknesses else 'None'}"

        return feedback

    def _create_failure_result(
        self,
        reason: str,
        classification: str
    ) -> GateResult:
        """Create a result for catastrophic failure."""
        return GateResult(
            passed=False,
            overall_score=0.0,
            criterion_results=[],
            next_action="abandon",
            failure_analysis=FailureAnalysis(
                classification=FailureClassification(classification),
                reasoning=reason,
                suggested_action="abandon",
                confidence=1.0
            ),
            trajectory_analysis=None,
            feedback=f"Critical failure: {reason}"
        )
