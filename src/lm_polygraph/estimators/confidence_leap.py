from typing import Dict

import numpy as np

from .estimator import Estimator


def _metric_values(stats: Dict[str, np.ndarray], key: str) -> np.ndarray:
    return np.array(
        [
            float(metrics.get(key, np.nan))
            if isinstance(metrics, dict) and metrics.get(key) is not None
            else np.nan
            for metrics in stats["confidence_leap_metrics"]
        ]
    )


class ConfidenceLeapFinalConfidence(Estimator):
    """
    Uncertainty from the final chunk's top option probability.
    """

    def __init__(self):
        super().__init__(["confidence_leap_metrics"], "sequence")

    def __str__(self):
        return "ConfidenceLeapFinalConfidence"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        return 1.0 - _metric_values(stats, "final_confidence")


class ConfidenceLeapMeanConfidence(Estimator):
    """
    Uncertainty from average top-option probability across reasoning chunks.
    """

    def __init__(self):
        super().__init__(["confidence_leap_metrics"], "sequence")

    def __str__(self):
        return "ConfidenceLeapMeanConfidence"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        return 1.0 - _metric_values(stats, "mean_confidence")


class ConfidenceLeapMaxConfidence(Estimator):
    """
    Uncertainty from the maximum top-option probability reached in the trace.
    """

    def __init__(self):
        super().__init__(["confidence_leap_metrics"], "sequence")

    def __str__(self):
        return "ConfidenceLeapMaxConfidence"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        return 1.0 - _metric_values(stats, "max_confidence")


class ConfidenceLeapMaxJump(Estimator):
    """
    Uncertainty from the largest adjacent probability jump. A larger jump means
    a clearer confidence leap, so the returned uncertainty is inverted.
    """

    def __init__(self):
        super().__init__(["confidence_leap_metrics"], "sequence")

    def __str__(self):
        return "ConfidenceLeapMaxJump"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        values = []
        for metrics in stats["confidence_leap_metrics"]:
            jump = metrics.get("max_jump") if isinstance(metrics, dict) else None
            values.append(1.0 - float(jump["delta"]) if jump is not None else np.nan)
        return np.array(values)


class ConfidenceLeapNumChanges(Estimator):
    """
    Uncertainty from the number of answer changes across chunks.
    """

    def __init__(self):
        super().__init__(["confidence_leap_metrics"], "sequence")

    def __str__(self):
        return "ConfidenceLeapNumChanges"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        return _metric_values(stats, "num_changes")
