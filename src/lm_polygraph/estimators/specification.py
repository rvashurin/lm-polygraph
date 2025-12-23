import numpy as np

from typing import List, Dict, Optional

from .estimator import Estimator

from lm_polygraph.utils.estimate_uncertainty import estimate_uncertainty
from lm_polygraph.estimators import SemanticEntropy
from lm_polygraph.utils.openai_chat import OpenAIChat

class SpecificationUncertainty(Estimator):
    def __init__(
        self, verbose: bool = False, n_clarifications: int = 3
    ):
        deps = ["clarified_entropies", "original_entropy"]
        super().__init__(deps, "sequence")
        self.verbose = verbose
        self.n_clarifications = n_clarifications

    def __str__(self):
        return f"SpecificationUncertainty_n{self.n_clarifications}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        spec_uncertainties = []

        for i, original_entropy in enumerate(stats["original_entropy"]):
            clarified_entropies = stats["clarified_entropies"][i]
            spec_uncertainty = original_entropy - np.mean(clarified_entropies)
            spec_uncertainties.append(spec_uncertainty)

        return np.array(spec_uncertainties)
