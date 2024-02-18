import numpy as np

from typing import Dict

from .estimator import Estimator


class MeanExpertEntropy(Estimator):
    def __init__(self):
        super().__init__(["mixtral_mean_entropies"], "sequence")

    def __str__(self):
        return "MeanExpertEntropy"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        entropies = stats["mixtral_mean_entropies"]

        return entropies[0]


class EntropyOfExpertMean(Estimator):
    def __init__(self):
        super().__init__(["mixtral_entropies_of_mean"], "sequence")

    def __str__(self):
        return "EntropyOfExpertMean"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        entropies = stats["mixtral_entropies_of_mean"]
        return entropies[0]
