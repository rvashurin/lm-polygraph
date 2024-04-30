import numpy as np

from typing import Dict

from .estimator import Estimator


class TotalMeanExpertEntropy(Estimator):
    def __init__(self):
        super().__init__(["mixtral_mean_entropies"], "sequence")

    def __str__(self):
        return "TotalMeanExpertEntropy"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        entropies = stats["mixtral_mean_entropies"]

        return entropies


class TotalEntropyOfExpertMean(Estimator):
    def __init__(self):
        super().__init__(["mixtral_entropies_of_mean"], "sequence")

    def __str__(self):
        return "TotalEntropyOfExpertMean"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        entropies = stats["mixtral_entropies_of_mean"]

        return entropies

class FirstTokenMeanExpertEntropy(Estimator):
    def __init__(self):
        super().__init__(["mixtral_last_token_mean_entropies"], "sequence")

    def __str__(self):
        return "FirstTokenMeanExpertEntropy"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        entropies = stats["mixtral_last_token_mean_entropies"]

        return entropies


class FirstTokenEntropyOfExpertMean(Estimator):
    def __init__(self):
        super().__init__(["mixtral_last_token_entropies_of_mean"], "sequence")

    def __str__(self):
        return "FirstTokenEntropyOfExpertMean"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        entropies = stats["mixtral_last_token_entropies_of_mean"]

        return entropies
