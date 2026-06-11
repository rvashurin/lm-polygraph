import numpy as np

from typing import Dict

from .estimator import Estimator


class SAELatentEntropy(Estimator):
    """
    Estimates sequence-level uncertainty using entropy of aggregated SAE latent
    activations.
    """

    def __init__(self, eps: float = 1e-12):
        super().__init__(["sae_latent_activations"], "sequence")
        self.eps = eps

    def __str__(self):
        return "SAELatentEntropy"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        activations = np.asarray(stats["sae_latent_activations"], dtype=np.float64)
        activations = np.maximum(activations, 0.0)

        activation_mass = activations.sum(axis=-1, keepdims=True)
        probabilities = np.divide(
            activations,
            activation_mass,
            out=np.zeros_like(activations),
            where=activation_mass > self.eps,
        )
        entropy_terms = np.zeros_like(probabilities)
        nonzero_mask = probabilities > self.eps
        entropy_terms[nonzero_mask] = (
            probabilities[nonzero_mask] * np.log(probabilities[nonzero_mask])
        )
        return -np.sum(entropy_terms, axis=-1)


class SAEEffectiveNumFeatures(Estimator):
    """
    Estimates sequence-level uncertainty using the effective number of active SAE
    features: (sum_i z_i)^2 / sum_i z_i^2.
    """

    def __init__(self, eps: float = 1e-12):
        super().__init__(["sae_latent_activations"], "sequence")
        self.eps = eps

    def __str__(self):
        return "SAEEffectiveNumFeatures"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        activations = np.asarray(stats["sae_latent_activations"], dtype=np.float64)
        activations = np.maximum(activations, 0.0)

        numerator = np.square(activations.sum(axis=-1))
        denominator = np.square(activations).sum(axis=-1)
        return np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > self.eps,
        )
