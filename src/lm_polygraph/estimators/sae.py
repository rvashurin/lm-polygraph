import numpy as np

from typing import Dict

from .estimator import Estimator


def _aggregate_token_scores(token_scores: np.ndarray, aggregation: str) -> float:
    if len(token_scores) == 0:
        return 0.0
    if aggregation == "mean":
        return float(np.mean(token_scores))
    if aggregation == "sum":
        return float(np.sum(token_scores))
    if aggregation == "max":
        return float(np.max(token_scores))
    raise ValueError(f"Unsupported SAE token score aggregation: {aggregation}")


def _sparse_token_values(sparse_latents: Dict[str, np.ndarray], token_idx: int):
    token_indices = np.asarray(sparse_latents["token_indices"])
    values = np.asarray(sparse_latents["values"], dtype=np.float64)
    return values[token_indices == token_idx]


def _num_sparse_tokens(sparse_latents: Dict[str, np.ndarray]) -> int:
    return int(np.asarray(sparse_latents["shape"])[0])


def _entropy_from_values(values: np.ndarray, eps: float) -> float:
    values = np.maximum(values, 0.0)
    mass = values.sum()
    if mass <= eps:
        return 0.0
    probabilities = values / mass
    nonzero_mask = probabilities > eps
    return float(
        -np.sum(probabilities[nonzero_mask] * np.log(probabilities[nonzero_mask]))
    )


def _effective_num_features_from_values(values: np.ndarray, eps: float) -> float:
    values = np.maximum(values, 0.0)
    denominator = np.square(values).sum()
    if denominator <= eps:
        return 0.0
    return float(np.square(values.sum()) / denominator)


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


class SAETokenLatentEntropy(Estimator):
    """
    Estimates sequence-level uncertainty by computing SAE latent entropy at each
    generated-token position first, then aggregating token scores.
    """

    def __init__(self, eps: float = 1e-12, token_aggregation: str = "mean"):
        super().__init__(["sae_token_latent_activations"], "sequence")
        self.eps = eps
        self.token_aggregation = token_aggregation

    def __str__(self):
        if self.token_aggregation == "mean":
            return "SAETokenLatentEntropy"
        return f"SAETokenLatentEntropy_{self.token_aggregation}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        result = []
        for sparse_latents in stats["sae_token_latent_activations"]:
            token_scores = np.array(
                [
                    _entropy_from_values(
                        _sparse_token_values(sparse_latents, token_idx),
                        self.eps,
                    )
                    for token_idx in range(_num_sparse_tokens(sparse_latents))
                ],
                dtype=np.float64,
            )
            result.append(_aggregate_token_scores(token_scores, self.token_aggregation))
        return np.array(result)


class SAETokenEffectiveNumFeatures(Estimator):
    """
    Estimates sequence-level uncertainty by computing the effective number of
    active SAE features at each generated-token position first, then aggregating
    token scores.
    """

    def __init__(self, eps: float = 1e-12, token_aggregation: str = "mean"):
        super().__init__(["sae_token_latent_activations"], "sequence")
        self.eps = eps
        self.token_aggregation = token_aggregation

    def __str__(self):
        if self.token_aggregation == "mean":
            return "SAETokenEffectiveNumFeatures"
        return f"SAETokenEffectiveNumFeatures_{self.token_aggregation}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        result = []
        for sparse_latents in stats["sae_token_latent_activations"]:
            token_scores = np.array(
                [
                    _effective_num_features_from_values(
                        _sparse_token_values(sparse_latents, token_idx),
                        self.eps,
                    )
                    for token_idx in range(_num_sparse_tokens(sparse_latents))
                ],
                dtype=np.float64,
            )
            result.append(_aggregate_token_scores(token_scores, self.token_aggregation))
        return np.array(result)
