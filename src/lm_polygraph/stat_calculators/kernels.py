import numpy as np

import itertools
from typing import Dict, List
from tqdm import tqdm

from .stat_calculator import StatCalculator
from sentence_transformers import CrossEncoder
from lm_polygraph.utils.model import WhiteboxModel
from scipy.linalg import expm

_EPS = 1e-12


class KernelsCalculator(StatCalculator):
    """
    Calculates the cross-encoder similarity matrix for generation samples using RoBERTa model.
    """

    def __init__(self):
        super().__init__(
            [
                "simple_kernel_matrix",
                "heat_kernel_matrix_t1",
                "heat_kernel_matrix_t2",
                "heat_kernel_matrix_t5",
                "heat_kernel_matrix_t0.5",
            ],
            ["sample_texts", "sample_sentence_similarity"],
        )

    def _symmetrize(self, W: np.ndarray) -> np.ndarray:
        return 0.5 * (W + W.T)

    def _build_kernel(self, W: np.ndarray, kernel: str, tau: float) -> np.ndarray:
        """
        Build a similarity kernel S from raw similarity W.
        - 'simple': scale to [0,1] by max; set diag to 1.
        - 'heat'  : elementwise exp(-tau * L_sym) for compatibility with your GWD.
                    (Matrix exponential would be more "correct", but we mirror your code path.)
        """
        W = self._symmetrize(W)
        if kernel == "simple":
            denom = max(W.max(), _EPS)
            S = W / denom
            np.fill_diagonal(S, 1.0)
            return S
        elif kernel == "heat":
            d = W.sum(axis=1)
            inv_sqrt_d = 1.0 / np.sqrt(np.clip(d, _EPS, None))
            D_inv_sqrt = np.diag(inv_sqrt_d)
            L = np.eye(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt
            S = expm(-tau * L)
            return S
        else:
            raise ValueError(f"Unknown kernel = {kernel}")

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        stats = dependencies
        batch_sample_sentence_similarity = stats["sample_sentence_similarity"]
        batch_samples = stats["sample_texts"]
        
        sim_matrices = {
            "simple_kernel_matrix": [],
            "heat_kernel_matrix_t1": [],
            "heat_kernel_matrix_t2": [],
            "heat_kernel_matrix_t5": [],
            "heat_kernel_matrix_t0.5": [],
        }
        for sample_texts, sample_sentence_similarity in tqdm(zip(batch_samples, batch_sample_sentence_similarity)):
            _, unique_ids = np.unique(sample_texts, return_index=True)
            W = sample_sentence_similarity[unique_ids][:, unique_ids]

            sim_matrices["simple_kernel_matrix"].append(self._build_kernel(W, "simple", tau=1.0))
            sim_matrices["heat_kernel_matrix_t1"].append(self._build_kernel(W, "heat", tau=1.0))
            sim_matrices["heat_kernel_matrix_t2"].append(self._build_kernel(W, "heat", tau=2.0))
            sim_matrices["heat_kernel_matrix_t5"].append(self._build_kernel(W, "heat", tau=5.0))
            sim_matrices["heat_kernel_matrix_t0.5"].append(self._build_kernel(W, "heat", tau=0.5))

        return sim_matrices
