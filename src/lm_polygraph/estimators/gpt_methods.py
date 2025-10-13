import numpy as np
from tqdm import tqdm
from typing import Dict
from copy import deepcopy

from .estimator import Estimator
from .common import sample_strategy_to_prefix, best_sample_ids, SAMPLE_SELECTION_STAT_KEYS

from scipy.special import softmax
from scipy.linalg import expm

_EPS = 1e-12


def normalize_probs(probs):
    #norm_probs = np.exp(probs)
    #norm_probs = softmax(probs)
    norm_probs = probs / probs.sum()

    return norm_probs

def _row_stochastic(K: np.ndarray) -> np.ndarray:
    """Row-normalize to make a random-walk matrix; self-loops for zero rows."""
    P = K.copy()
    rs = P.sum(axis=1, keepdims=True)
    mask = (rs <= _EPS)
    P = np.divide(P, np.maximum(rs, _EPS))
    if np.any(mask):
        # put probability 1 on self-loop for isolated rows
        P[mask.ravel(), :] = 0.0
        idxs = np.where(mask.ravel())[0]
        P[idxs, idxs] = 1.0
    return P

def _select_kernel_matrix(kernel_type: str, tau: float, stats: Dict[str, np.ndarray], i: int) -> np.ndarray:
    if kernel_type == "simple":
        K = stats["simple_kernel_matrix"][i]
    elif kernel_type == "heat":
        if abs(tau - 1.0) < 1e-8:
            K = stats["heat_kernel_matrix_t1"][i]
        elif abs(tau - 2.0) < 1e-8:
            K = stats["heat_kernel_matrix_t2"][i]
        elif abs(tau - 5.0) < 1e-8:
            K = stats["heat_kernel_matrix_t5"][i]
        elif abs(tau - 0.5) < 1e-8:
            K = stats["heat_kernel_matrix_t0.5"][i]
        else:
            raise ValueError(f"Unsupported tau={self.tau} for precomputed heat kernel.")
    return K

class GWD(Estimator):
    def __init__(
        self,
        verbose: bool = False,
        kernel: str = "simple",
        tau: float = 1.0,
    ):
        super().__init__(
            ["sample_texts",
             "sample_log_probs", 
             "simple_kernel_matrix",
             "heat_kernel_matrix_t1",
             "heat_kernel_matrix_t2",
             "heat_kernel_matrix_t5",
             "heat_kernel_matrix_t0.5"],
            "sequence"
        )
        self.verbose = verbose
        self.kernel = kernel
        self.tau = tau

    def __str__(self):
        if self.kernel == "simple":
            return f"GWD_simple"
        elif self.kernel == "heat":
            return f"GWD_heat_tau{self.tau}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_sample_log_probs = stats["sample_log_probs"]
        batch_samples = stats["sample_texts"]

        gwd = []
        for i, (sample_texts, sample_log_probs) in enumerate(tqdm(zip(
            batch_samples, batch_sample_log_probs
        ))):
            _, unique_ids = np.unique(sample_texts, return_index=True)

            K = _select_kernel_matrix(self.kernel, self.tau, stats, i)

            probs = np.array(sample_log_probs)[unique_ids]
            probs = normalize_probs(probs)

            gwd_value = np.matmul(
                probs.T,
                np.matmul(K, probs)
            )

            gwd.append(1 - gwd_value)

        return np.array(gwd)


class CDI(Estimator):
    """
    Consensus Diffusion Index (CDI)
    U_CDI(T) = 1 - max_{t=1..T} p^T P^t p,
    where P is the row-stochastic random-walk on the response graph.
    """
    def __init__(self, verbose: bool = False, kernel: str = "simple", tau: float = 1.0, T: int = 4):
        super().__init__(["sample_log_probs", "sample_sentence_similarity", "sample_texts"], "sequence")
        self.verbose = verbose
        self.kernel = kernel
        self.tau = tau
        self.T = int(T)

    def __str__(self):
        base = f"CDI_{self.kernel}"
        if self.kernel == "heat":
            base += f"_tau{self.tau}"
        return base + f"_T{self.T}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_lp = stats["sample_log_probs"]
        batch_sim = stats["sample_sentence_similarity"]
        batch_txt = stats["sample_texts"]

        out = []
        for i, (sample_texts, sample_log_probs, sim) in enumerate(zip(batch_txt, batch_lp, batch_sim)):
            _, unique_ids = np.unique(sample_texts, return_index=True)

            K = _select_kernel_matrix(self.kernel, self.tau, stats, i)

            P = _row_stochastic(K)

            p = np.array(sample_log_probs)[unique_ids]
            p = normalize_probs(p)  # your helper: softmax over log-probs

            # p^T P^t p via iterative matvecs
            v = p.copy()
            best = -np.inf
            for t in range(1, self.T + 1):
                v = P @ v
                F_t = float(p @ v)  # p^T (P^t p)
                if F_t > best:
                    best = F_t
            out.append(1.0 - best)

        return np.array(out)


class GTV_L2(Estimator):
    """
    Graph Total Variation (L2 / Rayleigh) of posterior mass.
    U = (q^T L_sym q) / (q^T q), with q = D^{-1/2} p and L_sym = I - D^{-1/2} K D^{-1/2}.
    Clipped to [0,1] by default for comparability across scores.
    """
    def __init__(self, verbose: bool = False, kernel: str = "simple", tau: float = 1.0, clip01: bool = True):
        super().__init__(["sample_log_probs", "sample_sentence_similarity", "sample_texts"], "sequence")
        self.verbose = verbose
        self.kernel = kernel
        self.tau = tau
        self.clip01 = clip01

    def __str__(self):
        if self.kernel == "simple":
            return f"GTV_L2_simple"
        elif self.kernel == "heat":
            return f"GTV_L2_heat_tau{self.tau}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_lp = stats["sample_log_probs"]
        batch_sim = stats["sample_sentence_similarity"]
        batch_txt = stats["sample_texts"]

        vals = []
        for i, (sample_texts, sample_log_probs, sim) in enumerate(zip(batch_txt, batch_lp, batch_sim)):
            _, unique_ids = np.unique(sample_texts, return_index=True)

            K = _select_kernel_matrix(self.kernel, self.tau, stats, i)

            d = K.sum(axis=1)
            inv_sqrt_d = 1.0 / np.sqrt(np.clip(d, _EPS, None))
            D_inv_sqrt = np.diag(inv_sqrt_d)
            L_sym = np.eye(K.shape[0]) - D_inv_sqrt @ K @ D_inv_sqrt

            p = np.array(sample_log_probs)[unique_ids]
            p = normalize_probs(p)
            q = D_inv_sqrt @ p

            num = float(q @ (L_sym @ q))
            den = float(q @ q) + _EPS
            u = num / den
            #if self.clip01:
            #    u = max(0.0, min(1.0, u))  # L_sym Rayleigh ∈ [0,2]; clipping helps align scales
            vals.append(u)

        return np.array(vals)


class HES(Estimator):
    """
    Heat-kernel Entropy Slope at t=0:
    U = - p^T L_rw log p, with L_rw = I - P and P row-stochastic from the kernel.
    """
    def __init__(self, verbose: bool = False, kernel: str = "simple", tau: float = 1.0, floor: float = 1e-12):
        super().__init__(["sample_log_probs", "sample_sentence_similarity", "sample_texts"], "sequence")
        self.verbose = verbose
        self.kernel = kernel
        self.tau = tau
        self.floor = float(floor)

    def __str__(self):
        if self.kernel == "simple":
            return f"HES_simple"
        elif self.kernel == "heat":
            return f"HES_heat_tau{self.tau}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_lp = stats["sample_log_probs"]
        batch_sim = stats["sample_sentence_similarity"]
        batch_txt = stats["sample_texts"]

        out = []
        for i, (sample_texts, sample_log_probs, sim) in enumerate(zip(batch_txt, batch_lp, batch_sim)):
            _, unique_ids = np.unique(sample_texts, return_index=True)

            K = _select_kernel_matrix(self.kernel, self.tau, stats, i)

            P = _row_stochastic(K)
            L_rw = np.eye(K.shape[0]) - P

            p = np.array(sample_log_probs)[unique_ids]
            p = normalize_probs(p)
            logp = np.log(np.clip(p, self.floor, 1.0))

            u = float(- p @ (L_rw @ logp))
            out.append(max(0.0, u))  # floor at 0 for interpretability

        return np.array(out)


class SimilarityAwareRenyi(Estimator):
    """
    Similarity-aware Hill/Rényi diversity (Leinster–Cobbold style).
    D_q^{(S)}(p) = ( sum_i p_i ( (S p)_i )^{q-1} )^{1/(1-q)} ; U = 1 / D_q^{(S)}.
    q=1 case uses the continuous limit (similarity-aware entropy).
    """
    def __init__(self, verbose: bool = False, kernel: str = "simple", tau: float = 1.0, q: float = 1.0):
        super().__init__(["sample_log_probs", "sample_sentence_similarity", "sample_texts"], "sequence")
        self.verbose = verbose
        self.kernel = kernel
        self.tau = tau
        self.q = float(q)

    def __str__(self):
        name = "RenyiSim"
        if self.kernel == "simple":
            name += "_simple"
        elif self.kernel == "heat":
            name += f"_heat_tau{self.tau}"
        return name + f"_q{self.q}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_lp = stats["sample_log_probs"]
        batch_sim = stats["sample_sentence_similarity"]
        batch_txt = stats["sample_texts"]

        res = []
        for i, (sample_texts, sample_log_probs, sim) in enumerate(zip(batch_txt, batch_lp, batch_sim)):
            _, unique_ids = np.unique(sample_texts, return_index=True)

            S = _select_kernel_matrix(self.kernel, self.tau, stats, i)

            p = np.array(sample_log_probs)[unique_ids]
            p = normalize_probs(p)
            Sp = S @ p
            Sp = np.clip(Sp, _EPS, None)

            if abs(self.q - 1.0) < 1e-8:
                # D1 = exp( - sum_i p_i log (S p)_i )
                D1 = np.exp(- float(np.dot(p, np.log(Sp))))
                U = 1.0 / max(D1, _EPS)
            else:
                t = float(np.sum(p * (Sp ** (self.q - 1.0))))
                Dq = t ** (1.0 / (1.0 - self.q))
                U = 1.0 / max(Dq, _EPS)
            res.append(U)

        return np.array(res)
