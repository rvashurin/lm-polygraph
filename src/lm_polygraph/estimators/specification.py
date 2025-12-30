import numpy as np

from typing import List, Dict, Optional

from .estimator import Estimator

class SpecificationUncertaintySemantic(Estimator):
    def __init__(
        self, verbose: bool = False, n_clarifications: int = 5
    ):
        deps = ["clarified_semantic_entropies", "original_semantic_entropy"]
        super().__init__(deps, "sequence")
        self.verbose = verbose
        self.n_clarifications = n_clarifications

    def __str__(self):
        return f"SpecificationUncertaintySemantic_n{self.n_clarifications}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        spec_uncertainties = []

        for i, original_entropy in enumerate(stats["original_semantic_entropy"]):
            clarified_entropies = stats["clarified_semantic_entropies"][i]
            spec_uncertainty = original_entropy - np.mean(clarified_entropies)
            spec_uncertainties.append(spec_uncertainty)

        return np.array(spec_uncertainties)


class SpecificationUncertaintySemanticNormalized(Estimator):
    def __init__(
        self, verbose: bool = False, n_clarifications: int = 5
    ):
        deps = ["clarified_semantic_entropies_normalized", "original_semantic_entropy_normalized"]
        super().__init__(deps, "sequence")
        self.verbose = verbose
        self.n_clarifications = n_clarifications

    def __str__(self):
        return f"SpecificationUncertaintySemanticNormalized_n{self.n_clarifications}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        spec_uncertainties = []

        for i, original_entropy in enumerate(stats["original_semantic_entropy_normalized"]):
            clarified_entropies = stats["clarified_semantic_entropies_normalized"][i]
            spec_uncertainty = original_entropy - np.mean(clarified_entropies)
            spec_uncertainties.append(spec_uncertainty)

        return np.array(spec_uncertainties)


class SpecificationUncertaintySemanticDirect(Estimator):
    def __init__(
        self, verbose: bool = False, n_clarifications: int = 5
    ):
        deps = ["clarified_semantic_entropies_direct", "original_semantic_entropy_direct"]
        super().__init__(deps, "sequence")
        self.verbose = verbose
        self.n_clarifications = n_clarifications
    
        def __str__(self):
            return f"SpecificationUncertaintySemanticDirect_n{self.n_clarifications}"
    
        def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
            spec_uncertainties = []
    
            for i, original_entropy in enumerate(stats["original_semantic_entropy_direct"]):
                clarified_entropies = stats["clarified_semantic_entropies_direct"][i]
                spec_uncertainty = original_entropy - np.mean(clarified_entropies)
                spec_uncertainties.append(spec_uncertainty)
    
            return np.array(spec_uncertainties)


class SpecificationUncertaintySemanticDirectNormalized(Estimator):
    def __init__(
        self, verbose: bool = False, n_clarifications: int = 5
    ):
        deps = ["clarified_semantic_entropies_direct_normalized", "original_semantic_entropy_direct_normalized"]
        super().__init__(deps, "sequence")
        self.verbose = verbose
        self.n_clarifications = n_clarifications

    def __str__(self):
        return f"SpecificationUncertaintySemanticDirectNormalized_n{self.n_clarifications}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        spec_uncertainties = []

        for i, original_entropy in enumerate(stats["original_semantic_entropy_direct_normalized"]):
            clarified_entropies = stats["clarified_semantic_entropies_direct_normalized"][i]
            spec_uncertainty = original_entropy - np.mean(clarified_entropies)
            spec_uncertainties.append(spec_uncertainty)

        return np.array(spec_uncertainties)


class SpecificationUncertaintyMCSE(Estimator):
    def __init__(
        self, verbose: bool = False, n_clarifications: int = 5
    ):
        deps = ["clarified_mcse_entropies", "original_mcse_entropy"]
        super().__init__(deps, "sequence")
        self.verbose = verbose
        self.n_clarifications = n_clarifications

    def __str__(self):
        return f"SpecificationUncertaintyMCSE_n{self.n_clarifications}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        spec_uncertainties = []

        for i, original_entropy in enumerate(stats["original_mcse_entropy"]):
            clarified_entropies = stats["clarified_mcse_entropies"][i]
            spec_uncertainty = original_entropy - np.mean(clarified_entropies)
            spec_uncertainties.append(spec_uncertainty)

        return np.array(spec_uncertainties)


class SpecificationUncertaintyMCNSE(Estimator):
    def __init__(
        self, verbose: bool = False, n_clarifications: int = 5
    ):
        deps = ["clarified_mcnse_entropies", "original_mcnse_entropy"]
        super().__init__(deps, "sequence")
        self.verbose = verbose
        self.n_clarifications = n_clarifications

    def __str__(self):
        return f"SpecificationUncertaintyMCNSE_n{self.n_clarifications}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        spec_uncertainties = []

        for i, original_entropy in enumerate(stats["original_mcnse_entropy"]):
            clarified_entropies = stats["clarified_mcnse_entropies"][i]
            spec_uncertainty = original_entropy - np.mean(clarified_entropies)
            spec_uncertainties.append(spec_uncertainty)

        return np.array(spec_uncertainties)
