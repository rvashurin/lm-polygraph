import numpy as np

from typing import List, Dict, Optional

from .estimator import Estimator



class SpecificationUncertaintySemantic(Estimator):
    def __init__(
        self, verbose: bool = False, n_clarifications: int = 5
    ):
        deps = ["clarified_semantic_entropies", "avg_semantic_entropy"]
        super().__init__(deps, "sequence")
        self.verbose = verbose
        self.n_clarifications = n_clarifications

    def __str__(self):
        return f"SpecificationUncertaintySemantic_n{self.n_clarifications}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        spec_uncertainties = []

        for i, avg_entropy in enumerate(stats["avg_semantic_entropy"]):
            clarified_entropies = stats["clarified_semantic_entropies"][i]
            spec_uncertainty = avg_entropy - np.mean(clarified_entropies)
            spec_uncertainties.append(spec_uncertainty)

        return np.array(spec_uncertainties)


class SpecificationUncertaintySemanticNormalized(Estimator):
    def __init__(
        self, verbose: bool = False, n_clarifications: int = 5
    ):
        deps = ["clarified_semantic_entropies_normalized", "avg_semantic_entropy_normalized"]
        super().__init__(deps, "sequence")
        self.verbose = verbose
        self.n_clarifications = n_clarifications

    def __str__(self):
        return f"SpecificationUncertaintySemanticNormalized_n{self.n_clarifications}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        spec_uncertainties = []

        for i, avg_entropy in enumerate(stats["avg_semantic_entropy_normalized"]):
            clarified_entropies = stats["clarified_semantic_entropies_normalized"][i]
            spec_uncertainty = avg_entropy - np.mean(clarified_entropies)
            spec_uncertainties.append(spec_uncertainty)

        return np.array(spec_uncertainties)


class SpecificationUncertaintySemanticDirect(Estimator):
    def __init__(
        self, verbose: bool = False, n_clarifications: int = 5
    ):
        deps = ["clarified_semantic_entropies_direct", "avg_semantic_entropy_direct"]
        super().__init__(deps, "sequence")
        self.verbose = verbose
        self.n_clarifications = n_clarifications
    
    def __str__(self):
        return f"SpecificationUncertaintySemanticDirect_n{self.n_clarifications}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        spec_uncertainties = []

        for i, avg_entropy in enumerate(stats["avg_semantic_entropy_direct"]):
            clarified_entropies = stats["clarified_semantic_entropies_direct"][i]
            spec_uncertainty = avg_entropy - np.mean(clarified_entropies)
            spec_uncertainties.append(spec_uncertainty)

        return np.array(spec_uncertainties)


class SpecificationUncertaintySemanticDirectNormalized(Estimator):
    def __init__(
        self, verbose: bool = False, n_clarifications: int = 5
    ):
        deps = ["clarified_semantic_entropies_direct_normalized", "avg_semantic_entropy_direct_normalized"]
        super().__init__(deps, "sequence")
        self.verbose = verbose
        self.n_clarifications = n_clarifications

    def __str__(self):
        return f"SpecificationUncertaintySemanticDirectNormalized_n{self.n_clarifications}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        spec_uncertainties = []

        for i, avg_entropy in enumerate(stats["avg_semantic_entropy_direct_normalized"]):
            clarified_entropies = stats["clarified_semantic_entropies_direct_normalized"][i]
            spec_uncertainty = avg_entropy - np.mean(clarified_entropies)
            spec_uncertainties.append(spec_uncertainty)

        return np.array(spec_uncertainties)


class SpecificationUncertaintyMCSE(Estimator):
    def __init__(
        self, verbose: bool = False, n_clarifications: int = 5
    ):
        deps = ["clarified_mcse_entropies", "avg_mcse_entropy"]
        super().__init__(deps, "sequence")
        self.verbose = verbose
        self.n_clarifications = n_clarifications

    def __str__(self):
        return f"SpecificationUncertaintyMCSE_n{self.n_clarifications}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        spec_uncertainties = []

        for i, avg_entropy in enumerate(stats["avg_mcse_entropy"]):
            clarified_entropies = stats["clarified_mcse_entropies"][i]
            spec_uncertainty = avg_entropy - np.mean(clarified_entropies)
            spec_uncertainties.append(spec_uncertainty)

        return np.array(spec_uncertainties)


class SpecificationUncertaintyMCNSE(Estimator):
    def __init__(
        self, verbose: bool = False, n_clarifications: int = 5
    ):
        deps = ["clarified_mcnse_entropies", "avg_mcnse_entropy"]
        super().__init__(deps, "sequence")
        self.verbose = verbose
        self.n_clarifications = n_clarifications

    def __str__(self):
        return f"SpecificationUncertaintyMCNSE_n{self.n_clarifications}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        spec_uncertainties = []

        for i, avg_entropy in enumerate(stats["avg_mcnse_entropy"]):
            clarified_entropies = stats["clarified_mcnse_entropies"][i]
            spec_uncertainty = avg_entropy - np.mean(clarified_entropies)
            spec_uncertainties.append(spec_uncertainty)

        return np.array(spec_uncertainties)


class DialogueSpecificationUncertaintySemantic(Estimator):
    def __init__(self, verbose: bool = False, n_answers: int = 5):
        deps = ["dialogue_semantic_entropies", "avg_dialogue_semantic_entropy"]
        super().__init__(deps, "sequence")
        self.verbose = verbose
        self.n_answers = n_answers

    def __str__(self):
        return f"DialogueSpecificationUncertaintySemantic_n{self.n_answers}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        return np.array([
            avg - np.mean(stats["dialogue_semantic_entropies"][i])
            for i, avg in enumerate(stats["avg_dialogue_semantic_entropy"])
        ])


class DialogueSpecificationUncertaintySemanticNormalized(Estimator):
    def __init__(self, verbose: bool = False, n_answers: int = 5):
        deps = ["dialogue_semantic_entropies_normalized", "avg_dialogue_semantic_entropy_normalized"]
        super().__init__(deps, "sequence")
        self.verbose = verbose
        self.n_answers = n_answers

    def __str__(self):
        return f"DialogueSpecificationUncertaintySemanticNormalized_n{self.n_answers}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        return np.array([
            avg - np.mean(stats["dialogue_semantic_entropies_normalized"][i])
            for i, avg in enumerate(stats["avg_dialogue_semantic_entropy_normalized"])
        ])


class DialogueSpecificationUncertaintySemanticDirect(Estimator):
    def __init__(self, verbose: bool = False, n_answers: int = 5):
        deps = ["dialogue_semantic_entropies_direct", "avg_dialogue_semantic_entropy_direct"]
        super().__init__(deps, "sequence")
        self.verbose = verbose
        self.n_answers = n_answers

    def __str__(self):
        return f"DialogueSpecificationUncertaintySemanticDirect_n{self.n_answers}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        return np.array([
            avg - np.mean(stats["dialogue_semantic_entropies_direct"][i])
            for i, avg in enumerate(stats["avg_dialogue_semantic_entropy_direct"])
        ])


class DialogueSpecificationUncertaintySemanticDirectNormalized(Estimator):
    def __init__(self, verbose: bool = False, n_answers: int = 5):
        deps = [
            "dialogue_semantic_entropies_direct_normalized",
            "avg_dialogue_semantic_entropy_direct_normalized",
        ]
        super().__init__(deps, "sequence")
        self.verbose = verbose
        self.n_answers = n_answers

    def __str__(self):
        return f"DialogueSpecificationUncertaintySemanticDirectNormalized_n{self.n_answers}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        return np.array([
            avg - np.mean(stats["dialogue_semantic_entropies_direct_normalized"][i])
            for i, avg in enumerate(stats["avg_dialogue_semantic_entropy_direct_normalized"])
        ])


class DialogueSpecificationUncertaintyMCSE(Estimator):
    def __init__(self, verbose: bool = False, n_answers: int = 5):
        deps = ["dialogue_mcse_entropies", "avg_dialogue_mcse_entropy"]
        super().__init__(deps, "sequence")
        self.verbose = verbose
        self.n_answers = n_answers

    def __str__(self):
        return f"DialogueSpecificationUncertaintyMCSE_n{self.n_answers}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        return np.array([
            avg - np.mean(stats["dialogue_mcse_entropies"][i])
            for i, avg in enumerate(stats["avg_dialogue_mcse_entropy"])
        ])


class DialogueSpecificationUncertaintyMCNSE(Estimator):
    def __init__(self, verbose: bool = False, n_answers: int = 5):
        deps = ["dialogue_mcnse_entropies", "avg_dialogue_mcnse_entropy"]
        super().__init__(deps, "sequence")
        self.verbose = verbose
        self.n_answers = n_answers

    def __str__(self):
        return f"DialogueSpecificationUncertaintyMCNSE_n{self.n_answers}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        return np.array([
            avg - np.mean(stats["dialogue_mcnse_entropies"][i])
            for i, avg in enumerate(stats["avg_dialogue_mcnse_entropy"])
        ])
