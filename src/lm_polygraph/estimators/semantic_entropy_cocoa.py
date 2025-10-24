import numpy as np

from typing import List, Dict, Optional

from .estimator import Estimator
from .common import sample_strategy_to_prefix, best_sample_ids, SAMPLE_SELECTION_STAT_KEYS


class SemanticEntropyCocoaMaxprob(Estimator):
    """
    Estimates the sequence-level uncertainty of a language model following the method of
    "Semantic entropy" as provided in the paper https://arxiv.org/abs/2302.09664.
    Works only with whitebox models (initialized using lm_polygraph.utils.model.WhiteboxModel).

    This method calculates the generation entropy estimations merged by semantic classes using Monte-Carlo.
    The number of samples is controlled by lm_polygraph.stat_calculators.sample.SamplingGenerationCalculator
    'samples_n' parameter.
    """

    def __init__(
        self, verbose: bool = False, class_probability_estimation: str = "sum", sample_strategy: str = "first"
    ):
        self.sample_strategy = sample_strategy
        self.class_probability_estimation = class_probability_estimation
        if self.class_probability_estimation == "sum":
            deps = ["sample_log_probs", "sample_texts", "semantic_classes_entail"]
        elif self.class_probability_estimation == "frequency":
            deps = ["sample_texts", "semantic_classes_entail"]
        else:
            raise ValueError(
                f"Unknown class_probability_estimation: {self.class_probability_estimation}. Use 'sum' or 'frequency'."
            )

        super().__init__(deps + SAMPLE_SELECTION_STAT_KEYS, "sequence")
        self.verbose = verbose

    def __str__(self):
        if self.class_probability_estimation == "sum":
            base = "SemanticEntropyCocoaMaxprob"
        elif self.class_probability_estimation == "frequency":
            base = "SemanticEntropyCocoaMaxprobEmpirical"
        return sample_strategy_to_prefix(self.sample_strategy) + base

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_sample_log_probs = stats["sample_log_probs"]
        batch_sample_class_to_sample = stats["semantic_classes_entail"]["class_to_sample"]
        batch_sample_sample_to_class = stats["semantic_classes_entail"]["sample_to_class"]
        batch_semantic_matrix = stats["semantic_matrix_entail"]

        sample_ids = best_sample_ids(self.sample_strategy, stats)

        results = []

        for sample_id, sample_log_probs, sample_to_class, class_to_sample, semantic_matrix in zip(
            sample_ids,
            batch_sample_log_probs,
            batch_sample_sample_to_class,
            batch_sample_class_to_sample,
            batch_semantic_matrix,
        ):
            sample_class = sample_to_class[sample_id]
            class_samples = class_to_sample[sample_class]

            class_p = -np.logaddexp.reduce(np.array(sample_log_probs)[class_samples])

            intercluster_similarities = []
            for class_i, other_class_samples in enumerate(class_to_sample):
                if class_i == sample_class:
                    continue

                similarity = semantic_matrix[class_samples, :][:, other_class_samples].mean()
                intercluster_similarities.append(similarity)

            if len(intercluster_similarities) == 0:
                intercluster_similarities = [0.0]

            result = (class_p * (1 - np.mean(intercluster_similarities)))
            results.append(result)

        return np.array(results)


class SemanticEntropyCocoaPPL(Estimator):
    """
    Estimates the sequence-level uncertainty of a language model following the method of
    "Semantic entropy" as provided in the paper https://arxiv.org/abs/2302.09664.
    Works only with whitebox models (initialized using lm_polygraph.utils.model.WhiteboxModel).

    This method calculates the generation entropy estimations merged by semantic classes using Monte-Carlo.
    The number of samples is controlled by lm_polygraph.stat_calculators.sample.SamplingGenerationCalculator
    'samples_n' parameter.
    """

    def __init__(
        self, verbose: bool = False, class_probability_estimation: str = "sum", sample_strategy: str = "first"
    ):
        self.sample_strategy = sample_strategy
        self.class_probability_estimation = class_probability_estimation
        if self.class_probability_estimation == "sum":
            deps = ["sample_log_likelihoods", "sample_texts", "semantic_classes_entail"]
        elif self.class_probability_estimation == "frequency":
            deps = ["sample_texts", "semantic_classes_entail"]
        else:
            raise ValueError(
                f"Unknown class_probability_estimation: {self.class_probability_estimation}. Use 'sum' or 'frequency'."
            )

        super().__init__(deps + SAMPLE_SELECTION_STAT_KEYS, "sequence")
        self.verbose = verbose

    def __str__(self):
        if self.class_probability_estimation == "sum":
            base = "SemanticEntropyCocoaPPL"
        elif self.class_probability_estimation == "frequency":
            base = "SemanticEntropyCocoaPPLEmpirical"
        return sample_strategy_to_prefix(self.sample_strategy) + base

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batch_sample_log_likelihoods = stats["sample_log_likelihoods"]
        batch_sample_class_to_sample = stats["semantic_classes_entail"]["class_to_sample"]
        batch_sample_sample_to_class = stats["semantic_classes_entail"]["sample_to_class"]
        batch_semantic_matrix = stats["semantic_matrix_entail"]

        sample_ids = best_sample_ids(self.sample_strategy, stats)

        results = []

        for sample_id, sample_log_likelihoods, sample_to_class, class_to_sample, semantic_matrix in zip(
            sample_ids,
            batch_sample_log_likelihoods,
            batch_sample_sample_to_class,
            batch_sample_class_to_sample,
            batch_semantic_matrix,
        ):
            sample_class = sample_to_class[sample_id]
            class_samples = class_to_sample[sample_class]

            sample_log_probs = np.array([np.mean(token_ll) for token_ll in sample_log_likelihoods])
            class_p = -np.logaddexp.reduce(np.array(sample_log_probs)[class_samples])

            intercluster_similarities = []
            for class_i, other_class_samples in enumerate(class_to_sample):
                if class_i == sample_class:
                    continue

                similarity = semantic_matrix[class_samples, :][:, other_class_samples].mean()
                intercluster_similarities.append(similarity)

            if len(intercluster_similarities) == 0:
                intercluster_similarities = [0.0]

            result = (class_p * (1 - np.mean(intercluster_similarities)))
            results.append(result)

        return np.array(results)
