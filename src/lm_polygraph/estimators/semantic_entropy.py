import numpy as np

from collections import defaultdict
from typing import List, Dict, Optional

from .estimator import Estimator


class SemanticEntropy(Estimator):
    """
    Estimates the sequence-level uncertainty of a language model following the method of
    "Semantic entropy" as provided in the paper https://arxiv.org/abs/2302.09664.
    Works only with whitebox models (initialized using lm_polygraph.utils.model.WhiteboxModel).

    This method calculates the generation entropy estimations merged by semantic classes using Monte-Carlo.
    The number of samples is controlled by lm_polygraph.stat_calculators.sample.SamplingGenerationCalculator
    'samples_n' parameter.
    """

    def __init__(self, verbose: bool = False, deps: list = None):
        if deps is None:
            deps = [
                "sample_log_probs",
                "sample_texts",
                "semantic_matrix_entail",
                "entailment_id",
            ]
        super().__init__(
            deps,
            "sequence",
        )
        self.verbose = verbose

    def __str__(self):
        return "SemanticEntropy"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates the semantic entropy for each sample in the input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * generated samples in 'sample_texts',
                * corresponding log probabilities in 'sample_log_probs',
                * matrix with semantic similarities in 'semantic_matrix_entail'
        Returns:
            np.ndarray: float semantic entropy for each sample in input statistics.
                Higher values indicate more uncertain samples.
        """
        loglikelihoods_list = stats["sample_log_probs"]

        # entailment_id = stats["deberta"].deberta.config.label2id["ENTAILMENT"] # TODO: Why this is here??
        self._is_entailment = stats["semantic_matrix_classes"] == stats["entailment_id"]

        return self.batched_call(stats['sample_texts'], loglikelihoods_list)

    def batched_call(
        self,
        hyps_list: List[List[str]],
        loglikelihoods_list: List[List[float]],
        log_weights: Optional[List[List[float]]] = None,
    ) -> np.array:
        if log_weights is None:
            log_weights = [None for _ in hyps_list]

        self.get_classes(hyps_list)

        semantic_logits = {}
        # Iteration over batch
        for i in range(len(hyps_list)):
            class_likelihoods = [
                np.array(loglikelihoods_list[i])[np.array(class_idx)]
                for class_idx in self._class_to_sample[i]
            ]

            class_lp = [
                np.logaddexp.reduce(likelihoods) for likelihoods in class_likelihoods
            ]
            if log_weights[i] is None:
                log_weights[i] = [0 for _ in hyps_list[i]]

            semantic_logits[i] = -np.mean(
                [
                    class_lp[self._sample_to_class[i][j]] * np.exp(log_weights[i][j])
                    for j in range(len(hyps_list[i]))
                ]
            )

        return np.array([semantic_logits[i] for i in range(len(hyps_list))])

    def get_classes(self, hyps_list: List[List[str]]):
        self._sample_to_class = {}
        self._class_to_sample: Dict[int, List] = defaultdict(list)
        self._class_to_unique_sample: Dict[int, List] = defaultdict(list)

        [
            self._determine_class(idx, i, hyps_list)
            for idx, hyp in enumerate(hyps_list)
            for i in range(len(hyp))
        ]

        return self._sample_to_class, self._class_to_sample, self._class_to_unique_sample

    def _determine_class(self, idx: int, i: int, hyps_list):
        # For first hypo just create a zeroth class
        if i == 0:
            self._class_to_sample[idx] = [[0]]
            self._class_to_unique_sample[idx] = [[0]]
            self._sample_to_class[idx] = {0: 0}

            return 0

        # Iterate over existing classes and return if hypo belongs to one of them
        for class_id in range(len(self._class_to_sample[idx])):
            class_text_id = self._class_to_sample[idx][class_id][0]
            forward_entailment = self._is_entailment[idx, class_text_id, i]
            backward_entailment = self._is_entailment[idx, i, class_text_id]
            if forward_entailment and backward_entailment:
                class_hyps = [
                    hyps_list[idx][hyp_id] for hyp_id in self._class_to_sample[idx][class_id]
                ]

                if hyps_list[idx][i] not in class_hyps:
                    self._class_to_unique_sample[idx][class_id].append(i)

                self._class_to_sample[idx][class_id].append(i)
                self._sample_to_class[idx][i] = class_id

                return class_id

        # If none of the existing classes satisfy - create new one
        new_class_id = len(self._class_to_sample[idx])
        self._sample_to_class[idx][i] = new_class_id
        self._class_to_sample[idx].append([i])
        self._class_to_unique_sample[idx].append([i])

        return new_class_id


class SemanticEntropyUnique(SemanticEntropy):
    def __init__(self, verbose: bool = False, deps: list = None):
        if deps is None:
            deps=[
                "sample_log_probs",
                "sample_texts",
                "semantic_matrix_entail",
                "entailment_id",
            ]
        super().__init__(deps=deps)
        self.verbose = verbose

    def __str__(self):
        return "SemanticEntropyUnique"

    def batched_call(
        self,
        hyps_list: List[List[str]],
        loglikelihoods_list: List[List[float]],
        log_weights: Optional[List[List[float]]] = None,
    ) -> np.array:
        if log_weights is None:
            log_weights = [None for _ in hyps_list]

        self.get_classes(hyps_list)

        semantic_logits = {}
        # Iteration over batch
        for i in range(len(hyps_list)):
            class_likelihoods = [
                np.array(loglikelihoods_list[i])[np.array(class_idx)]
                for class_idx in self._class_to_unique_sample[i]
            ]

            class_lp = [
                np.logaddexp.reduce(likelihoods) for likelihoods in class_likelihoods
            ]
            if log_weights[i] is None:
                log_weights[i] = [0 for _ in hyps_list[i]]

            semantic_logits[i] = -np.mean(
                [
                    class_lp[self._sample_to_class[i][j]] * np.exp(log_weights[i][j])
                    for j in range(len(hyps_list[i]))
                ]
            )

        return np.array([semantic_logits[i] for i in range(len(hyps_list))])


class SemanticEntropyNormalized(SemanticEntropyUnique):
    def __init__(self, verbose: bool = False):
        super().__init__(deps=[
                "normalized_sample_log_probs",
                "sample_texts",
                "semantic_matrix_entail",
                "entailment_id",
            ])
        self.verbose = verbose

    def __str__(self):
        return "SemanticEntropyNormalized"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates the semantic entropy for each sample in the input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * generated samples in 'sample_texts',
                * corresponding log probabilities in 'sample_log_probs',
                * matrix with semantic similarities in 'semantic_matrix_entail'
        Returns:
            np.ndarray: float semantic entropy for each sample in input statistics.
                Higher values indicate more uncertain samples.
        """
        loglikelihoods_list = stats["normalized_sample_log_probs"]

        # entailment_id = stats["deberta"].deberta.config.label2id["ENTAILMENT"] # TODO: Why this is here??
        self._is_entailment = stats["semantic_matrix_classes"] == stats["entailment_id"]

        return self.batched_call(stats['sample_texts'], loglikelihoods_list)


class SemanticEntropyNormalizedIW(SemanticEntropyUnique):
    def __init__(self, verbose: bool = False):
        super().__init__(deps=[
                "sample_log_probs",
                "normalized_sample_log_probs",
                "sample_texts",
                "semantic_matrix_entail",
                "entailment_id",
            ])
        self.verbose = verbose


    def __str__(self):
        return "SemanticEntropyNormalizedIW"


    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates the semantic entropy for each sample in the input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * generated samples in 'sample_texts',
                * corresponding log probabilities in 'sample_log_probs',
                * matrix with semantic similarities in 'semantic_matrix_entail'
        Returns:
            np.ndarray: float semantic entropy for each sample in input statistics.
                Higher values indicate more uncertain samples.
        """
        loglikelihoods_list = stats["sample_log_probs"]
        normalized_loglikelihoods_list = stats["normalized_sample_log_probs"]

        # entailment_id = stats["deberta"].deberta.config.label2id["ENTAILMENT"] # TODO: Why this is here??
        self._is_entailment = stats["semantic_matrix_classes"] == stats["entailment_id"]

        return self.batched_call(stats['sample_texts'],
                                 loglikelihoods_list, normalized_loglikelihoods_list)


    def batched_call(
        self,
        hyps_list: List[List[str]],
        loglikelihoods_list: List[List[float]],
        normalized_loglikelihoods_list: List[List[float]],
    ) -> np.array:
        log_weights = [None for _ in hyps_list]

        self.get_classes(hyps_list)

        semantic_logits = {}
        # Iteration over batch
        for i in range(len(hyps_list)):
            class_likelihoods = [
                np.array(loglikelihoods_list[i])[np.array(class_idx)]
                for class_idx in self._class_to_unique_sample[i]
            ]

            class_lp = [
                np.logaddexp.reduce(likelihoods) for likelihoods in class_likelihoods
            ]

            normalized_class_likelihoods = [
                np.array(normalized_loglikelihoods_list[i])[np.array(class_idx)]
                for class_idx in self._class_to_unique_sample[i]
            ]
            normalized_class_lp = [
                np.logaddexp.reduce(likelihoods) for likelihoods in normalized_class_likelihoods
            ]
            
            importance_weights = np.exp(np.array(normalized_class_lp) - np.array(class_lp))

            semantic_logits[i] = -np.mean(
                [
                    normalized_class_lp[self._sample_to_class[i][j]] * importance_weights[self._sample_to_class[i][j]]
                    for j in range(len(hyps_list[i]))
                ]
            )

        return np.array([semantic_logits[i] for i in range(len(hyps_list))])


class SemanticEntropyNormalizedIWLower(SemanticEntropyUnique):
    def __init__(self, verbose: bool = False, deps: list = None):
        if deps is None:
            deps=[
                "sample_log_probs",
                "normalized_sample_log_probs",
                "sample_texts",
                "importance_weights",
                "partition_lower",
                "semantic_matrix_entail",
                "entailment_id",
            ]
        super().__init__(deps=deps)
        self.verbose = verbose

    def __str__(self):
        return "SemanticEntropyNormalizedIWLower"


    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates the semantic entropy for each sample in the input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * generated samples in 'sample_texts',
                * corresponding log probabilities in 'sample_log_probs',
                * matrix with semantic similarities in 'semantic_matrix_entail'
        Returns:
            np.ndarray: float semantic entropy for each sample in input statistics.
                Higher values indicate more uncertain samples.
        """
        loglikelihoods_list = stats["sample_log_probs"]
        normalized_loglikelihoods_list = stats["normalized_sample_log_probs"]
        partition = stats["partition_lower"]

        # entailment_id = stats["deberta"].deberta.config.label2id["ENTAILMENT"] # TODO: Why this is here??
        self._is_entailment = stats["semantic_matrix_classes"] == stats["entailment_id"]

        return self.batched_call(stats['sample_texts'],
                                 loglikelihoods_list,
                                 normalized_loglikelihoods_list,
                                 partition)


    def batched_call(
        self,
        hyps_list: List[List[str]],
        loglikelihoods_list: List[List[float]],
        normalized_loglikelihoods_list: List[List[float]],
        partition: List[float],
    ) -> np.array:
        log_weights = [None for _ in hyps_list]

        self.get_classes(hyps_list)

        semantic_logits = {}

        # Iteration over batch
        for i in range(len(hyps_list)):
            # summation over unique hyps in class
            class_likelihoods = [
                np.array(loglikelihoods_list[i])[np.array(class_idx)]
                for class_idx in self._class_to_unique_sample[i]
            ]

            class_lp = [
                np.logaddexp.reduce(likelihoods) for likelihoods in class_likelihoods
            ]

            normalized_class_likelihoods = [
                np.array(normalized_loglikelihoods_list[i])[np.array(class_idx)]
                for class_idx in self._class_to_unique_sample[i]
            ]
            normalized_class_lp = [
                np.logaddexp.reduce(likelihoods) for likelihoods in normalized_class_likelihoods
            ]
            
            importance_weights = np.exp(np.array(normalized_class_lp) - np.array(class_lp)) / partition[i]
            summand = normalized_class_lp - np.log(partition[i])

            semantic_logits[i] = -np.mean(
                [
                    summand[self._sample_to_class[i][j]] * importance_weights[self._sample_to_class[i][j]]
                    for j in range(len(hyps_list[i]))
                ]
            )

        return np.array([semantic_logits[i] for i in range(len(hyps_list))])


class SemanticEntropyNormalizedIWUpper(SemanticEntropyNormalizedIWLower):
    def __init__(self, verbose: bool = False, deps: list = None):
        if deps is None:
            deps=[
                "sample_log_probs",
                "normalized_sample_log_probs",
                "sample_texts",
                "importance_weights",
                "partition_upper",
                "semantic_matrix_entail",
                "entailment_id",
            ]
        super().__init__(deps=deps)
        self.verbose = verbose

    def __str__(self):
        return "SemanticEntropyNormalizedIWUpper"


    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates the semantic entropy for each sample in the input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * generated samples in 'sample_texts',
                * corresponding log probabilities in 'sample_log_probs',
                * matrix with semantic similarities in 'semantic_matrix_entail'
        Returns:
            np.ndarray: float semantic entropy for each sample in input statistics.
                Higher values indicate more uncertain samples.
        """
        loglikelihoods_list = stats["sample_log_probs"]
        normalized_loglikelihoods_list = stats["normalized_sample_log_probs"]
        partition = stats["partition_upper"]

        # entailment_id = stats["deberta"].deberta.config.label2id["ENTAILMENT"] # TODO: Why this is here??
        self._is_entailment = stats["semantic_matrix_classes"] == stats["entailment_id"]

        return self.batched_call(stats['sample_texts'],
                                 loglikelihoods_list,
                                 normalized_loglikelihoods_list,
                                 partition)


class SemanticEntropyNormalizedIWAve(SemanticEntropyNormalizedIWLower):
    def __init__(self, verbose: bool = False, deps: list = None):
        if deps is None:
            deps=[
                "sample_log_probs",
                "normalized_sample_log_probs",
                "sample_texts",
                "importance_weights",
                "partition_ave",
                "semantic_matrix_entail",
                "entailment_id",
            ]
        super().__init__(deps=deps)
        self.verbose = verbose

    def __str__(self):
        return "SemanticEntropyNormalizedIWAve"


    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates the semantic entropy for each sample in the input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * generated samples in 'sample_texts',
                * corresponding log probabilities in 'sample_log_probs',
                * matrix with semantic similarities in 'semantic_matrix_entail'
        Returns:
            np.ndarray: float semantic entropy for each sample in input statistics.
                Higher values indicate more uncertain samples.
        """
        loglikelihoods_list = stats["sample_log_probs"]
        normalized_loglikelihoods_list = stats["normalized_sample_log_probs"]
        partition = stats["partition_ave"]

        # entailment_id = stats["deberta"].deberta.config.label2id["ENTAILMENT"] # TODO: Why this is here??
        self._is_entailment = stats["semantic_matrix_classes"] == stats["entailment_id"]

        return self.batched_call(stats['sample_texts'],
                                 loglikelihoods_list,
                                 normalized_loglikelihoods_list,
                                 partition)
