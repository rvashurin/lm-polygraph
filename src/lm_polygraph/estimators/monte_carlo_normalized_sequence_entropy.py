import numpy as np

from typing import Dict

from .estimator import Estimator


class MonteCarloNormalizedSequenceEntropy(Estimator):
    def __init__(self):
        """
        Estimates the sequence-level uncertainty of a language model following the method of
        "Length-normalized predictive entropy" as provided in the paper https://arxiv.org/abs/2302.09664.
        Works only with whitebox models (initialized using lm_polygraph.utils.model.WhiteboxModel).

        This method calculates the generation entropy estimations using Monte-Carlo estimation with length normalization.
        The number of samples is controlled by lm_polygraph.stat_calculators.sample.SamplingGenerationCalculator
        'samples_n' parameter.
        """
        super().__init__(["sample_log_probs", "sample_tokens"], "sequence")

    def __str__(self):
        return "MonteCarloNormalizedSequenceEntropy"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates length normalized entropy with Monte-Carlo for each sample in the input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * a list of tokens for each sample, in 'sample_tokens'
                * log probabilities for each token, in 'sample_log_probs'
        Returns:
            np.ndarray: float uncertainty for each sample in input statistics.
                Higher values indicate more uncertain samples.
        """
        logprobs = stats["sample_log_probs"]
        tokens = stats["sample_tokens"]
        return np.array(
            [
                -np.mean([lp_i / len(t_i) for lp_i, t_i in zip(lp, t) if len(t_i)])
                for lp, t in zip(logprobs, tokens)
            ]
        )

class MonteCarloNormalizedSequenceEntropyIW(Estimator):
    def __init__(self):
        """
        Estimates the sequence-level uncertainty of a language model following the method of
        "Length-normalized predictive entropy" as provided in the paper https://arxiv.org/abs/2302.09664.
        Works only with whitebox models (initialized using lm_polygraph.utils.model.WhiteboxModel).

        This method calculates the generation entropy estimations using Monte-Carlo estimation with length normalization.
        The number of samples is controlled by lm_polygraph.stat_calculators.sample.SamplingGenerationCalculator
        'samples_n' parameter.
        """
        super().__init__(["normalized_sample_log_probs", "importance_weights"], "sequence")

    def __str__(self):
        return "MonteCarloNormalizedSequenceEntropyIW"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates length normalized entropy with Monte-Carlo for each sample in the input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * a list of tokens for each sample, in 'sample_tokens'
                * log probabilities for each token, in 'sample_log_probs'
        Returns:
            np.ndarray: float uncertainty for each sample in input statistics.
                Higher values indicate more uncertain samples.
        """
        logprobs = stats["normalized_sample_log_probs"]
        weights = stats["importance_weights"]


        return -(logprobs * weights).mean(-1)

class MonteCarloNormalizedSequenceEntropyIWUpper(Estimator):
    def __init__(self):
        """
        Estimates the sequence-level uncertainty of a language model following the method of
        "Length-normalized predictive entropy" as provided in the paper https://arxiv.org/abs/2302.09664.
        Works only with whitebox models (initialized using lm_polygraph.utils.model.WhiteboxModel).

        This method calculates the generation entropy estimations using Monte-Carlo estimation with length normalization.
        The number of samples is controlled by lm_polygraph.stat_calculators.sample.SamplingGenerationCalculator
        'samples_n' parameter.
        """
        super().__init__(["normalized_sample_log_probs", "importance_weights", "partition_upper"], "sequence")

    def __str__(self):
        return "MonteCarloNormalizedSequenceEntropyIWUpper"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates length normalized entropy with Monte-Carlo for each sample in the input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * a list of tokens for each sample, in 'sample_tokens'
                * log probabilities for each token, in 'sample_log_probs'
        Returns:
            np.ndarray: float uncertainty for each sample in input statistics.
                Higher values indicate more uncertain samples.
        """
        logprobs = stats["normalized_sample_log_probs"]
        weights = stats["importance_weights"]
        partition = stats["partition_upper"]


        return -(logprobs * weights).mean(-1) * partition

class MonteCarloNormalizedSequenceEntropyIWLower(Estimator):
    def __init__(self):
        """
        Estimates the sequence-level uncertainty of a language model following the method of
        "Length-normalized predictive entropy" as provided in the paper https://arxiv.org/abs/2302.09664.
        Works only with whitebox models (initialized using lm_polygraph.utils.model.WhiteboxModel).

        This method calculates the generation entropy estimations using Monte-Carlo estimation with length normalization.
        The number of samples is controlled by lm_polygraph.stat_calculators.sample.SamplingGenerationCalculator
        'samples_n' parameter.
        """
        super().__init__(["normalized_sample_log_probs", "importance_weights", "partition_lower"], "sequence")

    def __str__(self):
        return "MonteCarloNormalizedSequenceEntropyIWLower"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates length normalized entropy with Monte-Carlo for each sample in the input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * a list of tokens for each sample, in 'sample_tokens'
                * log probabilities for each token, in 'sample_log_probs'
        Returns:
            np.ndarray: float uncertainty for each sample in input statistics.
                Higher values indicate more uncertain samples.
        """
        logprobs = stats["normalized_sample_log_probs"]
        weights = stats["importance_weights"]
        partition = stats["partition_lower"]


        return -(logprobs * weights).mean(-1) * partition

class MonteCarloNormalizedSequenceEntropyIWAve(Estimator):
    def __init__(self):
        """
        Estimates the sequence-level uncertainty of a language model following the method of
        "Length-normalized predictive entropy" as provided in the paper https://arxiv.org/abs/2302.09664.
        Works only with whitebox models (initialized using lm_polygraph.utils.model.WhiteboxModel).

        This method calculates the generation entropy estimations using Monte-Carlo estimation with length normalization.
        The number of samples is controlled by lm_polygraph.stat_calculators.sample.SamplingGenerationCalculator
        'samples_n' parameter.
        """
        super().__init__(["normalized_sample_log_probs", "importance_weights", "partition_ave"], "sequence")

    def __str__(self):
        return "MonteCarloNormalizedSequenceEntropyIWAve"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Estimates length normalized entropy with Monte-Carlo for each sample in the input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * a list of tokens for each sample, in 'sample_tokens'
                * log probabilities for each token, in 'sample_log_probs'
        Returns:
            np.ndarray: float uncertainty for each sample in input statistics.
                Higher values indicate more uncertain samples.
        """
        logprobs = stats["normalized_sample_log_probs"]
        weights = stats["importance_weights"]
        partition = stats["partition_ave"]


        return -(logprobs * weights).mean(-1) * partition
