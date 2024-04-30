import numpy as np

from typing import Dict

from .estimator import Estimator
import torch


def get_batched_logits(stats):
    return [torch.from_numpy(logits).to(torch.float32) for logits in stats['router_logits']]


class TotalMeanExpertEntropy(Estimator):
    def __init__(self):
        super().__init__(["router_logits"], "sequence")

    def __str__(self):
        return "TotalMeanExpertEntropy"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batched_logits = get_batched_logits(stats)

        entropies = []
        for router_logits, greedy_tokens in zip(batched_logits, stats['greedy_tokens']):
            expert_entropies = torch.distributions.Categorical(router_logits.softmax(-1)).entropy()

            entropies.append(expert_entropies.mean())

        return np.array(entropies)


class TotalEntropyOfExpertMean(Estimator):
    def __init__(self):
        super().__init__(["router_logits"], "sequence")

    def __str__(self):
        return "TotalEntropyOfExpertMean"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batched_logits = get_batched_logits(stats)

        entropies = []
        for router_logits, greedy_tokens in zip(batched_logits, stats['greedy_tokens']):
            mean_logits = router_logits.softmax(-1).mean(dim=(0,1))
            entropies.append(torch.distributions.Categorical(mean_logits).entropy())

        return np.array(entropies)

class FirstTokenMeanExpertEntropy(Estimator):
    def __init__(self):
        super().__init__(["router_logits"], "sequence")

    def __str__(self):
        return "FirstTokenMeanExpertEntropy"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batched_logits = get_batched_logits(stats)

        entropies = []
        for router_logits, greedy_tokens in zip(batched_logits, stats['greedy_tokens']):
            gen_logits = router_logits[-len(greedy_tokens):,:,:]
            expert_entropies = torch.distributions.Categorical(gen_logits.softmax(-1)).entropy()

            entropies.append(expert_entropies.mean())

        return np.array(entropies)


class FirstTokenEntropyOfExpertMean(Estimator):
    def __init__(self):
        super().__init__(["router_logits"], "sequence")

    def __str__(self):
        return "FirstTokenEntropyOfExpertMean"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        batched_logits = get_batched_logits(stats)

        entropies = []
        for router_logits, greedy_tokens in zip(batched_logits, stats['greedy_tokens']):
            gen_logits = router_logits[-len(greedy_tokens):,:,:]
            mean_logits = gen_logits.softmax(-1).mean(dim=(0,1))
            entropies.append(torch.distributions.Categorical(mean_logits).entropy())

        return np.array(entropies)
