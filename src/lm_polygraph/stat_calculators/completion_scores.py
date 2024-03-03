import torch
import numpy as np

from typing import Dict, List

from .embeddings import get_embeddings_from_output
from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel, BlackboxModel

class CompletionScoresCalculator(StatCalculator):
    def __init__(self):
        super().__init__(
            [
                "completion_scores",
                "mixtral_mean_entropies",
                "mixtral_entropies_of_mean",
            ],
            [],
        )

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        completions = [text.split('|') for text in texts]
        for compl in completions:
            with torch.no_grad():
                batch: Dict[str, torch.Tensor] = model.tokenize(compl)
                batch = {k: v.to(model.device()) for k, v in batch.items()}
                breakpoint()
                out = model(**batch)
                breakpoint()
                pass

        result_dict = {
            "completion_scores": texts,
            "mixtral_mean_entropies": torch.stack(out.mean_entropies).cpu().numpy(),
            "mixtral_entropies_of_mean": torch.stack(out.entropies_of_mean).cpu().numpy(),
        }

        return result_dict
