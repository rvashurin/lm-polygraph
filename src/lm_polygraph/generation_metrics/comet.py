import re
import numpy as np
from evaluate import load

from typing import List, Dict
from .generation_metric import GenerationMetric

from comet import download_model, load_from_checkpoint


class Comet(GenerationMetric):
    """
    Calculates COMET metric (https://aclanthology.org/2020.emnlp-main.213/)
    between model-generated texts and ground truth texts.
    """

    def __init__(self, source_ignore_regex=None, lang="en"):
        super().__init__(["greedy_texts", "input_texts"], "sequence")
        model_path = download_model("Unbabel/wmt22-comet-da")  
        self.scorer = load_from_checkpoint(model_path)

        self.source_ignore_regex = (
            re.compile(source_ignore_regex) if source_ignore_regex else None
        )

    def __str__(self):
        return "Comet"

    def _filter_text(self, text: str, ignore_regex: re.Pattern) -> str:
        if ignore_regex is not None:
            processed_text = ignore_regex.search(text)
            if processed_text:
                return processed_text.group(1)
            else:
                raise ValueError(
                    f"Source text {text} does not match the ignore regex {ignore_regex}"
                )
        return text

    def __call__(
        self,
        stats: Dict[str, np.ndarray],
        target_texts: List[str],
    ) -> np.ndarray:
        """
        Calculates COMET (https://aclanthology.org/2020.emnlp-main.213/) between
        stats['greedy_texts'], and target_texts.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * model-generated texts in 'greedy_texts'
            target_texts (List[str]): ground-truth texts
            input_texts (List[str]): input texts before translation
        Returns:
            np.ndarray: list of COMET Scores for each sample in input.
        """
        sources = [
            self._filter_text(src, self.source_ignore_regex)
            for src in stats["input_texts"]
        ]
        data = []
        for original, translation, reference in zip(sources, stats["greedy_texts"], stats["target_texts"]):
            data.append({'src': original, 'mt': translation, 'ref': reference})

        scores = self.scorer.predict(data, batch_size=1, gpus=1)
        return scores
