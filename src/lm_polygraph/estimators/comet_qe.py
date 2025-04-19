import re
import numpy as np

from typing import List, Dict
from .estimator import Estimator

from comet import download_model, load_from_checkpoint


class CometQE(Estimator):
    """
    Calculates COMET metric (https://aclanthology.org/2020.emnlp-main.213/)
    between model-generated texts and ground truth texts.
    """

    def __init__(
        self,
        source_ignore_regex=None,
        translation_ignore_regex=None,
        gpus=0,
        model="Unbabel/wmt23-cometkiwi-da-xxl",
    ):
        super().__init__(["greedy_texts", "input_texts"], "sequence")
        model_path = download_model(model)
        self.scorer = load_from_checkpoint(model_path)
        self.model_name = model.split("/")[-1]
        self.source_ignore_regex = (
            re.compile(source_ignore_regex) if source_ignore_regex else None
        )
        self.translation_ignore_regex = (
            re.compile(translation_ignore_regex) if translation_ignore_regex else None
        )
        self.gpus = gpus

    def __str__(self):
        return f"CometQE-{self.model_name}"

    def _filter_source(self, text: str, ignore_regex: re.Pattern) -> str:
        if ignore_regex is not None:
            try:
                return ignore_regex.findall(text)[-1]
            except IndexError:
                raise ValueError(
                    f"Source text {text} does not match the ignore regex {ignore_regex}"
                )

    def _filter_translation(self, text: str, ignore_regex: re.Pattern) -> str:
        text = ignore_regex.sub("", text) if ignore_regex else text

        return text.strip()

    def __call__(
        self,
        stats: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Calculates COMET (https://aclanthology.org/2020.emnlp-main.213/) between
        stats['greedy_texts'], and target_texts.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * model-generated texts in 'greedy_texts'
        Returns:
            np.ndarray: list of COMET Scores for each sample in input.
        """
        sources = [
            self._filter_source(src, self.source_ignore_regex)
            for src in stats["input_texts"]
        ]
        translations = [
            self._filter_translation(tr, self.translation_ignore_regex)
            for tr in stats["greedy_texts"]
        ]

        data = []
        for original, translation in zip(sources, translations):
            data.append({'src': original, 'mt': translation})

        scores = self.scorer.predict(data, batch_size=1, gpus=self.gpus).scores
        return -np.array(scores)
