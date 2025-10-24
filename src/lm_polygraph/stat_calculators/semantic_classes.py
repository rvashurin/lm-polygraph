import numpy as np

from collections import defaultdict
from typing import Dict, List

from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel


class SemanticClassesCalculator(StatCalculator):
    """
    Paritions samples into semantic classes based on semantic matrix.
    """

    def __init__(self):
        super().__init__(
            [
                "semantic_classes_entail",
            ],
            [
                "sample_texts",
                "semantic_matrix_entail",
                "semantic_matrix_classes",
                "entailment_id",
            ],
        )

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        self._is_entailment = (
            dependencies["semantic_matrix_classes"] == dependencies["entailment_id"]
        )
        self.get_classes(dependencies["sample_texts"])

        return {
            "semantic_classes_entail": {
                "sample_to_class": self._sample_to_class,
                "class_to_sample": self._class_to_sample,
            }
        }


    def get_classes(self, hyps_list: List[List[str]]):
        self._sample_to_class = []
        self._class_to_sample = []

        for idx, hyp in enumerate(hyps_list):
            class_to_sample = [[0]]
            sample_to_class = {0: 0}

            for i in range(len(hyp)):
                self._determine_class(idx, i, class_to_sample, sample_to_class)

            self._sample_to_class.append(sample_to_class)
            self._class_to_sample.append(class_to_sample)


    def _determine_class(
        self,
        idx: int,
        i: int,
        class_to_sample: List[List[int]],
        sample_to_class: Dict[int, int],
    ) -> int:

        # For first hypo just create a zeroth class
        if i == 0:
            return

        # Iterate over existing classes and return if hypo belongs to one of them
        for class_id in range(len(class_to_sample)):
            class_text_id = class_to_sample[class_id][0]
            forward_entailment = self._is_entailment[idx, class_text_id, i]
            backward_entailment = self._is_entailment[idx, i, class_text_id]
            if forward_entailment and backward_entailment:
                class_to_sample[class_id].append(i)
                sample_to_class[i] = class_id

                return

        # If none of the existing classes satisfy - create new one
        new_class_id = len(class_to_sample)
        sample_to_class[i] = new_class_id
        class_to_sample.append([i])
