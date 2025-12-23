import numpy as np

from typing import List, Dict, Optional

from .estimator import Estimator

from lm_polygraph.utils.estimate_uncertainty import estimate_uncertainty
from lm_polygraph.estimators import SemanticEntropy
from lm_polygraph.utils.openai_chat import OpenAIChat

CLARIFICATION_PROMPT="""
In this task, you will receive a question that may contain ambiguities. First analyze the
following aspects to find if there is any ambiguities according to the real-world facts:
- Unresolved references to entities or people ("it", "he", "they" etc without referred entity explicitly mentioned elsewhere)
- Entities, objects, or events has multiple references or interpretations
- Unclear timestamps
- Unclear locations
- Unclear answer types (e.g., "When" refers to "which year or what date", and "Who" refers to "
which person or which team")
If there is any ambiguities, you need to remove ambiguities by adding some clarifications to
the question. Each clarification is an additional condition or explanations to the concept in
the question that resolve its ambiguity, or additional context that resolves references.
- You are only allowed to add conditions or explanations, and you cannot change the
original intent or semantics of the question.
- The conditions and explanations must be ground to real-word facts.
If there is no ambiguities, you only need to output the original question as it is.
Output only the final question after adding clarifications (if any).

Original Question: {original_question}
Question after adding clarification:
"""

class SpecificationUncertainty(Estimator):

    def __init__(
        self, verbose: bool = False, n_clarifications: int = 3
    ):
        deps = ["input_texts"]
        super().__init__(deps, "sequence")
        self.verbose = verbose
        self.n_clarifications = n_clarifications

    def __str__(self):
        return f"SpecificationUncertainty_n{self.n_clarifications}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        estimator = SemanticEntropy(samples="unique")
        model = stats["model"]
        spec_uncertainties = []

        for request in stats["input_texts"]:
            # TriviaQA
            #instruction = [request.split("\n")[0]]
            #question = request.split("\n")[-2:]
            
            # CoQA
            question = request.split("\n")[-2:]
            instruction = ['Answer the following question as briefly as possible.']

            original_question = "\n".join(instruction + question)

            original_semantic_entropy = estimate_uncertainty(
                model,
                estimator,
                input_text=original_question
            ).uncertainty

            clarifications = []
            for _ in range(self.n_clarifications):
                openai_chat = OpenAIChat(openai_model="gpt-5.1")
                prompt = CLARIFICATION_PROMPT.format(
                    original_question=original_question
                )
                clarified_question = openai_chat.ask(prompt)
                clarifications.append(clarified_question)

            clarified_entropies = []
            for clarified_question in clarifications:
                clarified_entropy = estimate_uncertainty(
                    model,
                    estimator,
                    input_text=clarified_question
                ).uncertainty
                clarified_entropies.append(clarified_entropy)

            spec_uncertainties.append(
                original_semantic_entropy - np.mean(clarified_entropies)
            )

        return np.array(spec_uncertainties)
