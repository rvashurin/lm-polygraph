import numpy as np
from typing import Dict, List
from .stat_calculator import StatCalculator

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
If there are any ambiguities, you need to remove ambiguities by adding some clarifications to
the question. Each clarification is an additional condition or explanations to the concept in
the question that resolve its ambiguity. If question contains unresolved references come up with a context (a story, previous dialogue, whatever) that would resolve them.
- You are only allowed to add conditions or explanations, and you cannot change the
original intent or semantics of the question.
- The conditions and explanations must be ground to real-word facts.
If there is no ambiguities, you only need to output the original question as it is.
Output only the final question after adding clarifications (if any).

Original Question: {original_question}
Question after adding clarification:
"""

class SpecificationCalculator(StatCalculator):
    @staticmethod
    def meta_info():
        # outputs, dependencies
        return ["clarifications", "clarified_entropies", "original_entropy", "original_question"], ["input_texts"]

    def __init__(self):
        super().__init__()

    def __call__(
        self,
        dependencies: Dict[str, np.ndarray],
        texts: List[str],
        model,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        estimator = SemanticEntropy(samples="unique")
        model = dependencies["model"]

        batch_clarifications = []
        batch_clarified_entropies = []
        batch_original_entropies = []
        batch_original_questions = []

        for request in dependencies["input_texts"]:
            # TriviaQA
            #instruction = [request.split("\n")[0]]
            #question = request.split("\n")[-2:]

            # CoQA
            question = request.split("\n")[-2:]
            instruction = ['Answer the following question as briefly as possible.']

            original_question = "\n".join(instruction + question)
            batch_original_questions.append(original_question) 

            original_semantic_entropy = estimate_uncertainty(
                model,
                estimator,
                input_text=original_question
            ).uncertainty
            batch_original_entropies.append(original_semantic_entropy)

            clarifications = []
            for _ in range(3):
                openai_chat = OpenAIChat(openai_model="gpt-5.1")
                prompt = CLARIFICATION_PROMPT.format(
                    original_question=original_question
                )
                clarified_question = openai_chat.ask(prompt)
                clarifications.append(clarified_question)
            batch_clarifications.append(clarifications)

            clarified_entropies = []
            for clarified_question in clarifications:
                clarified_entropy = estimate_uncertainty(
                    model,
                    estimator,
                    input_text=clarified_question
                ).uncertainty
                clarified_entropies.append(clarified_entropy)
            batch_clarified_entropies.append(clarified_entropies)

        return {
            "clarifications": np.array(batch_clarifications),
            "clarified_entropies": np.array(batch_clarified_entropies),
            "original_entropy": np.array(batch_original_entropies),
            "original_question": np.array(batch_original_questions),
        }
