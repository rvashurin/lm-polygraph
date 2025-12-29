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
        return [
            "original_question",
            "original_entropy",
            "original_samples",
            "original_sample_logprobs",
            "clarifications",
            "clarified_entropies",
            "clarified_samples",
            "clarified_sample_logprobs",
        ], ["input_texts"]

    def __init__(self):
        super().__init__()

    def __call__(
        self,
        dependencies: Dict[str, np.ndarray],
        texts: List[str],
        model,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        estimators = [
            SemanticEntropy(samples="unique"),
            MonteCarloSequenceEntropy(),
            MonteCarloNormalizedSequenceEntropy(),
        ]
        model = dependencies["model"]

        batch_original_questions = []
        batch_original_samples = []
        batch_original_logprobs = []
        batch_original_semantic_entropies = []
        batch_original_mcse_entropies = []
        batch_original_mcnse_entropies = []

        batch_clarifications = []
        batch_clarified_samples = []
        batch_clarified_logprobs = []
        batch_clarified_semantic_entropies = []
        batch_clarified_mcse_entropies = []
        batch_clarified_mcnse_entropies = []

        for request in dependencies["input_texts"]:
            # TriviaQA
            #instruction = [request.split("\n")[0]]
            #question = request.split("\n")[-2:]

            # CoQA
            question = request.split("\n")[-2:]
            instruction = ['Answer the following question as briefly as possible.']

            original_question = "\n".join(instruction + question)
            batch_original_questions.append(original_question) 

            original_output = estimate_uncertainty(
                model,
                estimator,
                input_text=original_question,
                output_stats=["sample_texts", "sample_log_probs"]
            )
            original_semantic_entropy = original_output.uncertainty['SemanticEntropy']
            original_mcse_entropy = original_output.uncertainty['MonteCarloSequenceEntropy']
            original_mcnse_entropy = original_output.uncertainty['MonteCarloNormalizedSequenceEntropy']
            original_samples = original_output.stats["sample_texts"]
            original_log_probs = original_output.stats["sample_log_probs"]

            batch_original_semantic_entropies.append(original_semantic_entropy)
            batch_original_mcse_entropies.append(original_mcse_entropy)
            batch_original_mcnse_entropies.append(original_mcnse_entropy)
            batch_original_samples.append(original_samples)
            batch_original_logprobs.append(original_log_probs)

            clarifications = []
            for _ in range(5):
                openai_chat = OpenAIChat(openai_model="gpt-5.1")
                prompt = CLARIFICATION_PROMPT.format(
                    original_question=original_question
                )
                clarified_question = openai_chat.ask(prompt)
                clarifications.append(clarified_question)
            batch_clarifications.append(clarifications)

            clarified_samples = []
            clarified_sample_log_probs = []
            clarified_semantic_entropies = []
            clarified_mcse_entropies = []
            clarified_mcnse_entropies = []
            for clarified_question in clarifications:
                clarified_output = estimate_uncertainty(
                    model,
                    estimator,
                    input_text=clarified_question,
                    output_stats=["sample_texts", "sample_log_probs"]
                )
                clarified_semantic_entropy = clarified_output.uncertainty['SemanticEntropy']
                clarified_semantic_entropies.append(clarified_semantic_entropy)

                clarified_mcse_entropy = clarified_output.uncertainty['MonteCarloSequenceEntropy']
                clarified_mcse_entropies.append(clarified_mcse_entropy)

                clarified_mcnse_entropy = clarified_output.uncertainty['MonteCarloNormalizedSequenceEntropy']
                clarified_mcnse_entropies.append(clarified_mcnse_entropy)

                clarified_samples = clarified_output.stats["sample_texts"]
                clarified_sample_log_probs = clarified_output.stats["sample_log_probs"]

                clarified_samples.append(clarified_samples)
                clarified_sample_log_probs.append(clarified_sample_log_probs)

            batch_clarified_samples.append(clarified_samples)
            batch_clarified_logprobs.append(clarified_sample_log_probs)
            batch_clarified_semantic_entropies.append(clarified_semantic_entropies)
            batch_clarified_mcse_entropies.append(clarified_mcse_entropies)
            batch_clarified_mcnse_entropies.append(clarified_mcnse_entropies)

        return {
            "original_question": batch_original_questions,
            "original_semantic_entropy": batch_original_semantic_entropies,
            "original_mcse_entropy": batch_original_mcse_entropies,
            "original_mcnse_entropy": batch_original_mcnse_entropies,
            "original_samples": batch_original_samples,
            "original_sample_logprobs": batch_original_logprobs,
            "clarifications": batch_clarifications,
            "clarified_entropies": batch_clarified_entropies,
            "clarified_mcse_entropies": batch_clarified_mcse_entropies,
            "clarified_mcnse_entropies": batch_clarified_mcnse_entropies,
            "clarified_samples": batch_clarified_samples,
            "clarified_sample_logprobs": batch_clarified_logprobs,
        }
