import numpy as np
from typing import Dict, List

from .stat_calculator import StatCalculator

from lm_polygraph.defaults.register_default_stat_calculators import (
    register_default_stat_calculators,
)
from lm_polygraph.estimators import (
    SemanticEntropy,
    MonteCarloSequenceEntropy,
    MonteCarloNormalizedSequenceEntropy,
)
from lm_polygraph.model_adapters.blackbox_model import BlackboxModel
from lm_polygraph.model_adapters.visual_whitebox_model import VisualWhiteboxModel
from lm_polygraph.model_adapters.whitebox_model import WhiteboxModel
from lm_polygraph.utils.builder_enviroment_stat_calculator import (
    BuilderEnvironmentStatCalculator,
)
from lm_polygraph.utils.estimate_uncertainty import estimate_uncertainty
from lm_polygraph.utils.factory_stat_calculator import FactoryStatCalculator
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
            "original_semantic_entropy",
            "original_semantic_entropy_normalized",
            "original_semantic_entropy_direct",
            "original_semantic_entropy_direct_normalized",
            "original_mcse_entropy",
            "original_mcnse_entropy",
            "original_samples",
            "original_sample_logprobs",
            "avg_semantic_entropy",
            "avg_semantic_entropy_normalized",
            "avg_semantic_entropy_direct",
            "avg_semantic_entropy_direct_normalized",
            "avg_mcse_entropy",
            "avg_mcnse_entropy",
            "clarifications",
            "clarified_semantic_entropies",
            "clarified_semantic_entropies_normalized",
            "clarified_semantic_entropies_direct",
            "clarified_semantic_entropies_direct_normalized",
            "clarified_mcse_entropies",
            "clarified_mcnse_entropies",
            "clarified_samples",
            "clarified_sample_logprobs",
        ], ["input_texts"]

    def __init__(self):
        super().__init__()
        self._avg_calculators = None
        self._avg_calc_model_type = None
        self._avg_required_stats = None

    @staticmethod
    def _infer_model_type(model) -> str:
        if isinstance(model, WhiteboxModel):
            return "Whitebox"
        if isinstance(model, VisualWhiteboxModel):
            return "VisualLM"
        if isinstance(model, BlackboxModel):
            return "Blackbox"
        return "Blackbox"

    @staticmethod
    def _log_avg(log_vals: List[float]) -> float:
        return float(np.logaddexp.reduce(np.array(log_vals, dtype=np.float64)) - np.log(len(log_vals)))

    def _ensure_avg_calculators(self, model, required_stats: List[str]):
        model_type = self._infer_model_type(model)
        required_stats_set = set(required_stats)
        if (
            self._avg_calculators is not None
            and self._avg_calc_model_type == model_type
            and self._avg_required_stats == required_stats_set
        ):
            return

        available_calculators = register_default_stat_calculators(
            model_type, model=model
        )

        calc_by_name = {sc.name: sc for sc in available_calculators}
        need_semantic_classes = "semantic_classes_entail" in required_stats_set
        semantic_matrix_stats = {
            "semantic_matrix_entail",
            "semantic_matrix_contra",
            "semantic_matrix_classes",
            "semantic_matrix_entail_logits",
            "semantic_matrix_contra_logits",
            "entailment_id",
        }
        need_semantic_matrix = need_semantic_classes or bool(
            semantic_matrix_stats.intersection(required_stats_set)
        )

        ordered_calculators = []
        if need_semantic_matrix:
            ordered_calculators.append(calc_by_name["SemanticMatrixCalculator"])
        if need_semantic_classes:
            ordered_calculators.append(calc_by_name["SemanticClassesCalculator"])

        factory = FactoryStatCalculator(BuilderEnvironmentStatCalculator(model))
        self._avg_calculators = factory(ordered_calculators)
        self._avg_calc_model_type = model_type
        self._avg_required_stats = required_stats_set

    def _build_averaged_stats(
        self,
        clarified_samples: List[List[str]],
        clarified_log_probs: List[List[float]],
        clarified_log_likelihoods: List[List[List[float]]],
        clarified_tokens: List[List[List[int]]],
    ) -> Dict[str, np.ndarray]:
        num_clarifications = len(clarified_samples)
        if num_clarifications == 0:
            raise ValueError("No clarifications provided for averaging.")

        union_samples = []
        union_index = {}
        per_clar_maps = []

        for samples, log_probs, log_likelihoods, tokens in zip(
            clarified_samples,
            clarified_log_probs,
            clarified_log_likelihoods,
            clarified_tokens,
        ):
            sample_map = {}
            for i, sample in enumerate(samples):
                if sample in sample_map:
                    continue
                sample_map[sample] = {
                    "log_prob": log_probs[i],
                    "log_likelihoods": log_likelihoods[i],
                    "tokens": tokens[i],
                }
                if sample not in union_index:
                    union_index[sample] = len(union_samples)
                    union_samples.append(sample)
            per_clar_maps.append(sample_map)

        avg_log_probs = []
        avg_log_likelihoods = []
        avg_tokens = []

        for sample in union_samples:
            canonical_tokens = None
            for sample_map in per_clar_maps:
                data = sample_map.get(sample)
                if data is not None:
                    canonical_tokens = data["tokens"]
                    break
            if canonical_tokens is None:
                raise ValueError("Missing tokens for averaged sample.")

            avg_tokens.append(canonical_tokens)

            log_vals = [
                sample_map[sample]["log_prob"] if sample in sample_map else -np.inf
                for sample_map in per_clar_maps
            ]
            avg_log_probs.append(self._log_avg(log_vals))

            token_log_likelihoods = []
            for t in range(len(canonical_tokens)):
                token_log_vals = []
                for sample_map in per_clar_maps:
                    data = sample_map.get(sample)
                    if data is None:
                        token_log_vals.append(-np.inf)
                        continue
                    ll = data["log_likelihoods"]
                    token_log_vals.append(ll[t] if t < len(ll) else -np.inf)
                token_log_likelihoods.append(self._log_avg(token_log_vals))
            avg_log_likelihoods.append(token_log_likelihoods)

        return {
            "sample_texts": [union_samples],
            "sample_log_probs": [avg_log_probs],
            "sample_log_likelihoods": [avg_log_likelihoods],
            "sample_tokens": [avg_tokens],
        }

    def _enrich_avg_stats(
        self,
        avg_stats: Dict[str, np.ndarray],
        input_text: str,
        model,
        max_new_tokens: int,
        required_stats: List[str],
    ) -> Dict[str, np.ndarray]:
        self._ensure_avg_calculators(model, required_stats)
        for stat_calculator in self._avg_calculators:
            new_stats = stat_calculator(
                avg_stats, [input_text], model, max_new_tokens
            )
            for stat, stat_value in new_stats.items():
                if stat in avg_stats:
                    continue
                avg_stats[stat] = stat_value
        return avg_stats

    def __call__(
        self,
        dependencies: Dict[str, np.ndarray],
        texts: List[str],
        model,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        estimators = [
            SemanticEntropy(samples="unique"),
            SemanticEntropy(samples="unique", normalize=True),
            SemanticEntropy(samples="unique", estimator="direct"),
            SemanticEntropy(samples="unique", estimator="direct", normalize=True),
            MonteCarloSequenceEntropy(),
            MonteCarloNormalizedSequenceEntropy(),
        ]
        required_stats = sorted(
            {
                dep
                for estimator in estimators
                for dep in estimator.stats_dependencies
            }
        )
        model = dependencies["model"]

        batch_original_questions = []
        batch_original_samples = []
        batch_original_logprobs = []
        batch_original_semantic_entropies = []
        batch_original_semantic_entropies_normalized = []
        batch_original_semantic_entropies_direct = []
        batch_original_semantic_entropies_direct_normalized = []
        batch_original_mcse_entropies = []
        batch_original_mcnse_entropies = []

        batch_avg_semantic_entropies = []
        batch_avg_semantic_entropies_normalized = []
        batch_avg_semantic_entropies_direct = []
        batch_avg_semantic_entropies_direct_normalized = []
        batch_avg_mcse_entropies = []
        batch_avg_mcnse_entropies = []

        batch_clarifications = []
        batch_clarified_samples = []
        batch_clarified_logprobs = []
        batch_clarified_semantic_entropies = []
        batch_clarified_semantic_entropies_normalized = []
        batch_clarified_semantic_entropies_direct = []
        batch_clarified_semantic_entropies_direct_normalized = []
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
                estimators,
                input_text=original_question,
                output_stats=[
                    "sample_texts",
                    "sample_log_probs",
                    "sample_log_likelihoods",
                    "sample_tokens",
                ],
            )
            original_semantic_entropy = original_output.uncertainty['SemanticEntropy']
            original_semantic_entropy_normalized = original_output.uncertainty['SemanticEntropyNormalized']
            original_semantic_entropy_direct = original_output.uncertainty['SemanticEntropyDirect']
            original_semantic_entropy_direct_normalized = original_output.uncertainty['SemanticEntropyDirectNormalized']
            original_mcse_entropy = original_output.uncertainty['MonteCarloSequenceEntropy']
            original_mcnse_entropy = original_output.uncertainty['MonteCarloNormalizedSequenceEntropy']
            original_samples = original_output.stats["sample_texts"][0]
            original_log_probs = original_output.stats["sample_log_probs"][0]

            batch_original_semantic_entropies.append(original_semantic_entropy)
            batch_original_semantic_entropies_normalized.append(original_semantic_entropy_normalized)
            batch_original_semantic_entropies_direct.append(original_semantic_entropy_direct)
            batch_original_semantic_entropies_direct_normalized.append(original_semantic_entropy_direct_normalized)
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
            clarified_sample_log_likelihoods = []
            clarified_sample_tokens = []
            clarified_semantic_entropies = []
            clarified_semantic_entropies_normalized = []
            clarified_semantic_entropies_direct = []
            clarified_semantic_entropies_direct_normalized = []
            clarified_mcse_entropies = []
            clarified_mcnse_entropies = []
            for clarified_question in clarifications:
                clarified_output = estimate_uncertainty(
                    model,
                    estimators,
                    input_text=clarified_question,
                    output_stats=[
                        "sample_texts",
                        "sample_log_probs",
                        "sample_log_likelihoods",
                        "sample_tokens",
                    ],
                )
                clarified_semantic_entropy = clarified_output.uncertainty['SemanticEntropy']
                clarified_semantic_entropies.append(clarified_semantic_entropy)

                clarified_semantic_entropy_normalized = clarified_output.uncertainty['SemanticEntropyNormalized']
                clarified_semantic_entropies_normalized.append(clarified_semantic_entropy_normalized)

                clarified_semantic_entropy_direct = clarified_output.uncertainty['SemanticEntropyDirect']
                clarified_semantic_entropies_direct.append(clarified_semantic_entropy_direct)

                clarified_semantic_entropy_direct_normalized = clarified_output.uncertainty['SemanticEntropyDirectNormalized']
                clarified_semantic_entropies_direct_normalized.append(clarified_semantic_entropy_direct_normalized)

                clarified_mcse_entropy = clarified_output.uncertainty['MonteCarloSequenceEntropy']
                clarified_mcse_entropies.append(clarified_mcse_entropy)

                clarified_mcnse_entropy = clarified_output.uncertainty['MonteCarloNormalizedSequenceEntropy']
                clarified_mcnse_entropies.append(clarified_mcnse_entropy)

                clarified_samples.append(clarified_output.stats["sample_texts"][0])
                clarified_sample_log_probs.append(clarified_output.stats["sample_log_probs"][0])
                if "sample_log_likelihoods" not in clarified_output.stats:
                    raise KeyError("sample_log_likelihoods missing in clarification stats.")
                if "sample_tokens" not in clarified_output.stats:
                    raise KeyError("sample_tokens missing in clarification stats.")
                clarified_sample_log_likelihoods.append(
                    clarified_output.stats["sample_log_likelihoods"][0]
                )
                clarified_sample_tokens.append(
                    clarified_output.stats["sample_tokens"][0]
                )

            avg_stats = self._build_averaged_stats(
                clarified_samples,
                clarified_sample_log_probs,
                clarified_sample_log_likelihoods,
                clarified_sample_tokens,
            )
            avg_stats["model"] = model
            avg_stats = self._enrich_avg_stats(
                avg_stats,
                original_question,
                model,
                max_new_tokens,
                required_stats,
            )
            avg_entropies = {}
            for estimator in estimators:
                values = estimator(avg_stats)
                if not isinstance(values, np.ndarray):
                    values = np.asarray(values)
                avg_entropies[str(estimator)] = float(values[0])

            batch_clarified_samples.append(clarified_samples)
            batch_clarified_logprobs.append(clarified_sample_log_probs)
            batch_clarified_semantic_entropies.append(clarified_semantic_entropies)
            batch_clarified_semantic_entropies_normalized.append(clarified_semantic_entropies_normalized)
            batch_clarified_semantic_entropies_direct.append(clarified_semantic_entropies_direct)
            batch_clarified_semantic_entropies_direct_normalized.append(clarified_semantic_entropies_direct_normalized)
            batch_clarified_mcse_entropies.append(clarified_mcse_entropies)
            batch_clarified_mcnse_entropies.append(clarified_mcnse_entropies)

            batch_avg_semantic_entropies.append(avg_entropies["SemanticEntropy"])
            batch_avg_semantic_entropies_normalized.append(
                avg_entropies["SemanticEntropyNormalized"]
            )
            batch_avg_semantic_entropies_direct.append(
                avg_entropies["SemanticEntropyDirect"]
            )
            batch_avg_semantic_entropies_direct_normalized.append(
                avg_entropies["SemanticEntropyDirectNormalized"]
            )
            batch_avg_mcse_entropies.append(
                avg_entropies["MonteCarloSequenceEntropy"]
            )
            batch_avg_mcnse_entropies.append(
                avg_entropies["MonteCarloNormalizedSequenceEntropy"]
            )

        return {
            "original_question": batch_original_questions,
            "original_semantic_entropy": batch_original_semantic_entropies,
            "original_semantic_entropy_normalized": batch_original_semantic_entropies_normalized,
            "original_semantic_entropy_direct": batch_original_semantic_entropies_direct,
            "original_semantic_entropy_direct_normalized": batch_original_semantic_entropies_direct_normalized,
            "original_mcse_entropy": batch_original_mcse_entropies,
            "original_mcnse_entropy": batch_original_mcnse_entropies,
            "original_samples": batch_original_samples,
            "original_sample_logprobs": batch_original_logprobs,
            "avg_semantic_entropy": batch_avg_semantic_entropies,
            "avg_semantic_entropy_normalized": batch_avg_semantic_entropies_normalized,
            "avg_semantic_entropy_direct": batch_avg_semantic_entropies_direct,
            "avg_semantic_entropy_direct_normalized": batch_avg_semantic_entropies_direct_normalized,
            "avg_mcse_entropy": batch_avg_mcse_entropies,
            "avg_mcnse_entropy": batch_avg_mcnse_entropies,
            "clarifications": batch_clarifications,
            "clarified_semantic_entropies": batch_clarified_semantic_entropies,
            "clarified_semantic_entropies_normalized": batch_clarified_semantic_entropies_normalized,
            "clarified_semantic_entropies_direct": batch_clarified_semantic_entropies_direct,
            "clarified_semantic_entropies_direct_normalized": batch_clarified_semantic_entropies_direct_normalized,
            "clarified_mcse_entropies": batch_clarified_mcse_entropies,
            "clarified_mcnse_entropies": batch_clarified_mcnse_entropies,
            "clarified_samples": batch_clarified_samples,
            "clarified_sample_logprobs": batch_clarified_logprobs,
        }
