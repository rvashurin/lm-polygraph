import re
import numpy as np
from typing import Dict, List, Union

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

CLARIFYING_QUESTION_PROMPT = """
You will receive a question that may contain ambiguities. Generate ONE clarifying question that would resolve the most important ambiguity in the original question.

The clarifying question should:
- Target the single most important ambiguity
- Be short and direct
- Have multiple plausible answers that would each lead to a different correct answer for the original question

Original Question: {original_question}
Clarifying question:
"""

CLARIFYING_ANSWERS_PROMPT = """
You will receive an original question and a clarifying question about it. Generate {k} different plausible answers to the clarifying question. Each answer should represent a distinct interpretation or context that is grounded in real-world facts.

Requirements:
- Each answer must be concise (one sentence or less)
- Each answer must be clearly distinct from the others
- Output exactly {k} answers, one per line, numbered like: 1. ... 2. ... etc.

Original Question: {original_question}
Clarifying Question: {clarifying_question}
Answers:
"""


class DialogueSpecificationCalculator(StatCalculator):
    @staticmethod
    def meta_info():
        return [
            "dialogue_original_question",
            "dialogue_clarifying_question",
            "dialogue_clarifying_answers",
            "dialogue_samples",
            "dialogue_sample_logprobs",
            "dialogue_semantic_entropies",
            "dialogue_semantic_entropies_normalized",
            "dialogue_semantic_entropies_direct",
            "dialogue_semantic_entropies_direct_normalized",
            "dialogue_mcse_entropies",
            "dialogue_mcnse_entropies",
            "avg_dialogue_semantic_entropy",
            "avg_dialogue_semantic_entropy_normalized",
            "avg_dialogue_semantic_entropy_direct",
            "avg_dialogue_semantic_entropy_direct_normalized",
            "avg_dialogue_mcse_entropy",
            "avg_dialogue_mcnse_entropy",
        ], ["input_texts"]

    def __init__(self, num_answers: int = 5, openai_model: str = "gpt-4o"):
        super().__init__()
        self.num_answers = num_answers
        self.openai_model = openai_model
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
        return float(
            np.logaddexp.reduce(np.array(log_vals, dtype=np.float64)) - np.log(len(log_vals))
        )

    def _parse_numbered_answers(self, text: str) -> List[str]:
        answers = []
        for line in text.strip().split("\n"):
            line = line.strip()
            match = re.match(r"^[\d]+[.)]\s*(.+)", line)
            if match:
                answers.append(match.group(1).strip())
            elif line.startswith("- "):
                answers.append(line[2:].strip())
            elif line:
                answers.append(line)
        return answers[: self.num_answers]

    def _build_dialogue_input(
        self,
        original_question: str,
        clarifying_question: str,
        answer: str,
        model,
    ) -> Union[str, List[Dict[str, str]]]:
        """Return chat messages for instruct models, flat string otherwise."""
        if isinstance(model, WhiteboxModel) and getattr(model, "instruct", False):
            return [
                {"role": "user", "content": original_question},
                {"role": "assistant", "content": clarifying_question},
                {"role": "user", "content": answer},
            ]
        return f"{original_question}\n{clarifying_question}\n{answer}"

    def _ensure_avg_calculators(self, model, required_stats: List[str]):
        model_type = self._infer_model_type(model)
        required_stats_set = set(required_stats)
        if (
            self._avg_calculators is not None
            and self._avg_calc_model_type == model_type
            and self._avg_required_stats == required_stats_set
        ):
            return

        available_calculators = register_default_stat_calculators(model_type, model=model)
        calc_by_name = {sc.name: sc for sc in available_calculators}

        semantic_matrix_stats = {
            "semantic_matrix_entail",
            "semantic_matrix_contra",
            "semantic_matrix_classes",
            "semantic_matrix_entail_logits",
            "semantic_matrix_contra_logits",
            "entailment_id",
        }
        need_semantic_classes = "semantic_classes_entail" in required_stats_set
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
        path_samples: List[List[str]],
        path_log_probs: List[List[float]],
        path_log_likelihoods: List[List[List[float]]],
        path_tokens: List[List[List[int]]],
    ) -> Dict[str, np.ndarray]:
        union_samples = []
        union_index: Dict[str, int] = {}
        per_path_maps = []

        for samples, log_probs, log_likelihoods, tokens in zip(
            path_samples, path_log_probs, path_log_likelihoods, path_tokens
        ):
            sample_map: Dict[str, dict] = {}
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
            per_path_maps.append(sample_map)

        avg_log_probs = []
        avg_log_likelihoods = []
        avg_tokens = []

        for sample in union_samples:
            canonical_tokens = None
            for sample_map in per_path_maps:
                data = sample_map.get(sample)
                if data is not None:
                    canonical_tokens = data["tokens"]
                    break
            if canonical_tokens is None:
                raise ValueError("Missing tokens for averaged sample.")
            avg_tokens.append(canonical_tokens)

            log_vals = [
                sample_map[sample]["log_prob"] if sample in sample_map else -np.inf
                for sample_map in per_path_maps
            ]
            avg_log_probs.append(self._log_avg(log_vals))

            token_log_likelihoods = []
            for t in range(len(canonical_tokens)):
                token_log_vals = []
                for sample_map in per_path_maps:
                    data = sample_map.get(sample)
                    token_log_vals.append(
                        data["log_likelihoods"][t]
                        if data is not None and t < len(data["log_likelihoods"])
                        else -np.inf
                    )
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
            new_stats = stat_calculator(avg_stats, [input_text], model, max_new_tokens)
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
            {dep for estimator in estimators for dep in estimator.stats_dependencies}
        )
        model = dependencies["model"]

        batch_original_questions = []
        batch_clarifying_questions = []
        batch_clarifying_answers = []
        batch_dialogue_samples = []
        batch_dialogue_logprobs = []
        batch_dialogue_semantic_entropies = []
        batch_dialogue_semantic_entropies_normalized = []
        batch_dialogue_semantic_entropies_direct = []
        batch_dialogue_semantic_entropies_direct_normalized = []
        batch_dialogue_mcse_entropies = []
        batch_dialogue_mcnse_entropies = []
        batch_avg_semantic_entropies = []
        batch_avg_semantic_entropies_normalized = []
        batch_avg_semantic_entropies_direct = []
        batch_avg_semantic_entropies_direct_normalized = []
        batch_avg_mcse_entropies = []
        batch_avg_mcnse_entropies = []

        for request in dependencies["input_texts"]:
            question = request.split("\n")[-2:]
            instruction = ["Answer the following question as briefly as possible."]
            original_question = "\n".join(instruction + question)
            batch_original_questions.append(original_question)

            openai_chat = OpenAIChat(openai_model=self.openai_model)

            cq_prompt = CLARIFYING_QUESTION_PROMPT.format(original_question=original_question)
            clarifying_question = openai_chat.ask(cq_prompt).strip()
            batch_clarifying_questions.append(clarifying_question)

            answers_prompt = CLARIFYING_ANSWERS_PROMPT.format(
                original_question=original_question,
                clarifying_question=clarifying_question,
                k=self.num_answers,
            )
            answers_text = openai_chat.ask(answers_prompt)
            clarifying_answers = self._parse_numbered_answers(answers_text)
            while len(clarifying_answers) < self.num_answers:
                clarifying_answers.append(clarifying_answers[-1] if clarifying_answers else "N/A")
            clarifying_answers = clarifying_answers[: self.num_answers]
            batch_clarifying_answers.append(clarifying_answers)

            path_samples = []
            path_log_probs = []
            path_log_likelihoods = []
            path_tokens = []
            path_semantic_entropies = []
            path_semantic_entropies_normalized = []
            path_semantic_entropies_direct = []
            path_semantic_entropies_direct_normalized = []
            path_mcse_entropies = []
            path_mcnse_entropies = []

            for answer in clarifying_answers:
                dialogue_input = self._build_dialogue_input(
                    original_question, clarifying_question, answer, model
                )
                path_output = estimate_uncertainty(
                    model,
                    estimators,
                    input_text=dialogue_input,
                    output_stats=[
                        "sample_texts",
                        "sample_log_probs",
                        "sample_log_likelihoods",
                        "sample_tokens",
                    ],
                )
                path_semantic_entropies.append(path_output.uncertainty["SemanticEntropy"])
                path_semantic_entropies_normalized.append(
                    path_output.uncertainty["SemanticEntropyNormalized"]
                )
                path_semantic_entropies_direct.append(
                    path_output.uncertainty["SemanticEntropyDirect"]
                )
                path_semantic_entropies_direct_normalized.append(
                    path_output.uncertainty["SemanticEntropyDirectNormalized"]
                )
                path_mcse_entropies.append(path_output.uncertainty["MonteCarloSequenceEntropy"])
                path_mcnse_entropies.append(
                    path_output.uncertainty["MonteCarloNormalizedSequenceEntropy"]
                )
                path_samples.append(path_output.stats["sample_texts"][0])
                path_log_probs.append(path_output.stats["sample_log_probs"][0])
                path_log_likelihoods.append(path_output.stats["sample_log_likelihoods"][0])
                path_tokens.append(path_output.stats["sample_tokens"][0])

            avg_stats = self._build_averaged_stats(
                path_samples, path_log_probs, path_log_likelihoods, path_tokens
            )
            avg_stats["model"] = model
            avg_stats = self._enrich_avg_stats(
                avg_stats, original_question, model, max_new_tokens, required_stats
            )

            avg_entropies = {}
            for estimator in estimators:
                values = estimator(avg_stats)
                if not isinstance(values, np.ndarray):
                    values = np.asarray(values)
                avg_entropies[str(estimator)] = float(values[0])

            batch_dialogue_samples.append(path_samples)
            batch_dialogue_logprobs.append(path_log_probs)
            batch_dialogue_semantic_entropies.append(path_semantic_entropies)
            batch_dialogue_semantic_entropies_normalized.append(path_semantic_entropies_normalized)
            batch_dialogue_semantic_entropies_direct.append(path_semantic_entropies_direct)
            batch_dialogue_semantic_entropies_direct_normalized.append(
                path_semantic_entropies_direct_normalized
            )
            batch_dialogue_mcse_entropies.append(path_mcse_entropies)
            batch_dialogue_mcnse_entropies.append(path_mcnse_entropies)
            batch_avg_semantic_entropies.append(avg_entropies["SemanticEntropy"])
            batch_avg_semantic_entropies_normalized.append(
                avg_entropies["SemanticEntropyNormalized"]
            )
            batch_avg_semantic_entropies_direct.append(avg_entropies["SemanticEntropyDirect"])
            batch_avg_semantic_entropies_direct_normalized.append(
                avg_entropies["SemanticEntropyDirectNormalized"]
            )
            batch_avg_mcse_entropies.append(avg_entropies["MonteCarloSequenceEntropy"])
            batch_avg_mcnse_entropies.append(avg_entropies["MonteCarloNormalizedSequenceEntropy"])

        return {
            "dialogue_original_question": batch_original_questions,
            "dialogue_clarifying_question": batch_clarifying_questions,
            "dialogue_clarifying_answers": batch_clarifying_answers,
            "dialogue_samples": batch_dialogue_samples,
            "dialogue_sample_logprobs": batch_dialogue_logprobs,
            "dialogue_semantic_entropies": batch_dialogue_semantic_entropies,
            "dialogue_semantic_entropies_normalized": batch_dialogue_semantic_entropies_normalized,
            "dialogue_semantic_entropies_direct": batch_dialogue_semantic_entropies_direct,
            "dialogue_semantic_entropies_direct_normalized": batch_dialogue_semantic_entropies_direct_normalized,
            "dialogue_mcse_entropies": batch_dialogue_mcse_entropies,
            "dialogue_mcnse_entropies": batch_dialogue_mcnse_entropies,
            "avg_dialogue_semantic_entropy": batch_avg_semantic_entropies,
            "avg_dialogue_semantic_entropy_normalized": batch_avg_semantic_entropies_normalized,
            "avg_dialogue_semantic_entropy_direct": batch_avg_semantic_entropies_direct,
            "avg_dialogue_semantic_entropy_direct_normalized": batch_avg_semantic_entropies_direct_normalized,
            "avg_dialogue_mcse_entropy": batch_avg_mcse_entropies,
            "avg_dialogue_mcnse_entropy": batch_avg_mcnse_entropies,
        }
