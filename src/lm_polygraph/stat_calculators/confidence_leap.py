import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel


DEFAULT_TRIGGERS = ("Wait", "Alternatively", "Hmm", "Perhaps", "Maybe", "But", "However")
OPTION_LABELS = ("A", "B", "C", "D")


class ConfidenceLeapCalculator(StatCalculator):
    """
    Computes confidence dynamics over reasoning chunks for A-D multiple-choice
    generations. This POC is whitebox-only and probes hardcoded option labels.
    """

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        return [
            "confidence_leap_chunks",
            "confidence_leap_option_probs",
            "confidence_leap_predictions",
            "confidence_leap_metrics",
            "confidence_leap_status",
        ], ["greedy_tokens"]

    def __init__(
        self,
        max_chunks: int = 40,
        reasoning_open: str = "<think>\n",
        reasoning_close: str = "\n</think>\n\n",
        answer_prefix: str = "",
        triggers: Optional[List[str]] = None,
    ):
        super().__init__()
        self.max_chunks = max_chunks
        self.reasoning_open = reasoning_open
        self.reasoning_close = reasoning_close
        self.answer_prefix = answer_prefix
        self.triggers = tuple(triggers) if triggers is not None else DEFAULT_TRIGGERS

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        if not isinstance(model, WhiteboxModel):
            raise ValueError("ConfidenceLeapCalculator supports WhiteboxModel only")
        if not hasattr(model, "tokenize_assistant_continuations"):
            raise ValueError(
                "ConfidenceLeapCalculator requires a model with "
                "tokenize_assistant_continuations"
            )

        option_token_ids = _option_token_ids(model.tokenizer)
        if any(len(ids) == 0 for ids in option_token_ids.values()):
            raise ValueError("Could not resolve single-token ids for all A-D options")

        batch_chunks = []
        batch_option_probs = []
        batch_predictions = []
        batch_metrics = []
        batch_status = []

        input_texts = dependencies.get("input_texts", texts)
        for input_text, generated_tokens in zip(input_texts, dependencies["greedy_tokens"]):
            generated_text = model.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=False,
            )
            reasoning = _extract_reasoning(generated_text)
            if not reasoning:
                batch_chunks.append([])
                batch_option_probs.append([])
                batch_predictions.append([])
                batch_metrics.append(_empty_metrics())
                batch_status.append("no_reasoning")
                continue

            chunks = _split_reasoning(reasoning, self.triggers)
            if not chunks:
                batch_chunks.append([])
                batch_option_probs.append([])
                batch_predictions.append([])
                batch_metrics.append(_empty_metrics())
                batch_status.append("no_chunks")
                continue
            if len(chunks) > self.max_chunks:
                batch_chunks.append(chunks)
                batch_option_probs.append([])
                batch_predictions.append([])
                batch_metrics.append(_empty_metrics(num_chunks=len(chunks)))
                batch_status.append("too_many_chunks")
                continue

            prefixes = ["\n\n".join(chunks[: i + 1]) for i in range(len(chunks))]
            assistant_prefixes = [
                f"{self.reasoning_open}{prefix}{self.reasoning_close}{self.answer_prefix}"
                for prefix in prefixes
            ]
            option_probs = self._score_option_probs(
                model,
                input_text,
                assistant_prefixes,
                option_token_ids,
            )
            predictions = [
                max(probs, key=probs.get) if probs else None for probs in option_probs
            ]
            metrics = _compute_metrics(option_probs, predictions)

            batch_chunks.append(chunks)
            batch_option_probs.append(option_probs)
            batch_predictions.append(predictions)
            batch_metrics.append(metrics)
            batch_status.append("ok")

        return {
            "confidence_leap_chunks": batch_chunks,
            "confidence_leap_option_probs": batch_option_probs,
            "confidence_leap_predictions": batch_predictions,
            "confidence_leap_metrics": batch_metrics,
            "confidence_leap_status": batch_status,
        }

    def _score_option_probs(
        self,
        model: WhiteboxModel,
        input_text,
        assistant_prefixes: List[str],
        option_token_ids: Dict[str, List[int]],
    ) -> List[Dict[str, float]]:
        batch = model.tokenize_assistant_continuations(
            [input_text] * len(assistant_prefixes),
            assistant_prefixes,
        )
        batch = {key: value.to(model.device()) for key, value in batch.items()}
        with torch.no_grad():
            out = model(**batch)
            probs = torch.softmax(out.logits[:, -1, :].float(), dim=-1)

        option_probs = []
        for row in probs:
            row_probs = {}
            for label, token_ids in option_token_ids.items():
                row_probs[label] = max(float(row[token_id].item()) for token_id in token_ids)
            option_probs.append(row_probs)
        return option_probs


def _option_token_ids(tokenizer) -> Dict[str, List[int]]:
    label_to_ids = {}
    for label in OPTION_LABELS:
        ids = []
        for candidate in (label, f" {label}"):
            encoded = tokenizer.encode(candidate, add_special_tokens=False)
            if len(encoded) == 1:
                ids.append(encoded[0])
        label_to_ids[label] = list(dict.fromkeys(ids))
    return label_to_ids


def _extract_reasoning(text: str) -> str:
    if not text:
        return ""

    think_match = re.search(r"<think>\s*(.*?)\s*</think>", text, flags=re.DOTALL)
    if think_match:
        return think_match.group(1).strip()

    harmony_match = re.search(
        r"<\|start\|>analysis<\|message\|>(.*?)(?:<\|end\|>|$)",
        text,
        flags=re.DOTALL,
    )
    if harmony_match:
        return harmony_match.group(1).strip()

    close_pos = text.find("</think>")
    if close_pos != -1:
        return text[:close_pos].replace("<think>", "").strip()

    return text.strip()


def _split_reasoning(reasoning: str, triggers: Tuple[str, ...]) -> List[str]:
    if not reasoning:
        return []

    trigger_pattern = re.compile(
        r"^\s*(?:" + "|".join(re.escape(trigger) for trigger in triggers) + r")\b",
        flags=re.IGNORECASE,
    )
    paragraphs = re.split(r"\n\s*\n", reasoning.strip())
    if not paragraphs or not paragraphs[0].strip():
        return []

    chunks = []
    current = [paragraphs[0]]
    for paragraph in paragraphs[1:]:
        stripped = paragraph.strip()
        if not stripped:
            continue
        if trigger_pattern.match(stripped):
            chunks.append("\n\n".join(current).strip())
            current = [paragraph]
        else:
            current.append(paragraph)
    if current:
        chunks.append("\n\n".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def _compute_metrics(
    option_probs: List[Dict[str, float]],
    predictions: List[Optional[str]],
) -> Dict[str, Optional[object]]:
    top_confidences = [
        float(probs[predictions[i]]) if probs and predictions[i] is not None else np.nan
        for i, probs in enumerate(option_probs)
    ]
    finite_confidences = [conf for conf in top_confidences if np.isfinite(conf)]

    max_jump = None
    max_drop = None
    for i in range(1, len(option_probs)):
        prev = option_probs[i - 1]
        cur = option_probs[i]
        for label in OPTION_LABELS:
            delta = float(cur.get(label, 0.0) - prev.get(label, 0.0))
            if max_jump is None or delta > max_jump["delta"]:
                max_jump = {
                    "chunk_index": i,
                    "option": label,
                    "delta": delta,
                    "prob_before": float(prev.get(label, 0.0)),
                    "prob_after": float(cur.get(label, 0.0)),
                }
            if max_drop is None or delta < max_drop["delta"]:
                max_drop = {
                    "chunk_index": i,
                    "option": label,
                    "delta": delta,
                    "prob_before": float(prev.get(label, 0.0)),
                    "prob_after": float(cur.get(label, 0.0)),
                }

    num_changes = 0
    last_prediction = None
    for prediction in predictions:
        if prediction is None:
            continue
        if last_prediction is None:
            last_prediction = prediction
            continue
        if prediction != last_prediction:
            num_changes += 1
            last_prediction = prediction

    stabilized_index = None
    stabilized_value = None
    for i, prediction in enumerate(predictions):
        if prediction is None:
            continue
        if all(other == prediction for other in predictions[i:]):
            stabilized_index = i
            stabilized_value = prediction
            break

    return {
        "num_chunks": len(option_probs),
        "per_chunk_top_confidence": top_confidences,
        "max_confidence": max(finite_confidences) if finite_confidences else np.nan,
        "mean_confidence": (
            float(np.mean(finite_confidences)) if finite_confidences else np.nan
        ),
        "final_confidence": (
            top_confidences[-1] if top_confidences and np.isfinite(top_confidences[-1]) else np.nan
        ),
        "max_jump": max_jump,
        "max_drop": max_drop,
        "num_changes": num_changes,
        "final_prediction": predictions[-1] if predictions else None,
        "stabilized_index": stabilized_index,
        "stabilized_value": stabilized_value,
    }


def _empty_metrics(num_chunks: int = 0) -> Dict[str, Optional[object]]:
    return {
        "num_chunks": num_chunks,
        "per_chunk_top_confidence": [],
        "max_confidence": np.nan,
        "mean_confidence": np.nan,
        "final_confidence": np.nan,
        "max_jump": None,
        "max_drop": None,
        "num_changes": np.nan,
        "final_prediction": None,
        "stabilized_index": None,
        "stabilized_value": None,
    }
