import numpy as np
import torch

from lm_polygraph.estimators import (
    ConfidenceLeapFinalConfidence,
    ConfidenceLeapMaxJump,
    ConfidenceLeapNumChanges,
)
from lm_polygraph.defaults.register_default_stat_calculators import (
    register_default_stat_calculators,
)
from lm_polygraph.stat_calculators.confidence_leap import ConfidenceLeapCalculator
from lm_polygraph.utils.manager import order_calculators
from lm_polygraph.utils.model import WhiteboxModel


class _ChatTemplateTokenizer:
    chat_template = "dummy"

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=False,
    ):
        rendered = "".join(
            f"<|{message['role']}|>{message.get('content', '')}"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<|assistant|>"
        if not continue_final_message:
            rendered += "<|end|>"
        return rendered


class _LegacyChatTemplateTokenizer:
    chat_template = "dummy"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        rendered = "".join(
            f"<|{message['role']}|>{message.get('content', '')}"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<|assistant|>"
        return rendered


def test_format_assistant_continuation_uses_open_final_assistant_message():
    model = WhiteboxModel(None, _ChatTemplateTokenizer(), instruct=True)

    rendered = model.format_assistant_continuation("Question?", "partial answer")

    assert rendered == "<|user|>Question?<|assistant|>partial answer"


def test_format_assistant_continuation_falls_back_for_legacy_chat_templates():
    model = WhiteboxModel(None, _LegacyChatTemplateTokenizer(), instruct=True)

    rendered = model.format_assistant_continuation("Question?", "partial answer")

    assert rendered == "<|user|>Question?<|assistant|>partial answer"


class _OptionTokenizer:
    chat_template = None
    eos_token_id = 99
    bos_token_id = None

    _ids = {
        "A": 0,
        " A": 1,
        "B": 2,
        " B": 3,
        "C": 4,
        " C": 5,
        "D": 6,
        " D": 7,
    }

    def encode(self, text, add_special_tokens=False):
        return [self._ids[text]] if text in self._ids else [88]

    def decode(self, tokens, skip_special_tokens=False):
        return (
            "<think>Initial reasoning.\n\n"
            "Wait, revised reasoning.</think>\n"
            "\\boxed{B}"
        )


class _FakeWhiteboxModel(WhiteboxModel):
    def __init__(self):
        super().__init__(None, _OptionTokenizer(), instruct=False)
        self.prefixes = []

    def tokenize_assistant_continuations(self, input_texts, assistant_prefixes):
        self.prefixes = assistant_prefixes
        return {"input_ids": torch.ones((len(assistant_prefixes), 1), dtype=torch.long)}

    def device(self):
        return "cpu"

    def __call__(self, **args):
        logits = torch.full((len(self.prefixes), 1, 8), -8.0)
        for i, prefix in enumerate(self.prefixes):
            if "Wait" in prefix:
                logits[i, -1, 2] = 8.0
            else:
                logits[i, -1, 0] = 8.0

        class _Output:
            pass

        output = _Output()
        output.logits = logits
        return output


def test_confidence_leap_calculator_emits_chunks_probs_and_metrics():
    model = _FakeWhiteboxModel()
    calc = ConfidenceLeapCalculator()

    out = calc(
        {"input_texts": ["Q"], "greedy_tokens": [[1, 2, 3]]},
        ["Q"],
        model,
    )

    assert out["confidence_leap_status"] == ["ok"]
    assert out["confidence_leap_chunks"] == [
        ["Initial reasoning.", "Wait, revised reasoning."]
    ]
    assert out["confidence_leap_predictions"] == [["A", "B"]]
    metrics = out["confidence_leap_metrics"][0]
    assert metrics["num_chunks"] == 2
    assert metrics["num_changes"] == 1
    assert metrics["max_jump"]["option"] == "B"
    assert metrics["final_prediction"] == "B"


def test_confidence_leap_estimators_follow_uncertainty_direction():
    stats = {
        "confidence_leap_metrics": [
            {
                "final_confidence": 0.8,
                "max_jump": {"delta": 0.6},
                "num_changes": 2,
            }
        ]
    }

    assert np.allclose(ConfidenceLeapFinalConfidence()(stats), [0.2])
    assert np.allclose(ConfidenceLeapMaxJump()(stats), [0.4])
    assert np.allclose(ConfidenceLeapNumChanges()(stats), [2.0])


def test_default_registry_resolves_confidence_leap_dependencies():
    calculators = register_default_stat_calculators("Whitebox")
    stat_calculators = {stat: calc for calc in calculators for stat in calc.stats}
    stat_dependencies = {stat: calc.dependencies for calc in calculators for stat in calc.stats}

    ordered, have_stats = order_calculators(
        ["confidence_leap_metrics"],
        stat_calculators,
        stat_dependencies,
    )

    assert ordered == ["greedy_tokens", "confidence_leap_metrics"]
    assert "confidence_leap_chunks" in have_stats
    assert "confidence_leap_metrics" in have_stats
