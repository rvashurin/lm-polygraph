import re


_THINK_BLOCK_RE = re.compile(r"(?is)<think>.*?</think>")
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^>]+?\|>")
_ANSWER_PATTERNS = (
    re.compile(r"(?is)\b(?:answer|guess|option|choice)\b\s*(?:is|:)?\s*[\(\[]?\s*([ABCD])\b"),
    re.compile(r"(?is)^\s*[\(\[]?\s*([ABCD])\b"),
    re.compile(r"(?is)\b([ABCD])\s*(?:[\)\].,:;!?]|$)"),
)


def strip_qwen_thinking(text: str) -> str:
    text = str(text or "")
    close_tag = "</think>"
    close_pos = text.lower().rfind(close_tag)
    if close_pos != -1:
        text = text[close_pos + len(close_tag) :]
    elif "<think>" in text.lower():
        return ""

    text = _THINK_BLOCK_RE.sub(" ", text)
    text = _SPECIAL_TOKEN_RE.sub(" ", text)
    return " ".join(text.split())


def normalize_mmlu_answer(text: str) -> str:
    text = strip_qwen_thinking(text)
    for pattern in _ANSWER_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).upper()
    return text.strip().upper()


def process_output_mmlu(output: str) -> str:
    return normalize_mmlu_answer(output)


def process_target_mmlu(target: str) -> str:
    return normalize_mmlu_answer(target)
