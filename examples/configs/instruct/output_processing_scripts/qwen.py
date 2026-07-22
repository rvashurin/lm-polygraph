import re


_THINK_BLOCK_RE = re.compile(r"(?is)<think>.*?</think>")
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^>]+?\|>")
_BOXED_ANSWER_RE = re.compile(r"(?is)\\boxed\s*\{\s*([ABCD])\s*\}")
_STRICT_ANSWER_LINE_RE = re.compile(
    r"(?im)^\s*(?:final\s+answer|answer|guess|option|choice)\s*(?:is|:)?\s*[\(\[]?\s*([ABCD])\s*[\)\].,:;!]?\s*$"
)
_BARE_ANSWER_RE = re.compile(r"(?is)^\s*[\(\[]?\s*([ABCD])\s*[\)\].,:;!]?\s*$")


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

    boxed_matches = list(_BOXED_ANSWER_RE.finditer(text))
    if boxed_matches:
        return boxed_matches[-1].group(1).upper()

    line_matches = list(_STRICT_ANSWER_LINE_RE.finditer(text))
    if line_matches:
        return line_matches[-1].group(1).upper()

    match = _BARE_ANSWER_RE.match(text)
    if match:
        return match.group(1).upper()

    return text.strip().upper()


def process_output_mmlu(output: str) -> str:
    return normalize_mmlu_answer(output)


def process_target_mmlu(target: str) -> str:
    return normalize_mmlu_answer(target)
