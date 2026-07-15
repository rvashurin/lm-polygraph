import re


# Noise stripped before any answer extraction.
GEMMA_OUTPUT_IGNORE_REGEX = re.compile(r"<end_of_turn>")
QWEN_FALCON_EOS_IGNORE_REGEX = re.compile(r"<\|im_end\|>|<\|endoftext\|>")
# Falcon chat templates use plain-text turn markers; if generation did not
# stop at our generate_until strings, drop everything from the next user turn
# onward. The colon is optional because some outputs end with 'User ' alone.
FALCON_TURN_LEAK_REGEX = re.compile(r"\n\n?User\b.*$", re.DOTALL)
# Strip everything up to and including '### Answer:' so reasoning-style
# prompts keep only the answer portion. Whitespace between '###' and 'Answer'
# is permissive (covers '###\nAnswer:').
REASONING_OUTPUT_IGNORE_REGEX = re.compile(r"(?s).*###\s*Answer:\s*")
# Leading '###' prefix without an 'Answer:' word (e.g. '### a)'). Applied
# only after REASONING_OUTPUT_IGNORE_REGEX has had its chance.
HASH_PREFIX_REGEX = re.compile(r"^\s*#{2,}\s*")
# Common preambles models like to emit before the answer.
LEADING_PHRASE_REGEX = re.compile(
    r"^\s*(?:the\s+answer\s+is|final\s+answer|answer)\s*[:\-]?\s*",
    re.IGNORECASE,
)

# A single MCQ letter (a-d), optionally wrapped in parens/brackets and
# optionally followed by ) ] . : , or whitespace, anchored at the start of
# the cleaned text.
MCQ_LETTER_REGEX = re.compile(
    r"^\s*[\(\[]?\s*([a-dA-D])\s*[\)\]\.\:\,]?(?:\s|$)"
)
# Fallback: search anywhere for 'the answer is X' / 'answer: X' / '(X)'.
# The captured letter must be standalone: '(?![A-Za-z])' stops 'the answer is
# Because...' from matching 'b' out of the middle of a word.
ANSWER_PHRASE_SEARCH_REGEX = re.compile(
    r"(?i)(?:the\s+answer\s+is|answer\s*[:\-])\s*[\(\[]?\s*([a-dA-D])(?![A-Za-z])"
)
PAREN_LETTER_SEARCH_REGEX = re.compile(r"\(([a-dA-D])\)")

# A whole number: optional sign, thousands separators, and an optional decimal
# part captured as ONE token (so '7.00' is a single match, not '7' then '00').
NUMBER_EXTRACTION_REGEX = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
# Back-compat alias (older configs import INTEGER_EXTRACTION_REGEX).
INTEGER_EXTRACTION_REGEX = NUMBER_EXTRACTION_REGEX

# Legacy regex kept for back-compat with old configs.
PARENTHESEIS_OUTPUT_IGNORE_REGEX = re.compile(r"\)")


def _strip_noise(output: str) -> str:
    output = GEMMA_OUTPUT_IGNORE_REGEX.sub("", output)
    output = QWEN_FALCON_EOS_IGNORE_REGEX.sub("", output)
    output = FALCON_TURN_LEAK_REGEX.sub("", output)
    output = REASONING_OUTPUT_IGNORE_REGEX.sub("", output)
    output = HASH_PREFIX_REGEX.sub("", output)
    output = LEADING_PHRASE_REGEX.sub("", output)
    return output.strip()


def process_output_mcq(output: str) -> str:
    """Extract a single multiple-choice letter (a-d) for MMLU/medmcqa direct.

    Handles 'a', 'a)', '(a)', 'A', 'Answer: a', 'The answer is c.',
    'a) Coronary vasodilation', '### a)', '### a) Paap', and Falcon
    turn-marker leaks like 'a\\nUser'.
    """
    cleaned = _strip_noise(output)
    m = MCQ_LETTER_REGEX.match(cleaned)
    if m:
        return m.group(1).lower()
    m = ANSWER_PHRASE_SEARCH_REGEX.search(cleaned)
    if m:
        return m.group(1).lower()
    m = PAREN_LETTER_SEARCH_REGEX.search(cleaned)
    if m:
        return m.group(1).lower()
    return cleaned


def _canonicalize_number(token: str) -> str:
    """Normalize a matched number token to a canonical string for exact-string
    comparison against bare-integer gold: strip thousands separators, and render
    whole-valued floats as ints so '7', '7.0', '7.00' all become '7'. Non-integral
    values keep their decimal form; unparseable tokens pass through unchanged."""
    val = token.replace(",", "")
    try:
        f = float(val)
    except ValueError:
        return val
    return str(int(f)) if f.is_integer() else str(f)


def process_output_number(output: str) -> str:
    """Extract the answer number for gsm8k.

    Takes the LAST number in the cleaned text (GSM8k answers are conventionally
    the final number; reasoning traces state intermediate numbers first), matches
    whole numbers including any decimal part, and canonicalizes numerically
    ('1,234' -> '1234', '7.00' -> '7'). Does not attempt letter extraction so
    'a 5' style outputs still resolve to '5'.
    """
    cleaned = _strip_noise(output)
    matches = NUMBER_EXTRACTION_REGEX.findall(cleaned)
    if matches:
        return _canonicalize_number(matches[-1])
    return cleaned


def process_output(output: str) -> str:
    """General-purpose processor. Tries number first (legacy behavior),
    then MCQ letter, then falls back to cleaned text.

    Prefer process_output_mcq or process_output_number when the task type
    is known; this function exists for back-compat with configs that don't
    pin fn_name.

    NOTE ordering hazard: number extraction runs before MCQ-letter extraction,
    so an MCQ answer that contains a digit (e.g. 'option 2: b') resolves to the
    number, not the letter. This is why the benchmark pins process_output_mcq /
    process_output_number per task rather than relying on this general path."""
    cleaned = _strip_noise(output)
    cleaned_legacy = PARENTHESEIS_OUTPUT_IGNORE_REGEX.sub("", cleaned)

    matches = NUMBER_EXTRACTION_REGEX.findall(cleaned_legacy)
    if matches:
        return _canonicalize_number(matches[-1])

    m = MCQ_LETTER_REGEX.match(cleaned)
    if m:
        return m.group(1).lower()
    m = ANSWER_PHRASE_SEARCH_REGEX.search(cleaned)
    if m:
        return m.group(1).lower()

    return cleaned_legacy


def process_target(output: str) -> str:
    """Target passthrough. Lowercase + strip so case/whitespace differences
    in dataset labels don't fight the processors above."""
    return str(output).strip().lower()
