from collections import Counter

from datasets import Dataset as HFDataset

from dataset_builders.builders.mmlu import CONFIG, prepare_mmlu
from lm_polygraph.utils.dataset import Dataset


def test_limit_mmlu_subject_size_caps_each_subject_in_order():
    hf_dataset = HFDataset.from_dict(
        {
            "input": [f"q{i}" for i in range(7)],
            "output": ["A"] * 7,
            "subject": ["math", "math", "math", "history", "history", "history", "cs"],
        }
    )

    limited = Dataset.limit_mmlu_subject_size(
        hf_dataset, "LM-Polygraph/mmlu", max_subject_size=2
    )

    assert list(limited["input"]) == ["q0", "q1", "q3", "q4", "q6"]
    assert Counter(limited["subject"]) == {"math": 2, "history": 2, "cs": 1}


def test_limit_mmlu_subject_size_leaves_other_datasets_unchanged():
    hf_dataset = HFDataset.from_dict(
        {"input": ["q0", "q1", "q2"], "output": ["A", "B", "C"], "subject": ["x"] * 3}
    )

    limited = Dataset.limit_mmlu_subject_size(
        hf_dataset, "some/other-dataset", max_subject_size=1
    )

    assert len(limited) == 3


def test_mmlu_qwen_simple_instruct_prompts_boxed_answers():
    eval_dataset = HFDataset.from_dict(
        {
            "question": ["What is 2+2?"],
            "choices": [["3", "4", "5", "6"]],
            "answer": [1],
            "subject": ["elementary_math"],
        }
    )
    few_shot_dataset = HFDataset.from_dict(
        {
            "question": [f"Example {i}?" for i in range(5)],
            "choices": [["A0", "B0", "C0", "D0"] for _ in range(5)],
            "answer": [0, 1, 2, 3, 0],
            "subject": ["elementary_math"] * 5,
        }
    )
    config = CONFIG["mmlu_qwen_simple_instruct"]

    x, y, _ = prepare_mmlu(
        dataset=eval_dataset,
        output_column="answer",
        prompt=config["prepare_func"].keywords["prompt"],
        description=config["prepare_func"].keywords["description"],
        mmlu_max_subject_size=100,
        n_shot=5,
        few_shot_dataset_func=lambda: few_shot_dataset,
        few_shot_prompt=config["prepare_func"].keywords["few_shot_prompt"],
        instruct=True,
    )

    assert y == ["B"]
    assert "\\boxed{A}" in x[0]
    assert "\\boxed{B}" in x[0]
    assert "Your final answer must be written exactly as \\boxed{A}" in x[0]
    assert x[0].rstrip().endswith("Answer:")
