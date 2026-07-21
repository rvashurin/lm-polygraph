from collections import Counter

from datasets import Dataset as HFDataset

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
