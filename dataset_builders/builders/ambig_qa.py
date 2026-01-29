from functools import partial

import datasets


SIMPLE_INSTRUCT_PROMPT = (
    "Answer the following question as briefly as possible.\n\n"
    "Question: {question}\n"
    "Answer:"
)


def _extract_annotation_type(record):
    annotations = record.get("annotations")
    if isinstance(annotations, dict):
        annotation_type = annotations.get("type")
        if isinstance(annotation_type, list):
            return annotation_type[0] if annotation_type else None
        return annotation_type
    if isinstance(annotations, list) and annotations:
        first = annotations[0]
        if isinstance(first, dict):
            annotation_type = first.get("type")
            if isinstance(annotation_type, list):
                return annotation_type[0] if annotation_type else None
            return annotation_type
    return None


def _build_ambigqa_filtered(
    dataset,
    prompt,
    train_split,
    validation_split,
    annotation_type,
):
    def prepare_split(split_name):
        if split_name not in dataset:
            return datasets.Dataset.from_dict({"input": [], "output": []})
        is_train = split_name == train_split
        inputs, outputs = [], []
        for record in dataset[split_name]:
            record_type = _extract_annotation_type(record)
            if record_type != annotation_type:
                continue
            question = record.get("question")
            answer = record.get("nq_answer") if is_train else ["empty"]
            inputs.append(prompt.format(question=question))
            outputs.append(answer)
        return datasets.Dataset.from_dict({"input": inputs, "output": outputs})

    return datasets.DatasetDict(
        {
            "train": prepare_split(train_split),
            "validation": prepare_split(validation_split),
        }
    )


CONFIG = {
    "ambig_qa_simple_instruct_ambiguous": {
        "name": ["sewon/ambig_qa", "full"],
        "build_func": partial(
            _build_ambigqa_filtered,
            prompt=SIMPLE_INSTRUCT_PROMPT,
            train_split="train",
            validation_split="validation",
            annotation_type="multipleQAs",
        ),
        "dataset": "ambig_qa",
        "subset": "simple_instruct_ambiguous",
    },
    "ambig_qa_simple_instruct_well_specified": {
        "name": ["sewon/ambig_qa", "full"],
        "build_func": partial(
            _build_ambigqa_filtered,
            prompt=SIMPLE_INSTRUCT_PROMPT,
            train_split="train",
            validation_split="validation",
            annotation_type="singleAnswer",
        ),
        "dataset": "ambig_qa",
        "subset": "simple_instruct_well_specified",
    },
}
