import torch
import transformers
from transformers import AutoTokenizer


def _resolve_torch_dtype(torch_dtype):
    if torch_dtype in (None, "auto"):
        return torch_dtype
    if isinstance(torch_dtype, str):
        return getattr(torch, torch_dtype)
    return torch_dtype


def _get_model_class():
    model_class = getattr(transformers, "AutoModelForMultimodalLM", None)
    if model_class is None:
        model_class = getattr(transformers, "Qwen3_5ForConditionalGeneration", None)
    if model_class is None:
        raise ImportError(
            "Qwen/Qwen3.5-27B requires a transformers build that provides "
            "AutoModelForMultimodalLM or Qwen3_5ForConditionalGeneration."
        )
    return model_class


def load_model(
    model_path: str,
    device_map: str = "auto",
    torch_dtype: str = "auto",
    attn_implementation: str = None,
):
    load_kwargs = {
        "trust_remote_code": True,
        "device_map": device_map,
        "torch_dtype": _resolve_torch_dtype(torch_dtype),
    }
    if attn_implementation is not None:
        load_kwargs["attn_implementation"] = attn_implementation

    model = _get_model_class().from_pretrained(model_path, **load_kwargs)
    model.eval()
    return model


def load_tokenizer(model_path: str, add_bos_token: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="left",
        add_bos_token=add_bos_token,
        trust_remote_code=True,
    )

    tokenizer.padding_side = "left"
    if hasattr(tokenizer, "add_bos_token"):
        tokenizer.add_bos_token = add_bos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer
