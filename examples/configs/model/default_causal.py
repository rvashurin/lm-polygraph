import torch

from transformers import AutoModelForCausalLM, AutoTokenizer


def _resolve_torch_dtype(torch_dtype: str = None):
    if torch_dtype is None or torch_dtype == "auto":
        return torch_dtype
    if isinstance(torch_dtype, torch.dtype):
        return torch_dtype
    torch_dtype = torch_dtype.lower()
    if torch_dtype in {"float32", "fp32"}:
        return torch.float32
    if torch_dtype in {"float16", "fp16"}:
        return torch.float16
    if torch_dtype in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported torch_dtype: {torch_dtype}")


def load_model(model_path: str, device_map: str, torch_dtype: str = None):
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": device_map,
        "attn_implementation": "eager",
    }
    resolved_dtype = _resolve_torch_dtype(torch_dtype)
    if resolved_dtype is not None:
        model_kwargs["torch_dtype"] = resolved_dtype

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **model_kwargs,
    )
    model.eval()

    return model


def load_tokenizer(model_path: str, add_bos_token: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="left",
        add_bos_token=add_bos_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer
