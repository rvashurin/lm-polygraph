import re
import torch
import torch.nn.functional as F
import numpy as np

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from huggingface_hub import hf_hub_download

from .stat_calculator import StatCalculator
from lm_polygraph.model_adapters.whitebox_model import WhiteboxModel


def _resolve_dtype(dtype: Optional[str], device: torch.device) -> torch.dtype:
    if dtype is None or dtype == "auto":
        return torch.float32 if device.type == "cpu" else torch.bfloat16

    dtype = str(dtype).lower()
    if dtype in {"float32", "fp32"}:
        return torch.float32
    if dtype in {"float16", "fp16"}:
        return torch.float16
    if dtype in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported SAE dtype: {dtype}")


def _load_torch_checkpoint(path: str) -> Dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _get_state_tensor(
    state_dict: Dict[str, torch.Tensor],
    key: str,
    required: bool = True,
) -> Optional[torch.Tensor]:
    value = state_dict.get(key)
    if value is None and required:
        raise KeyError(f"SAE checkpoint is missing required key: {key}")
    return value


class _BatchTopKSAEEncoder:
    def __init__(
        self,
        repo_id: str,
        sae_path: str,
        hf_cache: Optional[str] = None,
        hf_token: Optional[str] = None,
        device: str = "auto",
        dtype: str = "auto",
        use_threshold: bool = True,
        fallback_to_top_k: bool = True,
        k: Optional[int] = None,
    ):
        self.repo_id = repo_id
        self.sae_path = sae_path
        self.hf_cache = hf_cache
        self.hf_token = hf_token
        self.device = device
        self.dtype = dtype
        self.use_threshold = use_threshold
        self.fallback_to_top_k = fallback_to_top_k
        self.k = k

        self.encoder_weight = None
        self.encoder_bias = None
        self.b_dec = None
        self.threshold = None
        self.activation_dim = None
        self.dict_size = None

    def _resolve_device(self, reference_device: torch.device) -> torch.device:
        if self.device in {None, "auto", "model"}:
            return reference_device
        return torch.device(self.device)

    def _checkpoint_path(self) -> str:
        if Path(self.sae_path).exists():
            return self.sae_path
        return hf_hub_download(
            repo_id=self.repo_id,
            filename=self.sae_path,
            cache_dir=self.hf_cache,
            token=self.hf_token,
        )

    def load(self, reference_device: torch.device):
        if self.encoder_weight is not None:
            return

        device = self._resolve_device(reference_device)
        dtype = _resolve_dtype(self.dtype, device)
        state_dict = _load_torch_checkpoint(self._checkpoint_path())

        self.encoder_weight = _get_state_tensor(state_dict, "encoder.weight").to(
            device=device,
            dtype=dtype,
        )
        self.encoder_bias = _get_state_tensor(state_dict, "encoder.bias").to(
            device=device,
            dtype=dtype,
        )
        self.b_dec = _get_state_tensor(state_dict, "b_dec").to(
            device=device,
            dtype=dtype,
        )

        threshold = _get_state_tensor(state_dict, "threshold", required=False)
        if threshold is not None:
            self.threshold = threshold.to(device=device, dtype=dtype)

        if self.k is None:
            loaded_k = _get_state_tensor(state_dict, "k", required=False)
            if loaded_k is not None:
                self.k = int(loaded_k.item())

        self.dict_size, self.activation_dim = self.encoder_weight.shape
        del state_dict

    def encode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.load(hidden_states.device)
        hidden_states = hidden_states.to(
            device=self.encoder_weight.device,
            dtype=self.encoder_weight.dtype,
        )
        if hidden_states.shape[-1] != self.activation_dim:
            raise ValueError(
                f"SAE expected activation dim {self.activation_dim}, got "
                f"{hidden_states.shape[-1]}"
            )

        pre_acts = F.linear(
            hidden_states - self.b_dec,
            self.encoder_weight,
            self.encoder_bias,
        )
        latents = torch.relu(pre_acts)

        if (
            self.use_threshold
            and self.threshold is not None
            and self.threshold.item() >= 0
        ):
            latents = latents * (latents > self.threshold)
        elif self.fallback_to_top_k and self.k is not None and self.k > 0:
            flattened_latents = latents.flatten()
            top_k = min(self.k * latents.shape[0], flattened_latents.shape[0])
            values, indices = torch.topk(flattened_latents, top_k, sorted=False)
            latents = (
                torch.zeros_like(flattened_latents)
                .scatter(-1, indices, values)
                .reshape(latents.shape)
            )

        return latents


class SAELatentActivationsCalculator(StatCalculator):
    """
    Computes aggregated SAE latent activations for generated sequences.

    The calculator reuses greedy tokens generated by GreedyProbsCalculator, runs a
    teacher-forced forward pass over prompt plus generation, captures the configured
    residual stream layer with a forward hook, and encodes selected positions with
    a BatchTopK SAE checkpoint.
    """

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        return ["sae_latent_activations"], ["greedy_tokens"]

    def __init__(
        self,
        repo_id: str,
        sae_path: str,
        layer: int,
        module_path: Optional[str] = None,
        hf_cache: Optional[str] = None,
        hf_token: Optional[str] = None,
        device: str = "auto",
        dtype: str = "auto",
        use_threshold: bool = True,
        fallback_to_top_k: bool = True,
        k: Optional[int] = None,
        token_positions: str = "prediction",
        aggregation: str = "mean",
    ):
        super().__init__()
        self.layer = layer
        self.module_path = module_path or f"model.layers.{layer}"
        self.token_positions = token_positions
        self.aggregation = aggregation
        self.encoder = _BatchTopKSAEEncoder(
            repo_id=repo_id,
            sae_path=sae_path,
            hf_cache=hf_cache,
            hf_token=hf_token,
            device=device,
            dtype=dtype,
            use_threshold=use_threshold,
            fallback_to_top_k=fallback_to_top_k,
            k=k,
        )

    @staticmethod
    def infer_layer_from_path(sae_path: str) -> Optional[int]:
        match = re.search(r"resid_post_layer_(\d+)", sae_path)
        if match:
            return int(match.group(1))
        return None

    def _get_hook_module(self, model: WhiteboxModel) -> torch.nn.Module:
        try:
            return model.model.get_submodule(self.module_path)
        except AttributeError:
            module = model.model
            for name in self.module_path.split("."):
                module = getattr(module, name)
            return module

    def _build_forward_batch(
        self,
        texts: List[str],
        greedy_tokens: List[List[int]],
        model: WhiteboxModel,
    ) -> Tuple[Dict[str, torch.Tensor], List[List[int]]]:
        batch = model.tokenize(texts)
        input_ids = batch["input_ids"].cpu().tolist()
        attention_mask = batch.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(batch["input_ids"])
        attention_mask = attention_mask.cpu().tolist()

        pad_token_id = model.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = model.tokenizer.eos_token_id

        full_input_ids = []
        full_attention_mask = []
        selected_positions = []
        for prompt_ids, prompt_mask, sample_greedy_tokens in zip(
            input_ids,
            attention_mask,
            greedy_tokens,
        ):
            sample_greedy_tokens = list(sample_greedy_tokens)
            full_ids = prompt_ids + sample_greedy_tokens
            full_mask = prompt_mask + [1] * len(sample_greedy_tokens)
            prompt_end = len(prompt_ids)
            generation_len = len(sample_greedy_tokens)

            if generation_len == 0:
                positions = [prompt_end - 1]
            elif self.token_positions == "prediction":
                positions = list(range(prompt_end - 1, prompt_end + generation_len - 1))
            elif self.token_positions == "generated":
                positions = list(range(prompt_end, prompt_end + generation_len))
            elif self.token_positions == "last_prediction":
                positions = [prompt_end + generation_len - 2]
            elif self.token_positions == "last_generated":
                positions = [prompt_end + generation_len - 1]
            else:
                raise ValueError(
                    f"Unsupported SAE token_positions: {self.token_positions}"
                )

            full_input_ids.append(full_ids)
            full_attention_mask.append(full_mask)
            selected_positions.append(positions)

        max_len = max(len(ids) for ids in full_input_ids)
        for ids, mask in zip(full_input_ids, full_attention_mask):
            pad_len = max_len - len(ids)
            ids.extend([pad_token_id] * pad_len)
            mask.extend([0] * pad_len)

        device = model.device()
        return (
            {
                "input_ids": torch.tensor(full_input_ids, dtype=torch.long, device=device),
                "attention_mask": torch.tensor(
                    full_attention_mask,
                    dtype=torch.long,
                    device=device,
                ),
            },
            selected_positions,
        )

    def _aggregate(self, latents: torch.Tensor) -> torch.Tensor:
        if self.aggregation == "mean":
            return latents.mean(dim=0)
        if self.aggregation == "sum":
            return latents.sum(dim=0)
        if self.aggregation == "max":
            return latents.max(dim=0).values
        raise ValueError(f"Unsupported SAE aggregation: {self.aggregation}")

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        if model.model_type != "CausalLM":
            raise NotImplementedError(
                "SAE latent activations are implemented for CausalLM models."
            )

        greedy_tokens = dependencies["greedy_tokens"]
        forward_batch, selected_positions = self._build_forward_batch(
            texts,
            greedy_tokens,
            model,
        )

        captured_hidden_states = []

        def capture_residual_stream(_module, _inputs, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            captured_hidden_states.append(hidden_states.detach())

        hook = self._get_hook_module(model).register_forward_hook(capture_residual_stream)
        try:
            with torch.no_grad():
                model.model(**forward_batch, use_cache=False)
        finally:
            hook.remove()

        if len(captured_hidden_states) != 1:
            raise RuntimeError(
                f"Expected to capture exactly one SAE layer activation, got "
                f"{len(captured_hidden_states)}"
            )

        hidden_states = captured_hidden_states[0]
        sample_latents = []
        for sample_idx, positions in enumerate(selected_positions):
            activations = hidden_states[sample_idx, positions, :]
            latents = self.encoder.encode(activations)
            sample_latents.append(self._aggregate(latents).float().cpu().numpy())

        return {"sae_latent_activations": np.stack(sample_latents, axis=0)}
