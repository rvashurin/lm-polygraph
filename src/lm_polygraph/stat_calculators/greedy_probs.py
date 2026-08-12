import torch
import numpy as np

from typing import Dict, List, Tuple, Union

from .embeddings import get_embeddings_from_output
from .stat_calculator import StatCalculator
from lm_polygraph.model_adapters import WhiteboxModel, WhiteboxModelvLLM


class GreedyProbsCalculator(StatCalculator):
    """
    For Whitebox model (lm_polygraph.WhiteboxModel), at input texts batch calculates:
    * generation texts
    * tokens of the generation texts
    * probabilities distribution of the generated tokens
    * attention masks across the model (if applicable)
    * embeddings from the model
    """

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        """
        Returns the statistics and dependencies for the calculator.
        """
        return [
            "input_texts",
            "input_tokens",
            "greedy_log_probs",
            "greedy_tokens",
            "greedy_tokens_alternatives",
            "greedy_texts",
            "greedy_log_likelihoods",
            "embeddings",
            "attention_all",
            "attention_selected",
            "tokenizer",
        ], []

    def __init__(
        self,
        output_attentions: bool = True,
        output_hidden_states: bool = False,
        n_alternatives: int = 10,
        output_attentions_selected: bool = False,
    ):
        """
        Parameters:
            output_attentions (bool): Whether to collect the full `attention_all` map.
            output_hidden_states (bool): Whether to collect embeddings.
            n_alternatives (int): Number of alternative tokens to store per position.
            output_attentions_selected (bool): Whether to additionally collect
                `attention_selected` — per-head attention from each generated token
                back over the prompt, for a few chosen layers. Off by default: it is
                far larger than `attention_all` (hundreds of MB per sample for long
                generations), so enable it only for targeted runs.
        """
        super().__init__()
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.n_alternatives = n_alternatives
        self.output_attentions_selected = output_attentions_selected

    @staticmethod
    def _eos_token_ids(model) -> set:
        """
        Collect every token id that terminates a generation.

        Models such as Falcon3, Qwen and Llama-3 declare several EOS ids (e.g. a
        base `<|endoftext|>` plus a chat-template `<|im_end|>`), and the tokenizer
        and the model config do not always agree on the full set. Comparing against
        `tokenizer.eos_token_id` alone misses the alternates, so the generation is
        not truncated where it actually ended and the trailing padding is scored as
        if the model had produced it.

        Returns the union of the tokenizer's and the model config's ids, tolerating
        an int, a list, or None from either side.
        """

        def _as_ids(value) -> list:
            if value is None:
                return []
            if isinstance(value, int):
                return [value]
            return [v for v in value if isinstance(v, int)]

        ids = _as_ids(getattr(model.tokenizer, "eos_token_id", None))

        inner = getattr(model, "model", None)
        config = getattr(inner, "config", None)
        ids += _as_ids(getattr(config, "eos_token_id", None))

        return set(ids)

    def _preprocess_attention(
        self,
        attentions: torch.Tensor,
        current_idx: int,
        start_idx: int,
        end_idx: int,
        prompt_len: int,
    ) -> torch.Tensor:
        """
        Preprocess attention weights before stacking.

        Parameters:
            attentions (torch.Tensor): Attention weights from a specific layer and head for a current token
            current_idx (int): Current position in the sequence
            start_idx (int): Start index of the generated tokens
            end_idx (int): End index of the generated tokens for current position
            prompt_len (int): Length of the prompt

        Returns:
            torch.Tensor: Preprocessed attention weights
        """
        # Handle attention tensor processing for models with varying attention sizes (e.g. Gemma)
        n_attentions = attentions.shape[-1]

        # Handle empty tensor case
        if attentions.nelement() == 0:
            return torch.zeros(abs(current_idx), device=attentions.device)

        # Handle cases where attention size is smaller than expected
        if n_attentions < end_idx:
            if start_idx < 0:
                return attentions[start_idx:n_attentions]
            return attentions[n_attentions - current_idx : n_attentions]

        # Handle cases where attention spans beyond expected range
        if (n_attentions - current_idx) > end_idx and start_idx < 0:
            return attentions[prompt_len : prompt_len + current_idx]

        # Default case: return attention slice within expected range
        return attentions[start_idx:end_idx]

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: Union[WhiteboxModel, WhiteboxModelvLLM],
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        """
        Calculates the statistics of probabilities at each token position in the generation.

        Parameters:
            dependencies (Dict[str, np.ndarray]): input statistics, can be empty (not used).
            texts (List[str]): Input texts batch used for model generation.
            model (Model): Model used for generation.
            max_new_tokens (int): Maximum number of new tokens at model generation. Default: 100.
        Returns:
            Dict[str, np.ndarray]: dictionary with the following items:
                - 'input_tokens' (List[List[int]]): tokenized input texts,
                - 'greedy_log_probs' (List[List[np.array]]): logarithms of autoregressive
                        probability distributions at each token,
                - 'greedy_texts' (List[str]): model generations corresponding to the inputs,
                - 'greedy_tokens' (List[List[int]]): tokenized model generations,
                - 'attention' (List[List[np.array]]): attention maps at each token, if applicable to the model,
                - 'greedy_log_likelihoods' (List[List[float]]): log-probabilities of the generated tokens.
        """
        batch: Dict[str, torch.Tensor] = model.tokenize(texts)
        batch = {k: v.to(model.device()) for k, v in batch.items()}
        with torch.no_grad():
            out = model.generate(
                **batch,
                output_scores=True,
                return_dict_in_generate=True,
                max_new_tokens=max_new_tokens,
                min_new_tokens=2,
                output_attentions=self.output_attentions,
                output_hidden_states=self.output_hidden_states,
                num_return_sequences=1,
                suppress_tokens=(
                    []
                    if model.generation_parameters.allow_newlines
                    else [
                        t
                        for t in range(len(model.tokenizer))
                        if "\n" in model.tokenizer.decode([t])
                    ]
                ),
            )
            logits = torch.stack(out.scores, dim=1)
            if model.model_type == "vLLMCausalLM":
                logits = logits.transpose(1, 0)
            sequences = out.sequences
            if self.output_attentions:
                attentions = out.attentions
            if self.output_hidden_states:
                embeddings_encoder, embeddings_decoder = get_embeddings_from_output(
                    out, batch, model.model_type
                )
                if embeddings_decoder.dtype == torch.bfloat16:
                    embeddings_decoder = embeddings_decoder.to(
                        torch.float16
                    )  # numpy does not support bfloat16

        eos_ids = self._eos_token_ids(model)

        cut_logits = []
        cut_sequences = []
        cut_texts = []
        cut_alternatives = []
        for i in range(len(texts)):
            if model.model_type == "CausalLM":
                idx = batch["input_ids"].shape[1]
                seq = sequences[i, idx:].cpu()
            elif model.model_type == "vLLMCausalLM":
                seq = sequences[i].cpu()
            else:
                seq = sequences[i, 1:].cpu()
            length, text_length = len(seq), len(seq)
            for j in range(len(seq)):
                if seq[j].item() in eos_ids:
                    length = j + 1
                    text_length = j
                    break
            cut_sequences.append(seq[:length].tolist())
            cut_texts.append(model.tokenizer.decode(seq[:text_length]))
            cut_logits.append(logits[i, :length, :].cpu().numpy())
            cut_alternatives.append([[] for _ in range(length)])
            for j in range(length):
                lt = logits[i, j, :].cpu().numpy()
                best_tokens = np.argpartition(lt, -self.n_alternatives)
                ln = len(best_tokens)
                best_tokens = best_tokens[ln - self.n_alternatives : ln]
                for t in best_tokens:
                    cut_alternatives[-1][j].append((t.item(), lt[t].item()))
                cut_alternatives[-1][j].sort(
                    key=lambda x: x[0] == cut_sequences[-1][j],
                    reverse=True,
                )

        ll = []
        for i in range(len(texts)):
            log_probs = cut_logits[i]
            tokens = cut_sequences[i]
            assert len(tokens) == len(log_probs)
            ll.append([log_probs[j, tokens[j]] for j in range(len(log_probs))])

        attention_all = []
        if self.output_attentions and (model.model_type != "vLLMCausalLM"):
            prompt_len = batch["input_ids"].shape[1]
            for i in range(len(texts)):
                c = len(cut_sequences[i])
                attn_mask = np.zeros(
                    shape=(
                        model.model.config.num_attention_heads
                        * model.model.config.num_hidden_layers,
                        c,
                        c,
                    )
                )
                for j in range(1, c):
                    # Get attention dimensions
                    current_attention_len = attentions[j][0].shape[-1]

                    # Default case: use relative indexing from end
                    start_idx = -j
                    end_idx = current_attention_len

                    # Special case for models like Gemma that maintain consistent attention lengths
                    if attentions[0][0].shape[-1] == current_attention_len:
                        start_idx = prompt_len
                        end_idx = prompt_len + j

                    stacked_attention = torch.vstack(
                        [
                            self._preprocess_attention(
                                attentions[j][layer][0][head][0],
                                j,
                                start_idx,
                                end_idx,
                                prompt_len,
                            )
                            for layer in range(len(attentions[j]))
                            for head in range(len(attentions[j][layer][0]))
                        ]
                    )
                    if stacked_attention.dtype == torch.bfloat16:
                        stacked_attention = stacked_attention.to(
                            torch.float16
                        )  # numpy does not support bfloat16

                    attn_mask[:, j, :j] = stacked_attention.cpu().numpy()
                attention_all.append(attn_mask)

        attention_selected = []
        if (
            self.output_attentions
            and self.output_attentions_selected
            and (model.model_type != "vLLMCausalLM")
        ):
            # Attention from each generated token back over the prompt, kept per
            # head for a few layers rather than collapsed like `attention_all`.
            # Layers: the middle one plus the last two, which is where the
            # attention-based uncertainty signal is usually looked for.
            num_layers = len(attentions[0])
            selected_layers = sorted({num_layers // 2, num_layers - 2, num_layers - 1})
            selected_layers = [layer for layer in selected_layers if layer >= 0]
            num_heads = attentions[0][selected_layers[0]].shape[1]
            prompt_len = batch["input_ids"].shape[1]

            for i in range(len(texts)):
                c = len(cut_sequences[i])
                if c == 0:
                    attention_selected.append(None)
                    continue

                attn_mask = np.zeros(
                    shape=(len(selected_layers), num_heads, c, prompt_len)
                )
                for j in range(c):
                    if j >= len(attentions):
                        break
                    for li, layer in enumerate(selected_layers):
                        layer_attn = attentions[j][layer][i, :num_heads, 0, :prompt_len]
                        if layer_attn.dtype == torch.bfloat16:
                            layer_attn = layer_attn.to(
                                torch.float16
                            )  # numpy does not support bfloat16
                        key_len = layer_attn.shape[-1]
                        attn_mask[li, :num_heads, j, :key_len] = (
                            layer_attn.cpu().numpy()
                        )
                attention_selected.append(attn_mask)

        if not self.output_hidden_states:
            embeddings_dict = {}
        elif model.model_type == "CausalLM":
            embeddings_dict = {
                "embeddings_decoder": embeddings_decoder.cpu().detach().numpy(),
            }
        elif model.model_type == "Seq2SeqLM":
            embeddings_dict = {
                "embeddings_encoder": embeddings_encoder.cpu().detach().numpy(),
                "embeddings_decoder": embeddings_decoder.cpu().detach().numpy(),
            }
        else:
            raise NotImplementedError

        result_dict = {
            "input_tokens": batch["input_ids"].to("cpu").tolist(),
            "greedy_log_probs": cut_logits,
            "greedy_tokens": cut_sequences,
            "greedy_tokens_alternatives": cut_alternatives,
            "greedy_texts": cut_texts,
            "greedy_log_likelihoods": ll,
        }
        result_dict.update(embeddings_dict)
        if self.output_attentions:
            result_dict.update({"attention_all": attention_all})
            result_dict.update({"tokenizer": model.tokenizer})
            if self.output_attentions_selected:
                result_dict.update({"attention_selected": attention_selected})
        return result_dict
