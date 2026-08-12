import re
import logging

import torch
import numpy as np
from vllm import SamplingParams

from typing import Dict, List, Optional, Tuple, Union

from .embeddings import get_embeddings_from_output
from .stat_calculator import StatCalculator
from lm_polygraph.model_adapters import WhiteboxModel, WhiteboxModelvLLM

log = logging.getLogger("lm_polygraph")


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
            "greedy_texts_full", # UGRIP: To account for reasoning-only analysis
            "failed_generation", # UGRIP: per-sample flag, True when the answer slice is unusable
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
        answer_marker: str = "### Answer:",
        reasoning_marker: str = "### Reasoning:",
        slicing_target: str = None,
    ):
        """
        Initializes the calculator.

        Parameters:
            output_attentions (bool): Whether to calculate and return attention scores.
            output_hidden_states (bool): Whether to calculate and return embeddings.
            n_alternatives (int): Number of alternative tokens to store at each position.
            answer_marker (str): The string that separates different parts of the generation.
            reasoning_marker (str): Label that prefixes the reasoning portion of the
                generation; when slicing_target="reasoning", these tokens are stripped
                from the start of the slice so UQ is not computed on the label itself.
            slicing_target (str): Determines which part of the generation to analyze.
                - "answer": Analyzes the text after the answer_marker.
                - "reasoning": Analyzes the text before the answer_marker (and after the
                  reasoning_marker, if found at the start).
                - None or any other string: Analyzes the full generation without slicing.
        """
        super().__init__()
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.n_alternatives = n_alternatives

        if slicing_target not in ["answer", "reasoning", None]:
            self.slicing_target = None
        else:
            self.slicing_target = slicing_target
        self.answer_marker = answer_marker if self.slicing_target else None
        self.reasoning_marker = reasoning_marker if self.slicing_target == "reasoning" else None
        # Tolerant regexes for locating the markers in the decoded generation,
        # robust to per-model formatting (missing space, markdown bold/hashes,
        # casing) rather than exact-substring matching on token windows.
        self._answer_re = (
            self._tolerant_marker_regex(self.answer_marker) if self.answer_marker else None
        )
        self._reasoning_re = (
            self._tolerant_marker_regex(self.reasoning_marker) if self.reasoning_marker else None
        )

    @staticmethod
    def _tolerant_marker_regex(marker: str) -> "re.Pattern":
        """Build a tolerant regex from a marker like '### Answer:'. Matches the
        alphabetic label with flexible leading hashes, optional markdown bold,
        flexible whitespace, and case-insensitivity: '### Answer:', '###Answer:',
        '**Answer:**', 'answer :' all match. Falls back to the escaped literal if
        the marker has no letters."""
        word = re.sub(r"[^A-Za-z]", "", marker)
        if not word:
            return re.compile(re.escape(marker))
        return re.compile(r"#{0,6}\s*\*{0,2}\s*" + word + r"\s*\*{0,2}\s*:", re.IGNORECASE)

    def _marker_token_span(
        self, full_gen_seq: torch.Tensor, tokenizer, regex: "re.Pattern"
    ) -> Tuple[Optional[int], Optional[int]]:
        """Locate `regex` in the decoded generation and map its char span to token
        indices. Returns (start_tok, end_tok) where start_tok is the first token
        overlapping the marker and end_tok is the first token past it (so the
        text after the marker is full_gen_seq[end_tok:]). Returns (None, None) if
        the marker is not found."""
        text = tokenizer.decode(full_gen_seq.tolist())
        m = regex.search(text)
        if m is None:
            return None, None
        c0, c1 = m.start(), m.end()
        start_tok, end_tok = None, None
        for k in range(1, len(full_gen_seq) + 1):
            decoded_len = len(tokenizer.decode(full_gen_seq[:k].tolist()))
            if start_tok is None and decoded_len > c0:
                start_tok = k - 1
            if decoded_len >= c1:
                end_tok = k
                break
        if start_tok is None:
            start_tok = 0
        if end_tok is None:
            end_tok = len(full_gen_seq)
        return start_tok, end_tok

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

        generate_kwargs = {
            "output_scores": True,
            "return_dict_in_generate": True,
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": 2,
            "output_attentions": self.output_attentions,
            "output_hidden_states": self.output_hidden_states,
            "num_return_sequences": 1,
            "suppress_tokens": (
                []
                if model.generation_parameters.allow_newlines
                else [
                    t
                    for t in range(len(model.tokenizer))
                    if "\n" in model.tokenizer.decode([t])
                ]
            ),
        }

        with torch.no_grad():
            out = model.generate(**batch, **generate_kwargs)
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
                    embeddings_decoder = embeddings_decoder.to(torch.float16)

        cut_logits = []
        cut_sequences = []
        full_texts = [] # UGRIP: New list with full greedy texts
        cut_texts = []
        cut_alternatives = []
        all_slice_start_indices = []
        failed_generation = [] # UGRIP: per-sample, True when the answer slice is unusable

        # Count generations whose answer marker could not be located (visibility
        # into silent mis-slicing); the resulting samples are dropped from the
        # batch downstream via the per-sample `failed_generation` flag.
        marker_not_found = 0

        eos_ids_tokenizer = getattr(model.tokenizer, "eos_token_id", None)
        if isinstance(eos_ids_tokenizer, int):
            eos_ids_tokenizer = [eos_ids_tokenizer]
        elif eos_ids_tokenizer is None:
            eos_ids_tokenizer = []

        eos_ids_model_config = None
        if getattr(model, "model", None) is not None:
            eos_ids_model_config = getattr(
                getattr(model.model, "config", None), "eos_token_id", None
            )
        if isinstance(eos_ids_model_config, int):
            eos_ids_model_config = [eos_ids_model_config]
        elif eos_ids_model_config is None:
            eos_ids_model_config = []

        eos_ids = list(set(eos_ids_tokenizer) | set(eos_ids_model_config))

        for i in range(len(texts)):
            if model.model_type == "CausalLM":
                idx = batch["input_ids"].shape[1]
                full_gen_seq = sequences[i, idx:].cpu()
            elif model.model_type == "vLLMCausalLM":
                full_gen_seq = sequences[i].cpu()
            else:
                full_gen_seq = sequences[i, 1:].cpu()

            # UGRIP: Code to save full text
            full_text_length = len(full_gen_seq)
            for j in range(len(full_gen_seq)):
                if full_gen_seq[j] in eos_ids:
                    full_text_length = j
                    break 
            full_texts.append(model.tokenizer.decode(full_gen_seq[:full_text_length]))
            # END UGRIP

            slice_start_idx = 0
            slice_end_idx = len(full_gen_seq)
            failed = False

            if self.slicing_target == "answer":
                a_start, a_end = self._marker_token_span(
                    full_gen_seq, model.tokenizer, self._answer_re
                )
                if a_start is not None:
                    slice_start_idx = a_end  # analyze tokens after the answer marker
                else:
                    # No answer marker -> no answer span. Flag the sample so the
                    # manager drops it from the batch; the empty slice below would
                    # otherwise yield degenerate estimator output.
                    slice_start_idx, slice_end_idx = 0, 0
                    marker_not_found += 1
                    failed = True
            elif self.slicing_target == "reasoning":
                a_start, a_end = self._marker_token_span(
                    full_gen_seq, model.tokenizer, self._answer_re
                )
                if a_start is not None:
                    slice_end_idx = a_start  # reasoning = tokens before the answer marker
                else:
                    marker_not_found += 1  # marker missing -> treat whole seq as reasoning
                # Strip a leading reasoning label ('### Reasoning:') so UQ is not
                # diluted by the boilerplate label tokens.
                if self._reasoning_re is not None:
                    r_start, r_end = self._marker_token_span(
                        full_gen_seq[:slice_end_idx], model.tokenizer, self._reasoning_re
                    )
                    if r_start is not None:
                        slice_start_idx = r_end


            all_slice_start_indices.append(slice_start_idx)
            seq = full_gen_seq[slice_start_idx:slice_end_idx]

            length, text_length = len(seq), len(seq)
            for j in range(len(seq)):
                if seq[j] in eos_ids:
                    length = j + 1
                    text_length = j
                    break

            final_seq_tokens = seq[:length].tolist()
            final_seq_text_tokens = seq[:text_length]

            # An answer slice that came out empty (marker present but nothing
            # generated after it) is as unusable as a missing marker.
            if self.slicing_target == "answer" and not final_seq_text_tokens.size(0):
                failed = True
            failed_generation.append(failed)

            cut_sequences.append(final_seq_tokens)
            cut_texts.append(model.tokenizer.decode(final_seq_text_tokens))

            cut_logits.append(logits[i, slice_start_idx : slice_start_idx + length, :].cpu().numpy())

            cut_alternatives.append([[] for _ in range(length)])
            for j in range(length):
                # Absolute offset
                lt = logits[i, j + slice_start_idx, :].cpu().numpy()
                best_tokens = np.argpartition(lt, -self.n_alternatives)
                ln = len(best_tokens)
                best_tokens = best_tokens[ln - self.n_alternatives : ln]
                for t in best_tokens:
                    cut_alternatives[-1][j].append((t.item(), lt[t].item()))
                cut_alternatives[-1][j].sort(
                    key=lambda x: str(x)[0] == str(final_seq_tokens[j]),
                    reverse=True,
                )

        lls = []
        for i in range(len(texts)):
            log_probs = cut_logits[i]
            tokens = cut_sequences[i]
            if len(tokens) == 0:
                lls.append([])
                continue
            assert len(tokens) == len(log_probs)

            lls.append([log_probs[j, tokens[j]] for j in range(len(log_probs))])

        attention_all = []
        attention_selected = []

        if self.output_attentions and (model.model_type != "vLLMCausalLM"):
            config = model.model.config
            if hasattr(config, 'text_config'):
                config = config.text_config
            for i in range(len(texts)):
                slice_start_idx = all_slice_start_indices[i]
                c = len(cut_sequences[i])
                attn_mask = np.zeros(shape=(
                    config.num_attention_heads * config.num_hidden_layers, c, c
                ))
                
                if c == 0: # Empty sequence
                    attention_all.append(attn_mask.max(0))
                    continue

                for j in range(1, c):
                    original_token_index = j + slice_start_idx
                    if original_token_index < len(attentions):
                        stacked_attention = torch.vstack([
                            attentions[original_token_index][layer][0][head][0][-j:]
                            for layer in range(len(attentions[original_token_index]))
                            for head in range(len(attentions[original_token_index][layer][0]))
                        ])
                        if stacked_attention.dtype == torch.bfloat16:
                            stacked_attention = stacked_attention.to(torch.float16)
                        attn_mask[:, j, :j] = stacked_attention.cpu().numpy()
                attention_all.append(attn_mask.max(0))



            num_layers = len(attentions[0])
            mid_layer = num_layers // 2
            selected_layers = [mid_layer, num_layers - 2, num_layers - 1]
            # selected_layers = [mid_layer]

            for i in range(len(texts)):
                input_len = batch["input_ids"].shape[1]
                slice_start_idx = all_slice_start_indices[i]
                c = len(cut_sequences[i])

                if c == 0:
                    attention_selected.append(None)
                    continue

                num_heads = attentions[0][selected_layers[0]].shape[1]
                # num_heads = 1

                # get actual total_key_len from the last valid attention step
                # last_t = min(slice_start_idx + c - 1, len(attentions) - 1)
                total_key_len = len(sequences[i]) - c - 1 # attentions[last_t][selected_layers[0]].shape[-1] - c # actual size

                attn_mask = np.zeros(shape=(len(selected_layers), num_heads, c, total_key_len))

                for j in range(c):
                    original_token_index = j + slice_start_idx
                    if original_token_index < len(attentions):
                        for li, layer in enumerate(selected_layers):
                            layer_attn = attentions[original_token_index][layer][0, :num_heads, 0, :total_key_len]  # (num_heads, key_len) 
                            if layer_attn.dtype == torch.bfloat16:
                                layer_attn = layer_attn.to(torch.float16)
                            key_len_at_j = layer_attn.shape[-1]
                            attn_mask[li, :num_heads, j, :key_len_at_j] = layer_attn.cpu().numpy()

                attention_selected.append(attn_mask)  # (3, 16, c, total_key_len)

                
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
        
        if self.slicing_target and marker_not_found:
            log.warning(
                f"slicing_target={self.slicing_target}: answer marker "
                f"({self.answer_marker!r}) not found in {marker_not_found}/{len(texts)} "
                f"generations in this batch."
            )

        result_dict = {
            "input_tokens": batch["input_ids"].to("cpu").tolist(),
            "greedy_log_probs": cut_logits,
            "greedy_tokens": cut_sequences,
            "greedy_tokens_alternatives": cut_alternatives,
            "greedy_texts": cut_texts,
            "greedy_texts_full": full_texts, # UGRIP: full text
            "failed_generation": failed_generation,
            "greedy_log_likelihoods": lls,
        }
        result_dict.update(embeddings_dict)
        if self.output_attentions:
            result_dict.update({"attention_all": attention_all})
            result_dict.update({"attention_selected": attention_selected})
            result_dict.update({"tokenizer": model.tokenizer})
        return result_dict
