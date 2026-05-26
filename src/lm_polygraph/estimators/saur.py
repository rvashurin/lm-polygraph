import numpy as np
import nltk
from typing import Dict, List
from transformers import AutoTokenizer
from lm_polygraph.estimators.estimator import Estimator

# download NLTK punkt isfor sentence splitting if not th
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class SAUR(Estimator):
    def __init__(self, tokenizer_path: str, k: int = 150, min_len: int = 10):
        # get greedy_tokens and greedy_log_probs from the stat_calculators
        super().__init__(['greedy_tokens', 'greedy_log_probs'], 'sequence')
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.k = k
        self.min_len = min_len

    def __str__(self):
        return f'SAUR(k={self.k}, min_len={self.min_len})'

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculates the SAUR metric for a batch of generations. (higher = more uncertain)
        SAUR calculates the mean log-prob of the least confident sentences (which is a large negative number),
        so we negate it at the end so higher positive values = higher uncertainty.
        """
        batch_tokens = stats['greedy_tokens']
        batch_log_probs = stats['greedy_log_probs']
        
        uncertainties = []

        for tokens, log_probs in zip(batch_tokens, batch_log_probs):
            # Built sentences to align tokens w/text
            sentences_log_probs = []
            current_sentence_log_probs = []
            current_text = ""
            
            for token, log_prob in zip(tokens, log_probs):
                current_sentence_log_probs.append(log_prob)
                
                # Decode token, add to current string
                token_text = self.tokenizer.decode([token], skip_special_tokens=True)
                current_text += token_text
                
                # Check if the current accumulated text forms a complete sentence
                splits = nltk.sent_tokenize(current_text)
                
                # if NLTK splits it into >1 sentence, we're at a boundary
                if len(splits) > 1:
                    # Save the log probs of the last part (a full sentence)
                    # Keep the last split as a start of the next sentence
                    sentences_log_probs.append(current_sentence_log_probs[:-1])
                    
                    # Reset tracking variables for the new sentence
                    current_sentence_log_probs = [current_sentence_log_probs[-1]]
                    current_text = splits[-1]
            
            # Catch any remaining tokens as the final sentence
            if current_sentence_log_probs:
                sentences_log_probs.append(current_sentence_log_probs)

            # Merge short sentences (using min_len hyperparameter)
            merged_sentences = []
            for sent_probs in sentences_log_probs:
                if len(sent_probs) < self.min_len and len(merged_sentences) > 0:
                    # Merge with the previous sentence
                    merged_sentences[-1].extend(sent_probs)
                else:
                    merged_sentences.append(sent_probs)

            # Calculate the mean log-prob per sentence
            sentence_means = [np.mean(sent) for sent in merged_sentences if len(sent) > 0]
            
            if not sentence_means:
                uncertainties.append(np.nan)
                continue

            # Sort ascending (most uncertain first)
            sentence_means.sort()

            # Take the mean of the top k (or fewer, if the response has < k sentences)
            k_prime = min(self.k, len(sentence_means))
            saur_score = np.mean(sentence_means[:k_prime])
            
            # Negate so that higher score means higher uncertainty 
            uncertainties.append(-saur_score)

        return np.array(uncertainties)