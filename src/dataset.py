import re
from collections import Counter
from typing import Generator, List, Tuple
import numpy as np
from numpy.typing import NDArray

from config import MIN_WORD_FREQUENCY, WINDOW_SIZE

class Dataset:
    def __init__(self, raw_text: str, window_size: int = WINDOW_SIZE, word_min_frequency: int = MIN_WORD_FREQUENCY) -> None:        
        self.tokens = self.tokenize(raw_text)[:10000]
        print(self.tokens[:40])
        self.build_vocab(word_min_frequency)
        self.window_size = window_size

    def tokenize(self, raw_text: str) -> List[str]:
        """
        Tokenize the raw text into words, splitting on non-word characters
        """
        return re.findall(r'\b\w+\b', raw_text.lower())
    
    def build_vocab(self, word_min_frequency: int) -> None:
        """
        Create the vocabulary, mappings, and token frequencies, also filter out low frequency words
        """
        word_freq = Counter(self.tokens)

        # Create word to index and index to word mappings
        filtered_words = [(word, freq) for word, freq in word_freq.items() if freq >= word_min_frequency]
        self.word_to_idx = {word: idx for idx, (word, freq) in enumerate(filtered_words)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab_size = len(self.word_to_idx)

        # Save token indices and frequencies
        self.token_indices = [self.word_to_idx[token] for token in self.tokens if token in self.word_to_idx]
        self.vocab_freq = np.array([word_freq[self.idx_to_word[idx]] for idx in range(self.vocab_size)])
    
    def generate_pairs(self) -> Generator[Tuple[int, int], None, None]:
        """
        Generate the positive examples for the dataset
        """
        # Enumerate to make sure we don't include low frequency words in the pairs
        for idx, token_idx in enumerate(self.token_indices):
            for context_idx in range(max(0, idx - self.window_size), min(len(self.token_indices), idx + self.window_size + 1)):
                if context_idx != idx:
                    yield (token_idx, self.token_indices[context_idx])
    
    def batch_generator(self, generator: Generator[Tuple[int, int], None, None], batch_size: int) -> Generator[NDArray[np.int_], None, None]:
        """
        Utility function to create batches from a generator
        """
        batch = []
        for item in generator:
            batch.append(item)
            if len(batch) == batch_size:
                yield np.array(batch)
                batch = []

        if batch:
            yield np.array(batch)