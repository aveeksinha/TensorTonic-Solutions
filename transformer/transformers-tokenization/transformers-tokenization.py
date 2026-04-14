import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # Reset vocab
        self.word_to_id = {}
        self.id_to_word = {}
        
        # Add special tokens with fixed IDs
        special_tokens = [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token
        ]
        
        for idx, token in enumerate(special_tokens):
            self.word_to_id[token] = idx
            self.id_to_word[idx] = token
        
        # Collect unique words from texts
        unique_words = set()
        for text in texts:
            words = text.lower().split()
            unique_words.update(words)
        
        # Sort words for deterministic IDs
        sorted_words = sorted(unique_words)
        
        # Add words after special tokens
        start_id = len(special_tokens)
        for i, word in enumerate(sorted_words):
            self.word_to_id[word] = start_id + i
            self.id_to_word[start_id + i] = word
        
        # Set vocab size
        self.vocab_size = len(self.word_to_id)
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        words = text.lower().split()
        unk_id = self.word_to_id[self.unk_token]
        
        return [
            self.word_to_id.get(word, unk_id)
            for word in words
        ]
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        return " ".join(
            self.id_to_word.get(i, self.unk_token)
            for i in ids
        )