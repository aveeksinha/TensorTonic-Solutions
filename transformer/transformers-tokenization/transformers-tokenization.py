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
        # 1. Start with special tokens in the fixed order
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        
        # 2. Extract unique words from texts, lowercased and split by whitespace
        unique_words = set()
        for text in texts:
            words = text.lower().split()
            unique_words.update(words)
        
        # 3. Sort unique words for deterministic IDs
        sorted_words = sorted(list(unique_words))
        
        # 4. Combine and build the mappings
        full_vocab = special_tokens + sorted_words
        
        for i, word in enumerate(full_vocab):
            self.word_to_id[word] = i
            self.id_to_word[i] = word
            
        self.vocab_size = len(full_vocab)
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        words = text.lower().split()
        unk_id = self.word_to_id.get(self.unk_token)
        
        return [self.word_to_id.get(word, unk_id) for word in words]
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        unk_word = self.unk_token
        
        # Map IDs back to words, defaulting to <UNK> if the ID is missing
        decoded_words = [self.id_to_word.get(i, unk_word) for i in ids]
        
        return " ".join(decoded_words)