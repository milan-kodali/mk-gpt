from .base import Tokenizer, get_stats, merge
import regex as re

class RegexTokenizer(Tokenizer):
  def __init__(self):
    super().__init__()
    self.vocab = self._build_vocab()
    GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
    self.pattern = GPT4_SPLIT_PATTERN

  def get_token_sets(self, text):
    # chunk by regex
    chunks = re.findall(self.pattern, text)
    # get list of integers in range 0..255 for each chunk
    token_sets = []
    for chunk in chunks:
      token_set = chunk.encode("utf-8") # get raw bytes
      token_sets.append(list(map(int, token_set))) # convert to a list of integers for convenience
    return token_sets
    
  def train(self, text, vocab_size, verbose=False):
    # chunk text & tokenize
    token_sets = self.get_token_sets(text)
    # training loop params
    idx_start = len(self.vocab)
    steps = vocab_size - idx_start
    # training loop
    for i in range(steps):
      # Get stats
      stats = {}
      for token_set in token_sets:
        stats = get_stats(token_set, stats)
      pair = max(stats, key=stats.get)
      idx = idx_start + i
      if verbose:
        print(f"merging {pair} into new token {idx}")
      # Perform merge for each token_set  
      new_token_sets = []
      for token_set in token_sets:
        new_token_sets.append(merge(token_set, pair, idx)) 
      # Replace token_sets with merged copy, remember the merge
      token_sets = new_token_sets
      self.merges[pair] = idx

    # update vocab based on merges
    self.vocab = self._build_vocab()
    if verbose:
      print(f"{self.merges = }")
      print(f"{self.vocab = }")

  def encode(self, text):
    # chunk text & tokenize
    token_sets = self.get_token_sets(text)
    new_token_sets = []

    for token_set in token_sets:
      new_token_set = []
      for pair, idx in self.merges.items():
        i = 0
        while i < len(token_set):
          if i < len(token_set) -1 and pair == (token_set[i], token_set[i+1]):
            new_token_set.append(idx)
            i+=2
          else:
            new_token_set.append(token_set[i])
            i +=1
        token_set = new_token_set
        new_token_set = []
      new_token_sets.append(token_set)

    return [token for token_set in new_token_sets for token in token_set]

  def decode(self, ids):
    tokens = b"".join(self.vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors="replace")
    return text