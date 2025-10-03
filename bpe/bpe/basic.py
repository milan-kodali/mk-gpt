from .base import Tokenizer, get_stats, merge

class BasicTokenizer(Tokenizer):
  def __init__(self):
    super().__init__()
    self.vocab = self._build_vocab()

  def train(self, text, vocab_size, verbose=False):
    tokens = text.encode("utf-8") # raw bytes
    tokens = list(map(int, tokens)) # convert to a list of integers in range 0..255 for convenience
    idx_start = len(self.vocab)
    steps = vocab_size - idx_start
    
    for i in range(steps):
      stats = get_stats(tokens)  
      pair = max(stats, key=stats.get)
      idx = idx_start + i
      if verbose:
        print(f"merging {pair} into new token {idx}")
      tokens = merge(tokens, pair, idx)
      self.merges[pair] = idx
      
    self.vocab = self._build_vocab()

    if verbose:
      print(f"{self.merges = }")
      print(f"{self.vocab = }")

  def encode(self, text):
    tokens = list(map(int, text.encode("utf-8")))
    new_tokens = []
    for pair, idx in self.merges.items():
      i = 0
      while i < len(tokens):
        if i < len(tokens) -1 and pair == (tokens[i], tokens[i+1]):
          new_tokens.append(idx)
          i+=2
        else:
          new_tokens.append(tokens[i])
          i +=1
      tokens = new_tokens
      new_tokens = []
    return tokens

  def decode(self, ids):
    tokens = b"".join(self.vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors="replace")
    return text