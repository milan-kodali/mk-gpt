import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken

from mk_gpt import GPT, GPTConfig

# set sampling parameters
num_return_sequences = 4
max_length = 32

def sample_sequences(num_return_sequences, max_length, device, model = None):
  # load model
  if model is None:
    model = GPT.from_pretrained("gpt2")
  # model = GPT(GPTConfig())
  model.eval() # set model to evaluation mode
  model.to(device)

  # set up prefix tokens for sampling
  enc = tiktoken.get_encoding("gpt2")
  tokens = enc.encode("Hello, I'm a language model,")
  tokens = torch.tensor(tokens, dtype=torch.long)
  tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
  x = tokens.to(device) # (B = num_return_sequences, prefix_length)
  sample_rng = torch.Generator(device=device)
  sample_rng.manual_seed(42)
  # sample
  while(x.size(1) < max_length):
    with torch.no_grad():
      logits, loss = model(x) # (B, T, vocab_size)
      logits = logits[:, -1, :] # (B, vocab_size)
      probs = F.softmax(logits, dim = -1) # (B, vocab_size)
      # Add top-k sampling to match HF default
      topk_probs, topk_indices = torch.topk(probs, 50, dim = -1) # (B, 50) for both
      ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
      x_col = torch.gather(topk_indices, -1, ix) # (B, 1)
      x = torch.cat((x, x_col), dim = 1) # (B, T + 1)

  # decode
  for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(f"> {i}: {decoded}\n")

# sample from pretrained model
# sample_sequences(num_return_sequences, max_length, device)