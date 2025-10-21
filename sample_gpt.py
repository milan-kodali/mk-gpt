import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import argparse
import os

from mk_gpt import GPT, GPTConfig

# set sampling parameters
num_return_sequences = 4
max_length = 32
default_prefix = "Hello, I'm a language model,"
default_device = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"

def sample_sequences(num_return_sequences, max_length, device, model, prefix = default_prefix):
  # load model
  model.eval() # set model to evaluation mode
  model.to(device)

  # set up prefix tokens for sampling
  enc = tiktoken.get_encoding("gpt2")
  tokens = enc.encode(prefix)
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

def sample_pretrained(num_return_sequences, max_length, checkpoint_file_name, device = default_device, prefix = default_prefix):
  checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache', 'checkpoints')
  model_file_path = os.path.join(checkpoint_dir, checkpoint_file_name)
  if os.path.exists(model_file_path):
    print(f"Loading model weights from {checkpoint_file_name}\n-----")
    checkpoint = torch.load(model_file_path)
    model = GPT(GPTConfig(vocab_size=50304))
    model.to(device)
    torch.set_float32_matmul_precision('high')
    if "cuda" in device:
      model = torch.compile(model)
    model.load_state_dict(checkpoint['model'])
    sample_sequences(num_return_sequences, max_length, device, model, prefix)
  else:
    print("No saved model found exiting\n-----")
    return

def sample_gpt2(num_return_sequences, max_length, version = "gpt2", device = default_device, prefix = default_prefix):
  model = GPT.from_pretrained(version)
  sample_sequences(num_return_sequences, max_length, device, model, prefix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample from pretrained model or open GPT-2 models")
    parser.add_argument("-v", "--version", type=str, default="gpt2", help="GPT-2 version to sample from (default: gpt2)")
    parser.add_argument("-c", "--checkpoint", type=str, default=None, help="Checkpoint file to sample from")
    parser.add_argument("-p", "--prefix", type=str, default=default_prefix, help="Prefix to sample with")
    parser.add_argument("-n", "--num-return-sequences", type=int, default=num_return_sequences, help="Number of sequences to return (default: 4)")
    parser.add_argument("-l", "--max-length", type=int, default=max_length, help="Maximum length of sequences to sample (default: 32)")
    args = parser.parse_args()
    if args.checkpoint:
      sample_pretrained(args.num_return_sequences, args.max_length, args.checkpoint, default_device, args.prefix)
    else:
      sample_gpt2(args.num_return_sequences, args.max_length, args.version, default_device, args.prefix)