'''
Version of fineweb_tokenizer.py that creates mini shards for testing dataloader
'''

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

#----
local_dir = "../data/mini_shards"
sample_name = "sample-10BT"
shard_size = int(2e6)
nprocs = max(1, os.cpu_count())

# set data cache dir
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
# init the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # 50256
# tokenizes a doc and returns a numpy array of uint16 tokens
def tokenize(doc):
  tokens = [eot]
  tokens.extend(enc.encode_ordinary(doc["text"]))
  assert all(0 <= token <= 2**16 for token in tokens), "Token out of range"
  tokens_np = np.array(tokens, dtype=np.uint16)
  return tokens_np

def write_datafile(filename, tokens_np):
  np.save(filename, tokens_np)

if __name__ == "__main__":
  # create local cache dir if needed
  os.makedirs(DATA_CACHE_DIR, exist_ok=True)

  # download dataset
  fw = load_dataset("HuggingFaceFW/fineweb-edu", name=sample_name, split="train")
  print(f"Loaded dataset; size on disk: {fw.info.dataset_size / 1e9 :.2f} gb\n-----")
  # tokenize all docs and write output shards of shard_size tokens
  with mp.Pool(nprocs) as pool:
    shard_index = 0
    # preallocate current shard buffer
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None

    for tokens in pool.imap(tokenize, fw, chunksize = 16):
      
      # check if current shard has space
      if token_count + len(tokens) < shard_size:
        # copy tokens to current shard buffer in correct position 
        all_tokens_np[token_count:token_count + len(tokens)] = tokens
        token_count += len(tokens)
        # update progress bar
        if progress_bar is None:
          progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
        progress_bar.update(len(tokens))
      else:
        # write current shard to disk
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"fineweb_edu10b_{split}_{shard_index:06d}")
        # split doc into whatever fits in current shard, drop rest in next shard
        remaining_space = shard_size - token_count
        progress_bar.update(remaining_space)
        all_tokens_np[token_count:token_count+remaining_space] = tokens[:remaining_space]
        write_datafile(filename, all_tokens_np)
        shard_index += 1
        if shard_index == 10:
          print(f"Reached 10 shards, stopping")
          break
        progress_bar = None
        # drop remainder in next shard
        all_tokens_np[0:len(tokens) - remaining_space] = tokens[remaining_space:]
        token_count = len(tokens) - remaining_space

    # write final remaining tokens as last shard
    if token_count > 0:
      split = "val" if shard_index == 0 else "train"
      filename = os.path.join(DATA_CACHE_DIR, f"fineweb_edu10b_{split}_{shard_index:06d}")
      write_datafile(filename, all_tokens_np[:token_count])