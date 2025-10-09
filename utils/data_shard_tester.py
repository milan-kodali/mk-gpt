import numpy as np
import tiktoken
import os

shard_path = "../data/fineweb-edu10b/fineweb_edu10b_train_000010.npy"
start_token_index = 50
tokens_to_sample = 1000

script_dir = os.path.dirname(os.path.abspath(__file__))
data = np.load(os.path.join(script_dir, shard_path))

print(f"Shape: {data.shape}")
print(f"Dtype: {data.dtype}")
print(f"{tokens_to_sample} tokens starting from {start_token_index}: {data[start_token_index:start_token_index+tokens_to_sample]}")

enc = tiktoken.get_encoding("gpt2")
print(enc.decode(data[start_token_index:start_token_index+tokens_to_sample]))