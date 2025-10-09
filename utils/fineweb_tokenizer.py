'''
Downloads, tokenizes, and shards FineWeb-EDU dataset
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Run: python fineweb_tokenizer.py
Saves shards to /data/fineweb-edu
'''

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

#----
local_dir = "../data/fineweb-edu10b"
sample_name = "sample-10BT"

# create local cache dir if needed
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)


# download dataset
fw = load_dataset("HuggingFaceFW/fineweb-edu", name=sample_name, split="train")
print(f"Loaded dataset; size on disk: {fw.info.dataset_size / 1e9 :.2f} gb\n-----")

# init the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # 50256




print("Hello from fineweb_tokenizer.py\n-----")