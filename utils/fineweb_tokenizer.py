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

print("Hello from fineweb_tokenizer.py")