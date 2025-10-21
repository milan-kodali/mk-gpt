import os
import numpy as np
import torch

def load_tokens(filename):
  tokens = np.load(filename)
  return torch.tensor(tokens, dtype=torch.long)

class DataLoaderLite:
  def __init__(self, B, T, rank, world_size, split = 'train', verbose = True):
    self.B = B
    self.T = T
    self.rank = rank
    self.world_size = world_size
    self.verbose = verbose

    assert split in ['train', 'val'], "split must be either 'train' or 'val'"
    self.split = split

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache', 'data', 'fineweb-edu10b')
    shards = [f for f in os.listdir(data_dir) if f.startswith(f'fineweb_edu10b_{self.split}') and f.endswith('.npy')]
    shards = [os.path.join(data_dir, f) for f in shards]
    self.shards = sorted(shards)
    self.shard_count = len(self.shards)
    assert self.shard_count > 0, f"no shards found for {split} split"
    if self.verbose: print(f"Found {len(self.shards)} shards for {split} split")

    # set initial state
    self.reset()
    
  def reset(self):
    self.shard_index = 0
    self.tokens = load_tokens(self.shards[self.shard_index])
    self.current_position = self.rank * self.B * self.T
  
  def next_batch(self):
    B, T = self.B, self.T
    buf = self.tokens[self.current_position:self.current_position + B*T + 1]
    x = buf[:-1].view(B, T)
    y = buf[1:].view(B, T)
    self.step(1)
    return x, y

  def next_shard(self):
    self.shard_index = (self.shard_index + 1) % self.shard_count
    if self.verbose: print(f"-----\nLoading {self.split} shard {self.shard_index} of {self.shard_count} for GPU {self.rank}\n-----")
    self.tokens = load_tokens(self.shards[self.shard_index])

  def step(self, num_steps = 1):
    for i in range(num_steps):
      B, T, world_size, rank = self.B, self.T, self.world_size, self.rank
      self.current_position += B * T * world_size
      # reset if next batch would be out of bounds
      next_position = self.current_position + B * T * world_size + 1
      if next_position > len(self.tokens):
        self.next_shard()
        self.current_position = rank * B * T