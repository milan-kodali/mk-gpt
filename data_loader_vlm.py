"""
DataLoader for VLM training from pre-saved finevision-sharegpt4vcoco shards
See finevision_downloader.py for more details.

Note: -100 is used for ignored positions in loss calculation for padding tokens
"""

import os
import torch
import tiktoken

def load_shard(filename):
  shard = torch.load(filename)
  return shard['images'], shard['labels']

class DataLoaderVLM:
  def __init__(self, B, T, rank=0, world_size=1, split='train', prefix="This is an image of", verbose=True):
    self.B = B
    self.T = T
    self.rank = rank
    self.world_size = world_size
    self.verbose = verbose
    self.prefix = prefix
    
    self.enc = tiktoken.get_encoding("gpt2")
    
    assert split in ['train', 'val'], "split must be either 'train' or 'val'"
    self.split = split
    
    # Find all shards for this split
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache', 'data', 'finevision-sharegpt4vcoco_')
    shards = [f for f in os.listdir(data_dir) if f.startswith(f'finevision_sharegpt4vcoco_{self.split}') and f.endswith('.pt')]
    shards = [os.path.join(data_dir, f) for f in shards]
    self.shards = sorted(shards)
    self.shard_count = len(self.shards)
    assert self.shard_count > 0, f"no shards found for {split} split"
    if self.verbose: 
      print(f"Found {len(self.shards)} shards for {split} split")
    
    # Set initial state
    self.reset()
  
  def reset(self):
    self.shard_index = 0
    self.images, self.labels = load_shard(self.shards[self.shard_index])
    self.shard_size = len(self.labels)
    self.current_position = self.rank * self.B
    if self.verbose:
      print(f"Loaded {self.split} shard {self.shard_index} with {self.shard_size} samples")
  
  def next_batch(self):
    B, T = self.B, self.T
    
    # Get batch of images and labels
    batch_images = []
    batch_input_tokens = []
    batch_target_tokens = []
    
    for _ in range(B):
      # Check if we need to load next shard BEFORE accessing data
      if self.current_position >= self.shard_size:
        self.next_shard()
        self.current_position = self.rank * self.B
      
      # Get current image and label
      img = self.images[self.current_position]
      label = self.labels[self.current_position]
      
      # Prepare full text with prefix
      full_text = self.prefix + label
      full_tokens = self.enc.encode(full_text)
      
      # Create input and target sequences (shifted by 1 for autoregressive training)
      # Input: tokens[:-1], Target: tokens[1:]
      input_tokens = full_tokens[:-1]
      target_tokens = full_tokens[1:]
      
      # Truncate or pad to T
      if len(input_tokens) > T:
        input_tokens = input_tokens[:T]
        target_tokens = target_tokens[:T]
      else:
        # Pad input tokens with 0
        pad_len = T - len(input_tokens)
        input_tokens = input_tokens + [0] * pad_len
        # Pad target tokens with -100 (ignored in loss)
        target_tokens = target_tokens + [-100] * pad_len
      
      batch_images.append(img)
      batch_input_tokens.append(torch.tensor(input_tokens, dtype=torch.long))
      batch_target_tokens.append(torch.tensor(target_tokens, dtype=torch.long))
      
      # Move to next sample for next iteration
      self.current_position += self.world_size
    
    # Stack into batches
    images = torch.stack(batch_images)
    input_tokens = torch.stack(batch_input_tokens)
    target_tokens = torch.stack(batch_target_tokens)
    
    return images, input_tokens, target_tokens
  
  def next_shard(self):
    self.shard_index = (self.shard_index + 1) % self.shard_count
    if self.verbose: 
      print(f"-----\nLoading {self.split} shard {self.shard_index} of {self.shard_count} for GPU {self.rank}\n-----")
    self.images, self.labels = load_shard(self.shards[self.shard_index])
    self.shard_size = len(self.labels)
  