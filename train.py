"""
Run the training loop
"""

import os
import sys
import time
import math
import inspect
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from data_loader import DataLoaderLite
from mk_gpt import GPT, GPTConfig, sample_sequences
from sample import sample_sequences

# set sampling parameters
num_return_sequences = 4
max_length = 32

# set up DDP using env variables set by torchrun (RANK, LOCAL_RANK and WORLD_SIZE)
# use `torchrun --standalone --nproc_per_node=4 train.py third_attempt.pt` to run on 4 GPUs
# GPU count must be power of 2 for batch size assumptions
ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
  # current DDP requires CUDA
  assert torch.cuda.is_available(), "DDP assumes CUDA is available"
  init_process_group(backend="nccl")
  ddp_rank = int(os.environ.get("RANK", -1))
  ddp_local_rank = int(os.environ.get("LOCAL_RANK", -1))
  ddp_world_size = int(os.environ.get("WORLD_SIZE", -1))
  device = f'cuda:{ddp_local_rank}'
  torch.cuda.set_device(device) # does logging, checkpointing, etc.
  print(f"DDP running as rank {ddp_rank}, local rank {ddp_local_rank}, world size {ddp_world_size}\n-----")
  print(f"Using {device} device\n-----")
  master_process = ddp_rank == 0
else:
  ddp_rank = 0
  ddp_local_rank = 0
  ddp_world_size = 1
  master_process = True
  # auto-detect device
  device = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
  print(f"Using {device} device\n-----")

# Set seed across all device types
torch.manual_seed(42)
if "cuda" in device:
    torch.cuda.manual_seed_all(42)
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    torch.mps.manual_seed(42)


"""
Get Data & Optimize
"""

# batch hyperparametrs, using gradient accumulation on single GPU
total_batch_size = 524288 # 2^19, matching GPT-3 batch size, in # of tokens
T = 1024 # sequence length
B = 4 # small micro-batch size for CPU training
if "cuda" in device: B = 64 # largest power-of-2 micro-batch size that fits on A100 
assert total_batch_size % (B * T * ddp_world_size) == 0, "total_batch_size must be divisible by (B * T * ddp_world_size)"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process: print(f"Using grad accumulation w/ {grad_accum_steps} steps per batch of {total_batch_size} tokens\n-----")

train_loader = DataLoaderLite(B=B, T=T, rank=ddp_rank, world_size=ddp_world_size, split='train', verbose=master_process)
val_loader = DataLoaderLite(B=B, T=T, rank=ddp_rank, world_size=ddp_world_size, split='val', verbose=master_process)

def evaluate(model, device, val_loader):
  model.eval()
  val_loader.reset()
  with torch.no_grad():
    val_loss_accum = 0.0
    val_loss_steps = 20
    for _ in range(val_loss_steps):
      x, y = val_loader.next_batch()
      x, y = x.to(device), y.to(device)
      with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)
      loss = loss / val_loss_steps
      val_loss_accum += loss.detach()
  if ddp:
    dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
  if master_process:
    print(f"-----\nval loss: {val_loss_accum.item():.4f}\n-----")


# Use TF32 for operations on FP32s
torch.set_float32_matmul_precision('high')
if master_process: print("Initializing model\n-----")
# initialize model
model = GPT(GPTConfig(vocab_size=50304)) #only need 50257, but use a nicer number for performance
model.to(device)
# compile model for performance
if "cuda" in device:
  if master_process: print("Compiling model\n-----")
  model = torch.compile(model)
  torch.cuda.synchronize() # wait for all kernels to complete
  if master_process: print("Model compiled\n-----")

# learning rate schedule based on GPT-3 paper
max_lr = 6e-4 # matches GPT-3 small
min_lr = max_lr * 0.1 # matches GPT-3 small
warmup_steps = 715 #375e6 warmup tokens / 524288 batch size = 715 steps
max_steps = 19073 #10e9 tokens / 524288 batch size = 19073 steps
decay_horizon = max_steps # doesn't match proportion of gpt-3 decay horizon
start_step = 0
checkpoint_interval = 100

def get_lr(step):
  # start with linear warmup
  if step < warmup_steps:
    return max_lr * (step + 1) / warmup_steps
  # end with min_lr
  if step > decay_horizon:
    return min_lr
  # use cosine decay in between
  decay_ratio = (step - warmup_steps) / (decay_horizon - warmup_steps)
  assert 0 <= decay_ratio <= 1
  coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
  return min_lr + coeff * (max_lr - min_lr)

# optimize
optimizer = model.configure_optimizers(weight_decay = 0.1, learning_rate = 6e-4, device = device, verbose = master_process)

model_file_name = "mkgpt2.pt"
checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache', 'checkpoints')
if len(sys.argv) > 1:  
  model_file_name = sys.argv[1]
  model_file_path = os.path.join(checkpoint_dir, model_file_name)
  if os.path.exists(model_file_path):
    if master_process: print(f"Loading model weights from {model_file_name}\n-----")
    checkpoint = torch.load(model_file_path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_step = checkpoint['step'] + 1
    train_loader.step(start_step * grad_accum_steps)
    if master_process: print(f"Starting from step {start_step}\n-----")
  else:
    if master_process: print("No saved model found, starting from scratch\n-----")

# wrap model in DDP if applicable
if ddp:
  model = DDP(model, device_ids=[ddp_local_rank])

for step in range(start_step, max_steps):
  # timing
  t0 = time.time()
  model.train()
  optimizer.zero_grad(set_to_none=True)
  #track loss across micro-batches
  accum_loss = 0
  # gradient accumulation
  for micro_step in range(grad_accum_steps):
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    # Use autocasting to cast some operations to BF16
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
      logits, loss = model(x, y) 
    # loss should be averaged across batch size, so need to divide by grad_accum_steps to avoid oversized gradients
    loss = loss / grad_accum_steps
    accum_loss += loss.detach()
    if ddp:
      model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
    loss.backward()
  if ddp:
    dist.all_reduce(accum_loss, op=dist.ReduceOp.AVG)
  # gradient clipping matching GPT-3 paper
  norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
  # determine & set learning rate for this iteration; step
  lr = get_lr(step)
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr

  optimizer.step()
  # timing
  if "cuda" in device:
    torch.cuda.synchronize() # wait for all kernels to complete
  t1 = time.time()
  dt = t1 - t0
  tokens_per_sec = (train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size) / dt
  
  if master_process: print(f"step {step}: | train_loss {accum_loss.item():.4f} | lr {lr:.2e} | norm {norm:.3f} | dt {dt:.2f}s | tps {int(tokens_per_sec)}")
  if step % checkpoint_interval == 0:
    evaluate(model, device, val_loader)
    if step > 0 and master_process:
      # save checkpoint and sample sequences
      model_to_save = model.module if ddp else model
      torch.save({'model': model_to_save.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step}, os.path.join(checkpoint_dir, model_file_name))
      sample_sequences(num_return_sequences, max_length, device, model)
      
if ddp:
  destroy_process_group()