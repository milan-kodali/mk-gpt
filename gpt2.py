'''
from scratch GPT-2 model
'''

from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import tiktoken
import math
import inspect
import os
import numpy as np
import sys


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    # dropout: float = 0.1

class GPT2Attention(nn.Module):
    """ multiple self-attention heads in parallel """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, batched together
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)) 


    def forward(self, x):
        B, T, C = x.shape # batch size, sequence length, n_embd
        # calculate query, key, value for all heads in a batch
        # C = n_head * head_size, eg n_head = 12, head_size = 64, so C = 768
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, -1).transpose(1, 2) #(B, T, n_head, head_size) -> (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, -1).transpose(1, 2) #(B, T, n_head, head_size) -> (B, n_head, T, head_size)
        v = v.view(B, T, self.n_head, -1).transpose(1, 2) #(B, T, n_head, head_size) -> (B, n_head, T, head_size)

        # compute attention scores (replaced with flash attention below)
        # wei = q @ k.transpose(-2, -1) * (k.size(-1))**-0.5 #(B,n_head,T,head_size) @ (B,n_head,head_size,T) -> (B,n_head,T,T)
        # wei = wei.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # wei = wei.softmax(dim = -1) # (B, n_head, T, T)
        # # wei = self.dropout1(wei) # (B, n_head, T, T)
        # y = wei @ v  # (B, n_head, T, T) @ (B, n_head, T, head_size) -> (B, n_head, T, head_size)
        
        # use flash attention (algorithmically identical) for ~50% speedup
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # (B, n_head, T, head_size)

        
        y = y.transpose(1, 2).reshape(B, T, -1) # (B, n_head, T, head_size) -> (B, T, n_head * head_size)

        y = self.c_proj(y) 
        return y    

class GPT2MLP(nn.Module):
    """ Linear layer + non-linearity to add compute after multi-head attention layer """

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd) # expand onto higher dimensional space
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd) # equivalent to self.proj in MHA layer, project back down to model's embedding dimensionality 
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # nn.Dropout(config.dropout),

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class GPT2Block(nn.Module):
    """ Transformer block: Communication followed by computation, with residual connection (x +) """ 

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd) # ToDo: Understand
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd) # ToDo: Understand
        self.mlp = GPT2MLP(config) 

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):

  def __init__(self, config: GPTConfig):
    super().__init__()
    self.config = config
    self.transformer = nn.ModuleDict(dict(
      wte = nn.Embedding(config.vocab_size, config.n_embd),
      wpe = nn.Embedding(config.block_size, config.n_embd),
      # drop = nn.Dropout(config.dropout)
      h = nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)]),
      ln_f = nn.LayerNorm(config.n_embd),
    ))
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    # weight sharing scheme from GPT-2 / AIAYN paper
    self.transformer.wte.weight = self.lm_head.weight

    # mirror GPT-2 weight initialization
    self.apply(self._init_weights)

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      std = 0.02
      if hasattr(module, 'NANOGPT_SCALE_INIT'):
        std *= (2 * self.config.n_layer) ** -0.5
      torch.nn.init.normal_(module.weight, mean=0.0, std=std)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    # Default LayerNorm init in pytorch matches GPT2, so no need to change

  def forward(self, idx, targets=None):
    B, T = idx.shape
    assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
      
    # idx and targets are both (B, T) tensor of integers
    token_emb = self.transformer.wte(idx) # (B, T, C = n_embd)
    pos_emb = self.transformer.wpe(torch.arange(T, device=idx.device)) # (T, C = n_embd)
    x = token_emb + pos_emb # (B, T, C = n_embd)
    for block in self.transformer.h:
      x = block(x) # (B, T, C = n_embd)
    x = self.transformer.ln_f(x)
    logits = self.lm_head(x) # (B, T, vocab_size)
    loss = None
    
    if targets is not None:
      B, T, C = logits.shape
      loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))

    return logits, loss

  @classmethod
  def from_pretrained(cls, model_type):
    """ Loads pretrained GPT-2 weights """
    assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
    from transformers import GPT2LMHeadModel
    print("loading weights from pretrained gpt %s" % model_type + "\n-----")

    # n_layer, n_head and n_embd are determined from model type
    config_args = {
      "gpt2":         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
      "gpt2-medium":  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
      "gpt2-large":   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
      "gpt2-xl":      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
    }[model_type]
    config_args["vocab_size"] = 50257 # always 50257 for GPT model
    config_args["block_size"] = 1024 # always 1024 for GPT model

    # create a from-scratch initialized minGPT model
    config = GPTConfig(**config_args)
    model = GPT(config)
    sd = model.state_dict()
    sd_keys = sd.keys()
    sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")] # discard this mask / buffer

    # init a huggingface/transformers model
    model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    sd_hf = model_hf.state_dict()

    # copy while ensuring all of the parameters are aligned & match in name/shape
    sd_keys_hf = sd_hf.keys()
    
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")]
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")]
    transposed = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]
    
    assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
    for k in sd_keys_hf:
      if any(k.endswith(w) for w in transposed):
        # special treatment for the Conv1D weights we have to transpose for nn.Linear
        assert sd_hf[k].shape[::-1] == sd[k].shape
        with torch.no_grad():
          sd[k].copy_(sd_hf[k].t())
      else:
        assert sd_hf[k].shape == sd[k].shape
        with torch.no_grad():
          sd[k].copy_(sd_hf[k])
    return model
  
  def configure_optimizers(self, weight_decay, learning_rate, device):
    # get all candidate parameters that require grad 
    param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
    # split into optim groups. 2D params (weight tensors in matmuls, embeddings) decay
    # 1D parames (biases, layernorms) don't decay
    decay_params = [p for p in param_dict.values() if p.dim() >= 2]
    nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
    optim_groups = [
      {"params": decay_params, "weight_decay": weight_decay},
      {"params": nodecay_params, "weight_decay": 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and used fused version if available in this version of PyTorch
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and 'cuda' in device
    print(f"using fused AdamW: {use_fused}\n-----")
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused) #betas, eps match GPT3 paper
    return optimizer

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

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'storage', 'data', 'fineweb-edu10b')
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


"""
Run the training loop
"""

# set up DDP using env variables set by torchrun (RANK, LOCAL_RANK and WORLD_SIZE)
# use `torchrun --standalone --nproc_per_node=4 gpt2.py third_attempt.pt` to run on 4 GPUs
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

# set sampling parameters
num_return_sequences = 4
max_length = 32

# sample from pretrained model
# sample_sequences(num_return_sequences, max_length, device)

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
optimizer = model.configure_optimizers(weight_decay = 0.1, learning_rate = 6e-4, device = device)

model_file_name = "mkgpt2.pt"
checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'storage', 'checkpoints')
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
    val_loader.step(start_step * grad_accum_steps)
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