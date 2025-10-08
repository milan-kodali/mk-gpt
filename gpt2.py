'''
from scratch GPT-2 model
'''

from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import tiktoken
import math

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

class DataLoaderLite:
  def __init__(self, B, T):
    self.B = B
    self.T = T

    # load tokens from disk and save to memory
    with open('inputs/shakespeare.txt', 'r') as f:
      data = f.read()
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(data)
    self.tokens = torch.tensor(tokens)
    print(f"Loaded {len(tokens)} tokens")
    print(f"1 epoch has {len(tokens) // (B * T)} batches\n-----")

    # state
    self.current_position = 0

  def next_batch(self):
    B, T = self.B, self.T
    buf = self.tokens[self.current_position:self.current_position + B*T + 1]
    x = buf[:-1].view(B, T)
    y = buf[1:].view(B, T)
    
    self.current_position += B*T
    # reset if next batch would be out of bounds
    if self.current_position + B*T + 1 > len(self.tokens):
      self.current_position = 0
    return x, y

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

  # sample
  while(x.size(1) < max_length):
    with torch.no_grad():
      logits, loss = model(x) # (B, T, vocab_size)
      logits = logits[:, -1, :] # (B, vocab_size)
      probs = F.softmax(logits, dim = -1) # (B, vocab_size)
      # Add top-k sampling to match HF default
      topk_probs, topk_indices = torch.topk(probs, 50, dim = -1) # (B, 50) for both
      ix = torch.multinomial(topk_probs, 1) # (B, 1)
      x_col = torch.gather(topk_indices, -1, ix) # (B, 1)
      x = torch.cat((x, x_col), dim = 1) # (B, T + 1)

  # decode
  for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(f"> {i}: {decoded}")


# ---------------------------------------------------
# ---------------------------------------------------

# auto-detect device
device = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device\n-----")

# Set seed across all device types
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    torch.mps.manual_seed(42)

# set sampling parameters
num_return_sequences = 5
max_length = 30

# sample from pretrained model
# sample_sequences(num_return_sequences, max_length, device)

'''
Get Data & Optimize
'''
# get data
B, T = 4, 1024 # small batch size for CPU training
if torch.cuda.is_available(): B = 96 # largest batch size that fits on A100 memory
train_loader = DataLoaderLite(B, T)

# Use TF32 for operations on FP32s
torch.set_float32_matmul_precision('high')
print("Initializing model\n-----")
# initialize model
model = GPT(GPTConfig(vocab_size=50304)) #only need 50257, but use a nicer number for performance
model.to(device)
# compile model for performance
if torch.cuda.is_available():
  print("Compiling model\n-----")
  model = torch.compile(model)
  torch.cuda.synchronize() # wait for all kernels to complete
  print("Model compiled\n-----")

# learning rate schedule based on GPT-3 paper
max_lr = 6e-4 #matches GPT-3 small
min_lr = max_lr * 0.1
warmup_steps = 10
decay_horizon = 40
max_steps = 50
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
optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), eps=1e-8) #betas, eps match GPT3 paper

for step in range(max_steps):
  # timing
  t0 = time.time()

  x, y = train_loader.next_batch()
  x, y = x.to(device), y.to(device)
  optimizer.zero_grad(set_to_none=True)
  # Use autocasting to cast some operations to BF16
  with torch.autocast(device_type=device, dtype=torch.bfloat16):
    logits, loss = model(x, y) 

  loss.backward()
  # gradient clipping matching GPT-3 paper
  norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
  # determine & set learning rate for this iteration; step
  lr = get_lr(step)
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr
  optimizer.step()
  # timing
  if torch.cuda.is_available():
    torch.cuda.synchronize() # wait for all kernels to complete
  t1 = time.time()
  dt = t1 - t0
  tokens_per_sec = (train_loader.B * train_loader.T) / dt
  
  if step % (max_steps//50)== 0 or step == max_steps - 1:
    print(f"step {step}: | loss {loss.item():.4f} | lr {lr:.2e} | norm {norm:.3f} | dt {dt:.2f}s | tps {int(tokens_per_sec)}")

sample_sequences(num_return_sequences, max_length, device, model)