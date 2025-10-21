'''
from scratch GPT model
'''

import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F

import tiktoken
from dataclasses import dataclass

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
    def __init__(self, config, is_decoder=True):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, batched together
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.is_decoder = is_decoder
        
        if is_decoder:
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
        y = F.scaled_dot_product_attention(q, k, v, is_causal=self.is_decoder) # (B, n_head, T, head_size)

        
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

    def __init__(self, config, is_decoder=True):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd) # ToDo: Understand
        self.attn = GPT2Attention(config, is_decoder=is_decoder)
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
  
  def configure_optimizers(self, weight_decay, learning_rate, device, verbose=False):
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
    if verbose: print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    if verbose: print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and used fused version if available in this version of PyTorch
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and 'cuda' in device
    if verbose: print(f"using fused AdamW: {use_fused}\n-----")
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused) #betas, eps match GPT3 paper
    return optimizer
        