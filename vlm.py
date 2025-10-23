'''
From scratch VLM model, using mk_gpt for language decoder
Needs quite a bit of work 😅
'''

import inspect
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms

from PIL import Image
import requests
from dataclasses import dataclass, field
import tiktoken

from mk_gpt import GPT, GPTConfig, GPT2Block
from data_loader_vlm import DataLoaderVLM

@dataclass
class VLMConfig:
  img_size: int = 384
  patch_size: int = 16
  n_embd: int = 768
  n_head: int = 12
  n_layer: int = 12
  block_size: int = 1024
  dropout: float = 0.1
  gpt_config: GPTConfig = field(default_factory=lambda: GPTConfig(vocab_size=50304))

class PatchEmbedding(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.img_size = config.img_size
    self.patch_size = config.patch_size
    assert self.img_size % self.patch_size == 0
    self.n_patch = (self.img_size // self.patch_size)**2 # N**2 = (96/16)**2 = 36
    self.conv = nn.Conv2d(in_channels=3, out_channels=config.n_embd, kernel_size=self.patch_size, stride=self.patch_size)

  def forward(self, x):
    x = self.conv(x) # [B, C=3, S=96, S=96] -> [B, C=512, N=6, N=6] 
    x = x.flatten(2) # [B, C=512, N=6, N=6] -> [B, C=512, N**2=T=36] 
    x = x.transpose(1,2)  #[B, C, T] -> [B, T=36, C=512] 

    return x

class ViT(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.patch_embedding = PatchEmbedding(config)
    # Learnable classification token
    self.cls_token = nn.Parameter(torch.zeros(1, 1, config.n_embd))
    assert config.img_size % config.patch_size == 0
    n_patch = (config.img_size // config.patch_size)**2
    # Learnable position embedding
    self.pos_embedding = nn.Parameter(torch.randn(1, n_patch+1, config.n_embd))
    self.dropout = nn.Dropout(config.dropout)
    self.blocks = nn.ModuleList([GPT2Block(config, is_decoder=False) for _ in range(config.n_layer)])
    self.ln = nn.LayerNorm(config.n_embd)

  def forward(self, x):
    x = self.patch_embedding(x)
    #concat the classification token
    cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)
    x += self.pos_embedding
    x = self.dropout(x)
    for block in self.blocks: 
      x = block(x)
    # Apply LayerNorm to classification token
    x = self.ln(x[:, 0])
    return x

class MultiModalProjector(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.proj = nn.Sequential(
        nn.Linear(config.n_embd, 4 * config.n_embd), # expand onto higher dim space 
        nn.GELU(approximate='tanh'),
        nn.Linear(4 * config.n_embd, config.gpt_config.n_embd), # project back down to embd dim space
        nn.Dropout(config.dropout)
    )
  
  def forward(self, x):
    x = self.proj(x)
    return x

class GPTWithImages(GPT):
  def __init__(self, config):
    super().__init__(config.gpt_config)
    self.img_proj = MultiModalProjector(config)

  def _init_weights(self, module):
    #TODO
    pass

  def forward(self, idx, image_embeds=None, targets=None):
    B, T = idx.shape
    assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

    token_emb = self.transformer.wte(idx) # (B, T, C = n_embd)
    pos_emb = self.transformer.wpe(torch.arange(T, device=idx.device)) # (T, C = n_embd)
    x = token_emb + pos_emb # (B, T, C = n_embd)

    if image_embeds is not None:
      img_emb = self.img_proj(image_embeds).unsqueeze(1) # (B, 1, C = n_embd)
      x = torch.cat([img_emb, x], dim=1) # (B, T+1, C = n_embd)

    for block in self.transformer.h:
      x = block(x) # (B, T+1, C = n_embd)

    x = self.transformer.ln_f(x)
    logits = self.lm_head(x) # (B, T+1, vocab_size)
    loss = None

    if targets is not None:
      B, T, C = logits.shape
      if image_embeds is not None: 
        # Prepare targets by concatenating a dummy target for the image embedding
        targets = torch.cat([torch.full((B, 1), -100, dtype=torch.long, device=idx.device), targets], dim=1)
      loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T), ignore_index=-100)

    return logits, loss

  
  def generate(self, idx, image_embeds, max_new_tokens):
    #TODO 
    pass
  
  def configure_optimizers():
    #TODO
    pass

class VLM(nn.Module):
  def __init__(self, config):
    super().__init__()
    assert config.n_embd % config.n_head == 0
    self.vision_encoder = ViT(config)
    self.decoder = GPTWithImages(config)

  def forward(self, img_array, idx, targets=None):
    image_embeds = None
    if img_array is not None:
      image_embeds = self.vision_encoder(img_array)
      assert image_embeds.nelement() != 0 and image_embeds.shape[1] != 0

    if targets is not None:
      logits, loss = self.decoder(idx, image_embeds, targets)
      return logits, loss
    else:
      logits = self.decoder(idx, image_embeds)
      return logits
  
  def generate(self, img_array, idx, max_new_tokens):
    image_embeds = self.vision_encoder(img_array)
    
    assert image_embeds.nelement() != 0 and image_embeds.shape[1] != 0
    
    generated_tokens = self.decoder.generate(idx, image_embeds, max_new_tokens)
    return generated_tokens
  
  def load_gpt_checkpoint(self, checkpoint_file_name, device='cpu', verbose=True, freeze=True):
    """
    Load a GPT checkpoint into the decoder's transformer and lm_head.
    Only loads weights for transformer and lm_head, ignoring img_proj weights.
    """
    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache', 'checkpoints')
    model_file_path = os.path.join(checkpoint_dir, checkpoint_file_name)
    if not os.path.exists(model_file_path):
      raise FileNotFoundError(f"Checkpoint not found at")
    
    if verbose: print(f"Loading GPT checkpoint from {checkpoint_file_name}")
    checkpoint = torch.load(model_file_path, map_location=device)
    
    # Extract state dict and handle compiled model keys
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    
    # Remove '_orig_mod.' prefix from compiled models
    cleaned_state_dict = {}
    for key, value in state_dict.items():
      if key.startswith('_orig_mod.'):
        cleaned_key = key.replace('_orig_mod.', '')
        cleaned_state_dict[cleaned_key] = value
      else:
        cleaned_state_dict[key] = value
    
    missing_keys, unexpected_keys = self.decoder.load_state_dict(cleaned_state_dict, strict=False)
    
    if verbose:
      if missing_keys:
        missing_keys = [k for k in missing_keys if 'img_proj' not in k] # MultiModalProjector expected to be missing
        if len(missing_keys) > 0:
          print(f"Missing keys in decoder: {len(missing_keys)}")
          print(f"  Missing keys: {missing_keys}")
      if unexpected_keys and len(unexpected_keys) > 0:
        print(f"Unexpected keys: {len(unexpected_keys)}")
        print(f"  Unexpected: {unexpected_keys}")
      print("Checkpoint loaded!")
    
    if freeze:
      self.freeze_decoder()
      if verbose: print("Decoder weights frozen (no gradients/updates)\n-----")
    
    return self
  
  def freeze_decoder(self, freeze=True):
    """Freeze the decoder's transformer and lm_head weights (no gradients/updates)."""
    for param in self.decoder.transformer.parameters():
      param.requires_grad = not freeze
    for param in self.decoder.lm_head.parameters():
      param.requires_grad = not freeze
  
  def get_trainable_params(self):
    """Get count of trainable vs total parameters."""
    trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
    total = sum(p.numel() for p in self.parameters())
    return trainable, total

  def configure_optimizers(self, weight_decay, learning_rate, device, verbose=False):
    param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
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
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and 'cuda' in device
    if verbose: print(f"using fused AdamW: {use_fused}\n-----")
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused) #betas, eps match GPT3 paper
    return optimizer


# ------------------------------
# Misc Utils
# ------------------------------

# image to tensor
itot = transforms.Compose([
  transforms.Resize((VLMConfig().img_size, VLMConfig().img_size)),
  transforms.ToTensor()
])
# tensor to image
ttoi = transforms.ToPILImage()

def load_image(image_name):
  filename = f"tmp_{image_name}"
  image_url = f"https://kodali.io/vla/{image_name}"
  r = requests.get(image_url)
  with open(filename, "wb") as f: 
    f.write(r.content)
  return Image.open(filename).convert("RGB")

# ------------------------------
# Sampling code
# ------------------------------

# set sampling parameters
num_return_sequences = 1
max_length = 32
default_prefix = "This is an image of a"

def sample_sequences(num_return_sequences, max_length, device, model, img_array = None, prefix = default_prefix):
  # load model
  model.eval() # set model to evaluation mode
  model.to(device)

  # set up prefix tokens for sampling
  enc = tiktoken.get_encoding("gpt2")
  tokens = enc.encode(prefix)
  tokens = torch.tensor(tokens, dtype=torch.long)
  tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
  x = tokens.to(device) # (B = num_return_sequences, prefix_length)
  sample_rng = torch.Generator(device=device)
  sample_rng.manual_seed(42)
  # sample
  while(x.size(1) < max_length):
    with torch.no_grad():
      logits, loss = model(img_array,x) # (B, T, vocab_size)
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

# ------------------------------
# Create Model & Load Pretrained Decoder
# ------------------------------

device = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device\n-----")

# Set seed across all device types
torch.manual_seed(42)
if "cuda" in device:
    torch.cuda.manual_seed_all(42)
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    torch.mps.manual_seed(42)

model = VLM(VLMConfig())
model.to(device)

model.load_gpt_checkpoint("third_attempt.pt", freeze=True, verbose=True)

# ------------------------------
# Set up training batches
# ------------------------------

T = 1024 # sequence length
B = 32 # start with small batch size

train_loader = DataLoaderVLM(B=B, T=T, split='train', verbose=True)
val_loader = DataLoaderVLM(B=1, T=T, split='val', verbose=False)

encoder = tiktoken.get_encoding("gpt2")

batch = train_loader.next_batch()
images, input_tokens, target_tokens = batch
# print(images.shape)
# print(input_tokens.shape)
# print(target_tokens.shape)

lr = 1e-3
weight_decay = 0.1
max_steps = 1000
optimizer = model.configure_optimizers(weight_decay, lr, device, verbose=True)

def train(steps=max_steps):
  model.train()
  for step in range(steps):
    images, input_tokens, target_tokens = train_loader.next_batch()
    images = images.to(device)
    input_tokens = input_tokens.to(device)
    target_tokens = target_tokens.to(device)
    
    logits, loss = model(images, input_tokens, target_tokens)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Step {step}: Loss {loss.item():.4f}")

def generate(image, input_tokens, max_new_tokens):
  model.eval()
  image = image.to(device)
  input_tokens = input_tokens.to(device)
  # sample tokens
  while(input_tokens.size(1) < max_new_tokens):
    with torch.no_grad():
      logits, loss = model(image, input_tokens)
      logits = logits[:, -1, :] # (B, vocab_size)
      probs = F.softmax(logits, dim = -1) # (B, vocab_size)
      ix = torch.multinomial(probs, 1) # (B, 1)
      input_tokens = torch.cat((input_tokens, ix), dim = 1) # (B, T + 1)
  return encoder.decode(input_tokens[0, 1:100].tolist())

def evaluate():
  model.eval()
  images, input_tokens, target_tokens = val_loader.next_batch()
  images = images.to(device)
  input_tokens = torch.tensor(encoder.encode("This is an image of"), dtype=torch.long).to(device).unsqueeze(0)
  target_tokens = target_tokens.to(device)
  
  expected_output = encoder.decode(target_tokens[0, :100].tolist())
  generated_output = generate(images, input_tokens, max_new_tokens=100)
  print(f"-------\nExpected Output: This{expected_output}")
  print(f"-------\nGenerated Output: {generated_output}")
  print(f"-------\n")
  
train(1)
evaluate()
import code; code.interact(local=locals())