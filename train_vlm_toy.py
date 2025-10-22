import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms

from PIL import Image
import tiktoken
import argparse
import os
import requests

from vlm import VLM, VLMConfig

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
# Sampling code
# ------------------------------

# set sampling parameters
num_return_sequences = 4
max_length = 32
default_prefix = "Hello, I'm a language model,"

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
# Toy Dataset
# ------------------------------

checker = itot(load_image("checker.png"))
red_blue = itot(load_image("red-blue.png"))
green_circle = itot(load_image("green-circle.png"))
green_square = itot(load_image("green-square.png"))
red_circle = itot(load_image("red-circle.png"))
red_square = itot(load_image("red-square.png"))
blue_circle = itot(load_image("blue-circle.png"))
blue_square = itot(load_image("blue-square.png"))
yellow_circle = itot(load_image("yellow-circle.png"))
yellow_square = itot(load_image("yellow-square.png"))
pink_circle = itot(load_image("pink-circle.png"))
pink_square = itot(load_image("pink-square.png"))

dataset_train = [
  (checker, "black and white checkerboard"),
  (red_blue, "red and blue checkerboard"),
  (green_circle, "green circle"),
  (green_square, "green square"),
  (red_circle, "red circle"),
  (red_square, "red square"),
  (blue_circle, "blue circle"),
  (yellow_circle, "yellow circle"),
  (yellow_square, "yellow square"),
  (pink_square, "pink square"),
]

dataset_val = [
  (pink_circle, "pink circle"),
  (blue_square, "blue square"),
]

# ------------------------------
# Toy Dataset Sampling Code
# ------------------------------

def sample_dataset(dataset):
  for image, label in dataset:
    print(f"> {label}")
    prefix = "This is an image of a"
    num_return_sequences = 1
    image = image.unsqueeze(0).to(device)
    sample_sequences(num_return_sequences, max_length, device, model, img_array=image, prefix=prefix)

def sample_toy_datasets():
  print("--------------------------------\nSampling toy train dataset\n--------------------------------")
  sample_dataset(dataset_train)
  print("--------------------------------\nSampling toy val dataset\n--------------------------------")
  sample_dataset(dataset_val)

# ------------------------------
# Toy Training Loop
# ------------------------------

def train(steps=1000):

  # Only optimize trainable parameters (ViT + projector, NOT frozen decoder)
  trainable_params = [p for p in model.parameters() if p.requires_grad]
  optimizer = torch.optim.AdamW(trainable_params, lr=1e-5)

  enc = tiktoken.get_encoding("gpt2")
  model.train()
  model.to(device)  # Move once, not in loop

  for step in range(steps):
    image, label = dataset_train[step % len(dataset_train)]
    
    # Add batch dimension
    image = image.unsqueeze(0).to(device)  # (1, 3, 24, 24)
    
    # Prepare input-target pair
    full_text = "This is an image of a " + label
    full_tokens = enc.encode(full_text)
    
    # Autoregressive: input[t] predicts target[t]
    input_tokens = torch.tensor(full_tokens[:-1], dtype=torch.long).unsqueeze(0).to(device)
    target_tokens = torch.tensor(full_tokens[1:], dtype=torch.long).unsqueeze(0).to(device)
    
    # Forward pass
    logits, loss = model(image, input_tokens, target_tokens)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if step % 10 == 0:
      print(f"Step {step}: Loss {loss.item():.4f}")


# ------------------------------
# Script
# ------------------------------

print("--------------------------------\nSampling before training\n--------------------------------")
sample_toy_datasets()

print("--------------------------------\nTraining\n--------------------------------")

train()

print("--------------------------------\nSampling after training\n--------------------------------")
sample_toy_datasets()

import code; code.interact(local=locals())