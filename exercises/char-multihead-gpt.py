# Character-level language model with multi-head attention
# Combines Head and MultiHeadAttention into a single class
# ToDo: combine qkv and treat as batch

import torch
import torch.nn as nn
from torch.nn import functional as F
import os

# model hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel
block_size = 16 # maximum context length for predictions
n_embd = 128
n_head = 2
n_layer = 2
dropout = 0.2

# training hyperparameters
max_iters = 2000
eval_interval = 50
learning_rate = 1e-3
eval_iters = 16
use_manual_seed = True

# other parameters
model_file_name = f"v2ex1_{batch_size}_{block_size}_{n_embd}_{n_head}_{n_layer}_{dropout}.pt"
training_file_name = 'shakespeare.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ----------------

if use_manual_seed:
    torch.manual_seed(1337)

with open(f'../inputs/{training_file_name}', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)


# create a mapping from characters to integers
stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# train and test split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # 90% of the data for training, 10% for validation
train_data = data[:n]
val_data = data[n:]

print(f"Training on: {training_file_name}\n-----")
print(f"Vocab size: {vocab_size}\n-----")
print(f"Dataset size: {len(data)}\n-----")
print(f"Using {device} device\n-----")

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class MultiHeadAttention(nn.Module):
    """ multiple self-attention heads in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.num_heads = num_heads
        self.key = nn.Linear(n_embd, num_heads * head_size, bias=False)
        self.query = nn.Linear(n_embd, num_heads * head_size, bias=False)
        self.value = nn.Linear(n_embd, num_heads * head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) 

        self.dropout1 = nn.Dropout(dropout)

        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) #(B, T, num_heads * head_size)
        k = k.view(B, T, self.num_heads, -1).transpose(1, 2) #(B, T, num_heads, head_size) -> (B, num_heads, T, head_size)

        q = self.query(x) #(B, T, num_heads * head_size)
        q = q.view(B, T, self.num_heads, -1).transpose(1, 2) #(B, T, num_heads, head_size) -> (B, num_heads, T, head_size)

        # compute attention scores ("affinities")
        wei = k @ q.transpose(-2, -1) * C**-0.5 #(B,num_heads,T,head_size) @ (B,num_heads,head_size,T) -> (B,num_heads,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = wei.softmax(dim = -1) # (B, num_heads, T, T)
        wei = self.dropout1(wei) # (B, num_heads, T, T)

        v = self.value(x) #(B, T, num_heads * head_size)
        v = v.view(B, T, self.num_heads, -1).transpose(1, 2) #(B, T, num_heads, head_size) -> (B, num_heads, T, head_size)

        out = wei @ v  # (B, num_heads, T, T) @ (B, num_heads, T, head_size) -> (B, num_heads, T, head_size)
        out = out.transpose(1, 2).reshape(B, T, -1) # (B, num_heads, T, head_size) -> (B, T, num_heads * head_size)

        out = self.dropout2(self.proj(out)) 
        return out             

class FeedForward(nn.Module):
    """ Linear layer + non-linearity to add compute after multi-head attention layer """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), #expand onto higher dimensional space
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # equivalent to self.proj in MHA layer, project back down to model's embedding dimensionality 
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: Communication followed by computation, with residual connection (x +) """ 

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd) 
        self.ln1 = nn.LayerNorm(n_embd) # ToDo: Understand
        self.ln2 = nn.LayerNorm(n_embd) # ToDo: Understand

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# super simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head = n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    # Funky: logits shape is different based on whether targets are provided
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # idx and targets are both (B, T) tensor of integers
        token_emb = self.token_embedding_table(idx) # (B, T, C = n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C = n_embd)
        x = token_emb + pos_emb # (B, T, C = n_embd)
        x = self.blocks(x) # (B, T, C = n_embd)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        for _ in range (max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get predicitons
            logits, loss = self(idx_cond)
            # focus on last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax
            probs = F.softmax(logits, dim = -1) # (B, C)
            # sample
            idx_next = torch.multinomial(probs, num_samples = 1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim = 1) # (B, T+1)

        return idx 
        
model = BigramLanguageModel()
m = model.to(device)

if os.path.exists(model_file_name):
    print(f"Loading model weights from {model_file_name}\n-----")
    m.load_state_dict(torch.load(model_file_name, map_location=device))
else:
    print("No saved model found, starting from scratch\n-----")

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # every eval_interval, check the loss on the validation set
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        torch.save(m.state_dict(), model_file_name)  # save checkpoint

    # sample a batch of data
    xb, yb = get_batch('train')
    
    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

#print(model.position_embedding_table.weight)
