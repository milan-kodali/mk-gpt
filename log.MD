# Training Runs

##third_attempt.pt

class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

step: 21000
val loss: 3.2596