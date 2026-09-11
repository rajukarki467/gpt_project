import torch

class Config:
    # ----- Data -----
    data_url      = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    data_path     = "data/input.txt"
    block_size    = 128            # Context length (max tokens per sample)

    # ----- Model Architecture -----
    n_layer       = 4              # Number of Transformer blocks
    n_head        = 4              # Number of attention heads
    n_embd        = 128            # Embedding dimension (must be divisible by n_head)
    dropout       = 0.1
    bias          = True           # Whether Linear layers use bias

    # ----- Training -----
    batch_size    = 64
    grad_accum    = 1              # Gradient accumulation steps
    max_iters     = 5000           # Total optimization steps
    eval_interval = 250
    eval_iters    = 100
    learning_rate = 3e-4
    weight_decay  = 0.1
    beta1         = 0.9
    beta2         = 0.95
    grad_clip     = 1.0

    # ----- Scheduler -----
    warmup_iters  = 200            # Linear warmup steps
    lr_decay_iters = 5000          # Should match max_iters
    min_lr        = 3e-5           # Cosine floor

    # ----- System -----
    device        = "cuda" if torch.cuda.is_available() else "cpu"
    dtype         = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
    compile       = False          # torch.compile (PyTorch >= 2.0)
    seed          = 1337