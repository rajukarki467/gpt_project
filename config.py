import torch


class Config:
    def __init__(self):
        # ----- Data -----
        self.data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        self.data_path = "data/input.txt"
        self.block_size = 128

        # ----- Model Architecture -----
        self.n_layer = 4
        self.n_head = 4
        self.n_embd = 128
        self.dropout = 0.1
        self.bias = True

        # ----- Training -----
        self.batch_size = 64
        self.grad_accum = 1
        self.max_iters = 5000
        self.eval_interval = 250
        self.eval_iters = 100
        self.learning_rate = 3e-4
        self.weight_decay = 0.1
        self.beta1 = 0.9
        self.beta2 = 0.95
        self.grad_clip = 1.0

        # ----- Scheduler -----
        self.warmup_iters = 200
        self.lr_decay_iters = 5000
        self.min_lr = 3e-5

        # ----- System -----
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = (
            "bfloat16"
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else "float16"
        )
        self.compile = False
        self.seed = 1337