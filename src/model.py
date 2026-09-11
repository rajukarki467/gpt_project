'''
# The Transformer Model
- implement everything from raw PyTorch primitives (nn.Linear, nn.LayerNorm, torch.matmul, etc.).
'''
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


# ----------------------------------------------------------------------
# 1. Causal Self-Attention (multi-head)
# ----------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    """
    Multi-head masked self-attention implemented from scratch.
    Uses a single Linear layer to compute Q, K, V for all heads at once,
    then reshapes into (B, n_head, T, head_dim).
    """
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"

        # Key, Query, Value projections (fused into one matmul for efficiency)
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        # Output projection
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)

        self.attn_dropout  = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        self.n_head    = cfg.n_head
        self.n_embd    = cfg.n_embd
        self.dropout   = cfg.dropout

        # Causal mask: lower-triangular matrix of ones, shape (1, 1, T, T)
        # We register it as a buffer so it moves with the model to GPU but is not trained.
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size))
                 .view(1, 1, cfg.block_size, cfg.block_size),
            persistent=False,
        )

    def forward(self, x):
        B, T, C = x.size()   # batch, sequence length, embedding dim

        # 1) Compute Q, K, V in one shot then split into 3 chunks
        qkv = self.c_attn(x)                 # (B, T, 3C)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # 2) Reshape into (B, n_head, T, head_dim)
        head_dim = C // self.n_head
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        # 3) Scaled dot-product attention scores: (B, n_head, T, T)
        #    att = Q K^T / sqrt(d_k)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))

        # 4) Apply causal mask: positions with 0 (above diagonal) are set to -inf
        #    so softmax gives them zero probability.
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

        # 5) Softmax over the last dimension (keys) -> probabilities
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # 6) Weighted sum of values: (B, n_head, T, head_dim)
        y = att @ v

        # 7) Concatenate heads back into (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 8) Final output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


# ----------------------------------------------------------------------
# 2. Feed-Forward Network (MLP)
# ----------------------------------------------------------------------
class MLP(nn.Module):
    """Standard Transformer FFN: Linear -> GELU -> Linear -> Dropout."""
    def __init__(self, cfg):
        super().__init__()
        self.c_fc    = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


# ----------------------------------------------------------------------
# 3. Transformer Block (Pre-LN)
# ----------------------------------------------------------------------
class Block(nn.Module):
    """
    A single Transformer decoder block.
    Uses Pre-LN: LayerNorm -> Attention -> residual -> LayerNorm -> MLP -> residual.
    Pre-LN is more stable than Post-LN, especially for deep networks.
    """
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp  = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))    # residual around attention
        x = x + self.mlp(self.ln_2(x))     # residual around MLP
        return x


# ----------------------------------------------------------------------
# 4. Full GPT Model
# ----------------------------------------------------------------------
class GPT(nn.Module):
    def __init__(self, cfg, vocab_size):
        super().__init__()
        self.cfg = cfg

        # ---- Embeddings ----
        # Token embedding + learned positional embedding (both trainable).
        self.wte = nn.Embedding(vocab_size, cfg.n_embd)   # token embedding
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)  # positional embedding

        self.drop = nn.Dropout(cfg.dropout)

        # ---- Stack of Transformer blocks ----
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])

        # ---- Final LayerNorm & LM head ----
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, vocab_size, bias=False)

        # Weight tying: share weights between token embedding and LM head.
        # This is standard in GPT-2 and reduces parameters.
        self.lm_head.weight = self.wte.weight

        # ---- Parameter init ----
        self.apply(self._init_weights)
        # Scale residual projections by 1/sqrt(2 * n_layer) for stability
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        idx:     (B, T) integer token ids
        targets: (B, T) integer token ids (or None at inference)
        Returns: logits (B, T, vocab_size), loss (scalar or None)
        """
        B, T = idx.size()
        assert T <= self.cfg.block_size, f"Sequence length {T} > block_size {self.cfg.block_size}"

        # 1) Token + positional embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))   # (B, T, C)

        # 2) Pass through each Transformer block
        for block in self.blocks:
            x = block(x)

        # 3) Final LayerNorm + project to vocab
        x = self.ln_f(x)
        logits = self.lm_head(x)                        # (B, T, vocab_size)

        # 4) Compute cross-entropy if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        return logits, loss

    # ------------------------------------------------------------------
    # Autoregressive Generation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Autoregressive sampling.
          idx:          (B, T) prompt token ids
          max_new_tokens: number of tokens to generate
          temperature:  >0, higher = more random
          top_k:        if set, only sample from the top-k most likely tokens
        """
        for _ in range(max_new_tokens):
            # Crop to context window
            idx_cond = idx[:, -self.cfg.block_size:]

            # Forward pass -> logits for next token
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature   # (B, vocab_size)

            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            # Softmax -> probabilities -> sample
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx