# src/visualize.py
# Generates plots and sample outputs for the report.
# Run AFTER training:  python src/visualize.py

import os
import sys
import math
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt

from config import Config
from src.scheduler import get_lr
from src.generate import load_model, generate_text, encode, decode


SAMPLES_DIR = "outputs/samples"
os.makedirs(SAMPLES_DIR, exist_ok=True)


# ----------------------------------------------------------------------
# 1. Learning-rate schedule curve
# ----------------------------------------------------------------------
def plot_lr_schedule(cfg):
    """Plot LR over training iterations (warmup + cosine decay)."""
    iters = list(range(cfg.max_iters))
    lrs = [get_lr(it, cfg) for it in iters]

    plt.figure(figsize=(8, 4))
    plt.plot(iters, lrs, linewidth=2, color="#1f77b4")
    plt.axvline(cfg.warmup_iters, color="gray", linestyle="--", alpha=0.6,
                label=f"warmup ends ({cfg.warmup_iters})")
    plt.xlabel("Iteration")
    plt.ylabel("Learning rate")
    plt.title("LR Schedule: Linear Warmup + Cosine Decay")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{SAMPLES_DIR}/lr_schedule.png", dpi=150)
    plt.close()
    print(f"  saved {SAMPLES_DIR}/lr_schedule.png")


# ----------------------------------------------------------------------
# 2. Training/val loss curve
# ----------------------------------------------------------------------
def plot_loss_curve():
    """
    Plot training + validation loss from the training log saved during training.
    """
    log_path = "outputs/checkpoints/training_log.json"
    if not os.path.exists(log_path):
        print(f"  [skip] no training log at {log_path}")
        print("         Retrain with the updated train.py to enable this plot.")
        return

    with open(log_path) as f:
        log = json.load(f)

    # Match keys produced by the updated train.py
    train_iters  = log.get("train_iters", [])
    train_losses = log.get("train_loss", [])
    val_iters    = log.get("val_iters", [])
    val_losses   = log.get("val_loss", [])
    lrs          = log.get("train_lr", [])

    if not train_iters:
        print("  [skip] training log is empty")
        return

    # ----- Plot 1: loss curves (train per-step, val per-eval) -----
    plt.figure(figsize=(8, 4))
    plt.plot(train_iters, train_losses, alpha=0.35, color="#888",
             label="train (per step)")
    if val_iters:
        plt.plot(val_iters, val_losses, linewidth=2, marker="o",
                 color="#d62728", label="val (per eval)")
    plt.xlabel("Iteration")
    plt.ylabel("Cross-entropy loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{SAMPLES_DIR}/loss_curve.png", dpi=150)
    plt.close()
    print(f"  saved {SAMPLES_DIR}/loss_curve.png")

    # ----- Plot 2: LR actually used (real schedule, not recomputed) -----
    if lrs:
        plt.figure(figsize=(8, 4))
        plt.plot(train_iters, lrs, linewidth=2, color="#1f77b4")
        plt.xlabel("Iteration")
        plt.ylabel("Learning rate")
        plt.title("Learning Rate Used During Training")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{SAMPLES_DIR}/lr_used.png", dpi=150)
        plt.close()
        print(f"  saved {SAMPLES_DIR}/lr_used.png")

# ----------------------------------------------------------------------
# 3. Generate and save samples under different sampling strategies
# ----------------------------------------------------------------------
def save_generation_samples(ckpt_path="outputs/checkpoints/best.pt"):
    """Produce text with different samplers and save each to a file."""
    configs = [
        ("greedy",              dict(greedy=True, max_new_tokens=400)),
        ("temperature_0.5",     dict(temperature=0.5, max_new_tokens=400)),
        ("temperature_0.8_k40", dict(temperature=0.8, top_k=40, max_new_tokens=400)),
        ("temperature_1.0",     dict(temperature=1.0, max_new_tokens=400)),
        ("temperature_1.5",     dict(temperature=1.5, max_new_tokens=400)),
    ]

    all_outputs = {}
    for name, kwargs in configs:
        print(f"  generating: {name} ...")
        text = generate_text(
            ckpt_path=ckpt_path,
            prompt="ROMEO: ",
            **kwargs,
        )
        all_outputs[name] = text
        with open(f"{SAMPLES_DIR}/sample_{name}.txt", "w", encoding="utf-8") as f:
            f.write(text)

    # Combined comparison file
    with open(f"{SAMPLES_DIR}/all_samples.txt", "w", encoding="utf-8") as f:
        for name, text in all_outputs.items():
            f.write("=" * 70 + "\n")
            f.write(f"Sampling: {name}\n")
            f.write("=" * 70 + "\n")
            f.write(text + "\n\n")
    print(f"  saved {SAMPLES_DIR}/sample_*.txt and all_samples.txt")


# ----------------------------------------------------------------------
# 4. Attention heatmap (optional but very impressive in a report)
# ----------------------------------------------------------------------
def plot_attention_heatmap(ckpt_path="outputs/checkpoints/best.pt", prompt="ROMEO: "):
    """
    Extract attention weights from the first layer's first head
    for a short prompt and visualize the causal mask in action.
    """
    device = "cpu"
    model, cfg, stoi, itos = load_model(ckpt_path, device)

    # Hook: capture the attention weights from the first block's attn
    captured = {}
    def hook(module, inp, out):
        captured["qkv_input"] = inp[0].detach()

    # Easier: monkey-patch CausalSelfAttention.forward to also store 'att'
    first_attn = model.blocks[0].attn
    original_forward = first_attn.forward

    def patched_forward(x):
        B, T, C = x.size()
        qkv = first_attn.c_attn(x)
        q, k, v = qkv.split(first_attn.n_embd, dim=2)
        head_dim = C // first_attn.n_head
        k = k.view(B, T, first_attn.n_head, head_dim).transpose(1, 2)
        q = q.view(B, T, first_attn.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, first_attn.n_head, head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
        att = att.masked_fill(first_attn.bias[:, :, :T, :T] == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)

        captured["att"] = att.detach()  # (B, n_head, T, T)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = first_attn.resid_dropout(first_attn.c_proj(y))
        return y

    first_attn.forward = patched_forward

    # Run forward pass on the prompt
    ids = encode(prompt, stoi)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        model(idx)

    # Restore
    first_attn.forward = original_forward

    att = captured["att"][0]  # (n_head, T, T)
    n_head = att.shape[0]

    fig, axes = plt.subplots(1, n_head, figsize=(3 * n_head, 3.2))
    if n_head == 1:
        axes = [axes]
    for h in range(n_head):
        ax = axes[h]
        im = ax.imshow(att[h].cpu().numpy(), cmap="viridis", aspect="auto")
        ax.set_title(f"Head {h}")
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(f'Causal attention — prompt: "{prompt}"', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{SAMPLES_DIR}/attention_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {SAMPLES_DIR}/attention_heatmap.png")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    cfg = Config()
    print("Generating report visualizations...")

    print("[1/4] LR schedule ...")
    plot_lr_schedule(cfg)

    print("[2/4] Loss curve ...")
    plot_loss_curve()

    print("[3/4] Sampling comparison ...")
    save_generation_samples()

    print("[4/4] Attention heatmap ...")
    plot_attention_heatmap()

    print("\nAll done. Check outputs/samples/")