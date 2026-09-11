# Training Loop with AMP
# This is where mixed-precision and scheduler stepping come together.

import os
import sys
import math
import time
import json

# Allow imports from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from src.dataset import build_datasets
from src.model import GPT
from src.optimizer import configure_optimizer
from src.scheduler import get_lr
from src.loss import lm_loss


def evaluate(model, loader, cfg):
    """Average loss over `eval_iters` batches (no grad)."""
    model.eval()
    losses = torch.zeros(cfg.eval_iters)
    it = iter(loader)
    with torch.no_grad():
        for k in range(cfg.eval_iters):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(loader)
                x, y = next(it)
            x, y = x.to(cfg.device), y.to(cfg.device)
            with torch.autocast(device_type=cfg.device, dtype=getattr(torch, cfg.dtype)):
                _, loss = model(x, y)
            losses[k] = loss.item()
    model.train()
    return losses.mean().item()


def train():
    cfg = Config()
    torch.manual_seed(cfg.seed)
    os.makedirs("outputs/checkpoints", exist_ok=True)

    # ---------- Data ----------
    train_ds, val_ds, tokenizer = build_datasets(cfg)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    print(f"vocab size: {tokenizer.vocab_size}")

    # ---------- Model ----------
    model = GPT(cfg, vocab_size=tokenizer.vocab_size).to(cfg.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e6:.2f}M")

    # ---------- Optimizer ----------
    optimizer = configure_optimizer(model, cfg)

    # ---------- AMP Setup ----------
    # bf16 doesn't need a GradScaler; fp16 does.
    use_scaler = (cfg.dtype == "float16")
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    # ---------- Training Log (for plotting later) ----------
    # We log two streams:
    #   - every step: (iter, train loss, lr)
    #   - every eval interval: (iter, val loss)
    # Two separate lists because eval happens less often than training steps.
    training_log = {
        "train_iters":   [],
        "train_loss":    [],
        "train_lr":      [],
        "val_iters":     [],
        "val_loss":      [],
        "best_val_loss": None,
    }

    # ---------- Training ----------
    best_val = float("inf")
    iter_num = 0
    t0 = time.time()

    model.train()
    train_iter = iter(train_loader)

    pbar = tqdm(range(cfg.max_iters), desc="training")
    for iter_num in pbar:
        # 1) Update LR according to our manual schedule
        lr = get_lr(iter_num, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # 2) Fetch a batch (recycle the loader if exhausted)
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        x, y = x.to(cfg.device), y.to(cfg.device)

        # 3) Forward + loss under autocast (mixed precision)
        with torch.autocast(device_type=cfg.device, dtype=getattr(torch, cfg.dtype)):
            logits, _ = model(x, y)
            loss = lm_loss(logits, y)
            loss = loss / cfg.grad_accum

        # 4) Backward + gradient clipping + optimizer step
        scaler.scale(loss).backward()

        if (iter_num + 1) % cfg.grad_accum == 0:
            # Unscale gradients before clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # Log the training step (unscaled loss for readability)
        training_log["train_iters"].append(iter_num + 1)
        training_log["train_loss"].append(loss.item() * cfg.grad_accum)
        training_log["train_lr"].append(lr)

        pbar.set_postfix(loss=f"{loss.item()*cfg.grad_accum:.4f}", lr=f"{lr:.2e}")

        # 5) Periodic evaluation
        if (iter_num + 1) % cfg.eval_interval == 0 or iter_num == cfg.max_iters - 1:
            val_loss = evaluate(model, val_loader, cfg)
            print(f"\n[iter {iter_num+1}] train loss {loss.item()*cfg.grad_accum:.4f} | "
                  f"val loss {val_loss:.4f} | lr {lr:.2e} | "
                  f"elapsed {(time.time()-t0)/60:.2f} min")

            # Log the eval point
            training_log["val_iters"].append(iter_num + 1)
            training_log["val_loss"].append(val_loss)

            # Save checkpoint if best
            if val_loss < best_val:
                best_val = val_loss
                training_log["best_val_loss"] = val_loss

                # Build a full config dict combining class-level and instance-level attrs.
                # Handles both styles of defining Config (class attrs vs __init__ attrs).
                cfg_dict = {}
                for k in dir(cfg):
                    if k.startswith("_"):
                        continue
                    try:
                        v = getattr(cfg, k)
                    except AttributeError:
                        continue
                    if callable(v):
                        continue
                    # Only keep primitives / plain values (skip modules, tensors, etc.)
                    if isinstance(v, (int, float, str, bool, type(None))):
                        cfg_dict[k] = v

                ckpt = {
                    "model": model.state_dict(),
                    "cfg": cfg_dict,
                    "iter": iter_num,
                    "val_loss": val_loss,
                    "stoi": tokenizer.stoi,
                    "itos": tokenizer.itos,
                }
                torch.save(ckpt, "outputs/checkpoints/best.pt")
                print(f"  → saved checkpoint (val_loss={val_loss:.4f}, "
                      f"cfg keys={len(cfg_dict)})")

            # Save training log after every eval (so a crash doesn't lose data)
            with open("outputs/checkpoints/training_log.json", "w") as f:
                json.dump(training_log, f, indent=2)

    # Final save after training completes
    with open("outputs/checkpoints/training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)

    print("Training complete. Best val loss:", best_val)


if __name__ == "__main__":
    train()