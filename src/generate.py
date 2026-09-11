# Autoregressive Generation
# Supports greedy, temperature, and top-k sampling.

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from config import Config
from src.model import GPT


def load_model(ckpt_path, device):
    """Load a checkpoint and rebuild the model + tokenizer.

    Tolerant of old checkpoints whose saved cfg dict may be empty or incomplete —
    in that case we fall back to values defined in config.py.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_dict = ckpt.get("cfg", {}) or {}

    # Start from a fresh Config (guaranteed to have all needed attributes),
    # then let any saved values overwrite the defaults.
    cfg = Config()
    for k, v in cfg_dict.items():
        setattr(cfg, k, v)

    # Attach runtime device (not stored in the checkpoint)
    cfg.device = device

    # Sanity check — fail loudly if something is missing
    required = ("n_embd", "n_layer", "n_head", "block_size", "dropout", "bias", "device")
    missing = [k for k in required if not hasattr(cfg, k)]
    if missing:
        raise RuntimeError(
            f"Config is missing required attributes: {missing}. "
            f"Checkpoint cfg keys: {list(cfg_dict.keys())}"
        )

    model = GPT(cfg, vocab_size=len(ckpt["stoi"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    return model, cfg, ckpt["stoi"], ckpt["itos"]


def encode(s, stoi):
    return [stoi[c] for c in s if c in stoi]


def decode(ids, itos):
    return "".join(itos[i] for i in ids)


def generate_text(
    ckpt_path="outputs/checkpoints/best.pt",
    prompt="ROMEO: ",
    max_new_tokens=300,
    temperature=1.0,
    top_k=None,          # e.g. 40 for top-k sampling
    greedy=False,        # if True, ignore temperature/top_k
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg, stoi, itos = load_model(ckpt_path, device)

    # Encode prompt -> tensor (B=1, T=len(prompt))
    ids = encode(prompt, stoi)
    if len(ids) == 0:
        # Fallback: if prompt had no known chars, start from a newline
        ids = encode("\n", stoi)
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    # Greedy: essentially argmax via near-zero temperature
    if greedy:
        temperature = 1e-6

    with torch.no_grad():
        out = model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

    # Decode the full sequence (prompt + generated)
    text = decode(out[0].tolist(), itos)
    return text


if __name__ == "__main__":
    print("=" * 60)
    print("Greedy decoding:")
    print("=" * 60)
    print(generate_text(prompt="ROMEO: ", max_new_tokens=200, greedy=True))

    print("\n" + "=" * 60)
    print("Temperature sampling (T=0.8, top_k=40):")
    print("=" * 60)
    print(generate_text(prompt="ROMEO: ", max_new_tokens=200, temperature=0.8, top_k=40))

    print("\n" + "=" * 60)
    print("High-temperature sampling (T=1.5):")
    print("=" * 60)
    print(generate_text(prompt="ROMEO: ", max_new_tokens=200, temperature=1.5))