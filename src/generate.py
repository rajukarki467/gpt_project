# Autoregressive Generation
# This supports greedy, temperature, and top-k sampling.



import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from config import Config
from src.model import GPT


def load_model(ckpt_path, device):
    """Load a checkpoint and rebuild the model + tokenizer."""
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_dict = ckpt["cfg"]

    # Rebuild a Config instance from the saved dict
    class _C: pass
    cfg = _C()
    for k, v in cfg_dict.items():
        setattr(cfg, k, v)
    cfg.device = device

    model = GPT(cfg, vocab_size=len(ckpt["stoi"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Return (model, cfg, stoi, itos)
    return model, cfg, ckpt["stoi"], ckpt["itos"]


def encode(s, stoi): return [stoi[c] for c in s if c in stoi]
def decode(ids, itos): return "".join(itos[i] for i in ids)


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

    # Encode prompt -> tensor
    ids = encode(prompt, stoi)
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    # Greedy: use temperature very low to make argmax dominant
    if greedy:
        temperature = 1e-6

    with torch.no_grad():
        out = model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

    # Decode and return only the newly generated portion
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