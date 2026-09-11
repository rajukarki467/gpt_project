# MiniGPT: A Decoder-Only Transformer From Scratch

I built this to properly understand how GPT-style language models actually work under the hood. Instead of importing `nn.TransformerDecoder` and calling it a day, everything here — causal attention, the Transformer block, weight tying, the training loop — is written from PyTorch primitives. It's a small model (~0.8M params) trained on Tiny Shakespeare, and it runs in a few minutes on a CPU or seconds on a GPU.

The goal wasn't to produce state-of-the-art text. It was to make every line of the forward pass explainable.

---

## What's inside

- **Causal multi-head self-attention** implemented by hand, including the lower-triangular mask
- **Pre-LayerNorm Transformer blocks** with residual connections and a GELU feed-forward network
- **Learned token + positional embeddings**, with weight tying between the embedding and LM head
- **Mixed-precision training** using `torch.autocast` and `GradScaler`
- **Linear warmup + cosine decay** learning-rate schedule
- **AdamW** with weight decay applied *only* to 2D parameters (biases and LayerNorm gains are excluded)
- **Three sampling modes** for generation: greedy, temperature, and top-k

---

## Layout

```
gpt_project/
├── config.py              # every hyperparameter lives here
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/                  # Tiny Shakespeare lands here on first run
│
├── outputs/               # checkpoints and sample generations
│   └── checkpoints/
│
└── src/
    ├── dataset.py         # char-level tokenizer + LM dataset
    ├── model.py           # attention, block, GPT
    ├── loss.py            # cross-entropy wrapper
    ├── optimizer.py       # AdamW param grouping
    ├── scheduler.py       # warmup + cosine
    ├── train.py           # training loop
    └── generate.py        # sampling
```

---

## Getting it running

Grab the code and set up a virtual environment:

```bash
git clone <your-repo-url>
cd gpt_project
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Then train:

```bash
python src/train.py
```

The Tiny Shakespeare corpus (~1.1 MB) downloads itself on the first run. Training defaults to 5,000 iterations with a batch size of 64 and a context length of 128 tokens. On a mid-range GPU it takes about 3–5 minutes. On CPU, expect closer to 30–45 minutes.

You'll see a progress bar tracking loss and learning rate. If things are healthy, the loss starts around 4.17 (that's `ln(65)`, the entropy of a uniform distribution over the character vocabulary) and drops below 1.6 by the end.

To sample from the trained model:

```bash
python src/generate.py
```

That runs three generations in sequence — greedy, temperature-with-top-k, and high-temperature — so you can compare how the different sampling strategies behave.

---

## Architecture notes

The defaults are deliberately tiny so the whole thing is debuggable:

| | |
|---|---|
| Layers | 4 |
| Attention heads | 4 |
| Embedding dim | 128 |
| Context length | 128 |
| Dropout | 0.1 |
| Parameters | ~0.8 M |

If you want a slightly more capable model, bump `n_layer` to 6 and `n_embd` to 256 in `config.py`. You'll get noticeably better output at the cost of speed.

### Attention

The scaled dot-product attention is written out by hand. Nothing is hidden behind a library call:

```python
att = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
att = att.masked_fill(causal_mask == 0, float("-inf"))
att = F.softmax(att, dim=-1)
y   = att @ v
```

The causal mask is a lower-triangular matrix registered as a module buffer, so it travels with the model to GPU automatically and isn't treated as a trainable parameter. Above the diagonal, scores are set to negative infinity before the softmax, which forces those positions to zero probability. That's what makes the model autoregressive.

Q, K, and V are computed in one fused `nn.Linear` projection and then split — this is what GPT-2 does, and it's meaningfully faster than three separate projections.

### Block

Each Transformer block is Pre-LayerNorm:

```
x = x + Attention(LayerNorm(x))
x = x + MLP(LayerNorm(x))
```

Pre-LN trains more stably than the original Post-LN formulation, especially as networks get deeper. It's the standard choice now for anything beyond toy depth.

### Weight tying

The token embedding matrix and the LM head share weights. It's a small thing, but it saves a meaningful number of parameters at scale and tends to help generalization. One line in `GPT.__init__`:

```python
self.lm_head.weight = self.wte.weight
```

---

## Training details

A few decisions worth calling out, since they're the ones that actually mattered in practice:

**Learning rate schedule.** Linear warmup for the first 200 steps, then cosine decay down to a floor of `3e-5`. The warmup exists because early gradients on a randomly-initialized Transformer are large and noisy — jumping straight to peak LR tends to destabilize the run.

**Weight decay grouping.** Only tensors with `dim() >= 2` get weight decay. Biases, LayerNorm gains, and embedding tables are excluded. This matches what GPT-3 and most modern recipes do; decaying a LayerNorm gain toward zero is not something you actually want.

**Mixed precision.** `bfloat16` when the GPU supports it (no scaler needed), otherwise `float16` with a `GradScaler`. The autocast context wraps only the forward pass and loss — the backward pass runs in the scaled dtype automatically.

**Gradient clipping.** Global norm clipped to 1.0 every step. It's a cheap safety net against the occasional loss spike.

---

## What the output looks like

After 5,000 iterations on Tiny Shakespeare, the model produces text that's character-valid and locally coherent. It won't fool anyone, but it gets the rhythm right:

```
ROMEO:
What says my love? I have not seen him so,
For I have been so much of his company,
And yet I know not what to say to him.
```

Greedy decoding tends to get stuck in loops. Temperature around 0.8 with top-k of 40 is the sweet spot — diverse enough to avoid repetition, focused enough to stay grammatical. Above 1.2 the model starts inventing words.

---

## Things that will break (and how to fix them)

| Symptom | Likely cause |
|---|---|
| `CUDA out of memory` | Drop `batch_size` to 32 or 16 |
| Loss goes to `nan` | Learning rate too high — try `1e-4` |
| Loss stuck near 4.17 | Tokenizer isn't seeing data, or LR is far too low |
| `mat1 and mat2 shapes cannot be multiplied` | `n_embd` not divisible by `n_head` |
| CUDA driver warning on import | Your NVIDIA driver is older than your PyTorch build; run on CPU or update the driver |

---

## Reading list

The papers and repos that shaped this implementation:

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — GPT-2, Radford et al., 2019
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) — Pre-LN, Xiong et al., 2020
- [nanoGPT](https://github.com/karpathy/nanoGPT) — Andrej Karpathy's minimal GPT, which this project draws heavily from in spirit

---

## License

MIT. Take it, modify it, use it for whatever. If it helps you understand Transformers a bit better, it did its job.