# MiniGPT: A Decoder-Only Transformer From Scratch

I built this to properly understand how GPT-style language models work under the hood. Instead of importing `nn.TransformerDecoder` and calling it a day, everything here — causal attention, the Transformer block, weight tying, the training loop — is written from PyTorch primitives. It's a small model (**0.82 M parameters**) trained on Tiny Shakespeare, and it finishes 5000 training steps in **under 4 minutes** on a laptop GPU.

The goal wasn't to produce state-of-the-art text. It was to make every line of the forward pass explainable.

---

## Results at a glance

| Metric | Value |
|---|---|
| Model parameters | 0.82 M |
| Training steps | 5000 |
| Final train loss | **1.613** |
| Final val loss | **1.545** |
| Final val perplexity | **≈ 4.69** |
| Wall-clock time | **3.83 min** |
| Throughput | **~28 it/s** (steady state) |
| Best checkpoint | `outputs/checkpoints/best.pt` |

Sample output after 5000 steps:

```
ROMEO: be more my fear.

BUCKINGHAM:
What, I beter the maid me, lord be the sold me
Besole reigness me with do of the did soul. Welcand you not
that I love not so weep even that I with intersise.
```

---

## What's inside

- **Causal multi-head self-attention** implemented by hand, including the lower-triangular mask
- **Pre-LayerNorm Transformer blocks** with residual connections and a GELU feed-forward network
- **Learned token + positional embeddings**, with weight tying between the embedding and LM head
- **Mixed-precision training** using `torch.autocast` and `GradScaler`
- **Linear warmup + cosine decay** learning-rate schedule
- **AdamW** with weight decay applied *only* to 2D parameters (18 decay / 34 no-decay tensors)
- **Three sampling modes** for generation: greedy, temperature, and top-k
- **Diagnostic visualisations**: loss curve, LR curve, attention heatmap

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
│   └── input.txt
│
├── outputs/
│   ├── checkpoints/
│   │   ├── best.pt              # final model weights
│   │   └── training_log.json    # per-step loss + LR history
│   └── samples/
│       ├── all_samples.txt
│       ├── attention_heatmap.png
│       ├── loss_curve.png
│       ├── lr_schedule.png
│       ├── lr_used.png
│       ├── sample_greedy.txt
│       ├── sample_temperature_0.5.txt
│       ├── sample_temperature_0.8_k40.txt
│       ├── sample_temperature_1.0.txt
│       └── sample_temperature_1.5.txt
│
└── src/
    ├── dataset.py         # char-level tokenizer + LM dataset
    ├── model.py           # attention, block, GPT
    ├── loss.py            # cross-entropy wrapper
    ├── optimizer.py       # AdamW param grouping
    ├── scheduler.py       # warmup + cosine
    ├── train.py           # training loop with AMP + logging
    ├── generate.py        # sampling (greedy / temperature / top-k)
    └── visualize.py       # loss curve, LR curve, attention map, samples
```

---

## Getting it running

```bash
git clone <your-repo-url>
cd gpt_project
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Train the model:

```bash
python src/train.py
```

The Tiny Shakespeare corpus (~1.1 MB) downloads itself on the first run. With the default config (5000 steps, batch size 64, context length 128), a modern GPU finishes in **~4 minutes**.

Generate samples:

```bash
python src/generate.py
```

This runs greedy, temperature-0.8-with-top-k-40, and high-temperature sampling from the same prompt so you can compare them side by side.

Produce all report figures:

```bash
python src/visualize.py
```

This writes the loss curve, LR curve, attention heatmap, and all sample `.txt` files into `outputs/samples/`.

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
| FFN hidden size | 4 × 128 = 512 |
| Parameters | ~0.82 M |

If you want a more capable model, bump `n_layer` to 6 and `n_embd` to 256 in `config.py`. You'll get noticeably better output at the cost of speed.

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

**Weight decay grouping.** Only tensors with `dim() >= 2` get weight decay. Biases, LayerNorm gains, and embedding tables are excluded. In this run, **18 tensors received weight decay and 34 did not**. This matches the GPT-3 recipe; decaying a LayerNorm gain toward zero is not something you actually want.

**Mixed precision.** `bfloat16` when the GPU supports it (no scaler needed), otherwise `float16` with a `GradScaler`. The autocast context wraps only the forward pass and loss — the backward pass runs in the scaled dtype automatically.

**Gradient clipping.** Global norm clipped to 1.0 every step. It's a cheap safety net against the occasional loss spike.

**Logging.** The training loop writes `outputs/checkpoints/training_log.json` after every evaluation, containing per-step train loss, LR, and per-eval validation loss. This makes it possible to redraw the loss curve at any time without retraining.

---

## What the output looks like

The five sampling strategies, on the same prompt, tell a very clear story about the coherence–diversity trade-off:

**Greedy** — collapses into loops within 15 tokens:

```
ROMEO: the shall shall be the stand of the stand
The shall be the some of the some of the son,
And the shall be the shall of the shall be the sent
```

**Temperature 0.8, top-k 40** — the sweet spot:

```
ROMEO: her, that no shall the hast bidgoth,
Show from and since and her stand truth
Than should to king from Patcive.

PAULINA:
Thou hast love merry me well the worthy own of must
```

**Temperature 1.5** — creative but incoherent:

```
ROMEO: 'TOnpe shall thou god my fnighthting, not,
Comentent,d, Larlen i now-cae hecce.
```

Full side-by-side comparison in `outputs/samples/all_samples.txt`.

---

## Diagnostic figures

All figures are generated by `python src/visualize.py`:

- **`loss_curve.png`** — training (per-step, faded) + validation (per-eval, bold)
- **`lr_used.png`** — the actual learning rate used at each step
- **`lr_schedule.png`** — the idealized schedule (useful for the report's methods section)
- **`attention_heatmap.png`** — attention weights from all heads of the first block, for the prompt `"ROMEO: "`. Shows strictly lower-triangular structure, confirming the causal mask.

---

## Things that will break (and how to fix them)

| Symptom | Likely cause |
|---|---|
| `CUDA out of memory` | Drop `batch_size` to 32 or 16 |
| Loss goes to `nan` | Learning rate too high — try `1e-4` |
| Loss stuck near 4.17 | Tokenizer isn't seeing data, or LR is far too low |
| `mat1 and mat2 shapes cannot be multiplied` | `n_embd` not divisible by `n_head` |
| CUDA driver warning on import | Your NVIDIA driver is older than your PyTorch build; run on CPU or update the driver |
| `KeyError: 'iters'` in `visualize.py` | Old checkpoint. Retrain with the updated `train.py` to produce `training_log.json` |

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