'''
# Learning Rate Schedule -  Warmup + Cosine 
'''

import math

def get_lr(it, cfg):
    """
    Learning-rate schedule:
      1) Linear warmup from 0 to lr over `warmup_iters` steps.
      2) Cosine decay from lr down to `min_lr` over remaining steps.
    """
    # 1) Linear warmup
    if it < cfg.warmup_iters:
        return cfg.learning_rate * (it + 1) / (cfg.warmup_iters + 1)

    # 2) After decay horizon -> floor value
    if it > cfg.lr_decay_iters:
        return cfg.min_lr

    # 3) Cosine decay between warmup and decay end
    decay_ratio = (it - cfg.warmup_iters) / (cfg.lr_decay_iters - cfg.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))   # 1 -> 0
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)