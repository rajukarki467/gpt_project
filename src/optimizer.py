'''
#  Optimizer — AdamW with Parameter Groups 
- requires weight decay only on 2D matrices.

'''

import torch

def configure_optimizer(model, cfg):
    """
    Build an AdamW optimizer with two parameter groups:
      - decay: 2D params (weights of Linear/Embedding) that get weight decay
      - no_decay: 1D params (biases, LayerNorm gains) with NO weight decay
    """
    # Collect all params that require grad
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}

    decay_params    = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params  = [p for n, p in param_dict.items() if p.dim() < 2]

    optim_groups = [
        {"params": decay_params,   "weight_decay": cfg.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    print(f"[optimizer] decay params: {len(decay_params)}, "
          f"no-decay params: {len(nodecay_params)}")

    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
    )
    return optimizer