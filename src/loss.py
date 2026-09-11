# Loss
# 
import torch.nn.functional as F

def lm_loss(logits, targets):
    """
    Cross-entropy loss for language modeling.
    logits:  (B, T, vocab_size)
    targets: (B, T)
    """
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=-1,
    )