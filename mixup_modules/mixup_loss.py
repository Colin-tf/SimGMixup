import torch.nn.functional as F
import torch

def mixup_loss_func(logits, y_l, y_r, lam) -> torch.FloatTensor:
    loss = lam * F.cross_entropy(logits, y_l) + (1 - lam) * F.cross_entropy(logits, y_r)
    return loss.mean()