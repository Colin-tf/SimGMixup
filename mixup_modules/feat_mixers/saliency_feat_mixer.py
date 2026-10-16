import torch
from mixup_modules.feat_mixers import FeatureMixer

class SaliencyFeatMixer(FeatureMixer):
    def __init__(self):
        super().__init__()
        
    def feat_mixup(self, 
                   x_l: torch.FloatTensor, 
                   x_r: torch.FloatTensor, 
                   lam: torch.FloatTensor,
                   saliency: torch.FloatTensor,
                   keep_prob: float = 0.3
                   ) -> torch.FloatTensor:
        node_dim = saliency.shape[1]
        saliency_dist = saliency.abs()
        saliency_dist += 1e-10
        saliency_dist /= torch.sum(saliency_dist, dim=1)
        
        K = int(node_dim * keep_prob)
        mask_idx = torch.multinomial(saliency_dist, K)
        lam = lam.expand(node_dim).clone()
        lam[torch.arange(lam.shape[0]), mask_idx] = 1.
        mixed_x = lam * x_l + (1-lam) * x_r
        
        return mixed_x