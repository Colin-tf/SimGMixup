import torch
from abc import ABC, abstractmethod

class FeatureMixer(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def feat_mixup(self, 
                   x_l: torch.FloatTensor, 
                   x_r: torch.FloatTensor, 
                   lam: torch.FloatTensor, 
                   *args
                   ) -> torch.FloatTensor:
        # x_mix = lam * x_l + (1 - lam) * x_r
        raise NotImplemented("FeatureMixer.feat_mixup should be implemeted.")
        
    def __call__(self, x_l: torch.FloatTensor, x_r: torch.FloatTensor, lam: torch.FloatTensor, *args) -> torch.FloatTensor:
        return self.feat_mixup(x_l, x_r, lam, *args)


