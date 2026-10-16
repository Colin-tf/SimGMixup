from abc import ABC, abstractmethod
import torch
from mixup_modules.objects import MixedIndexContainer

class MixingRatio(ABC):
    def __init__(self, pair_wise: bool = False, *args):
        self.lam = None
        self.pair_wise = pair_wise
        
    @abstractmethod
    def compute(self, p: torch.LongTensor=None, x: torch.LongTensor=None, grad: torch.LongTensor=None, mixed_index_container: MixedIndexContainer=None) -> torch.LongTensor:
        pass
    
    def clear(self):
        self.lam = None
    
    def _update(self, lam: torch.LongTensor):
        self.lam = lam
        
    def _get(self) -> torch.LongTensor:
        return self.lam
    




