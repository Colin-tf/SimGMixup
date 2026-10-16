import torch
from abc import ABC, abstractmethod
from mixup_modules.extractor import *
from mixup_modules.edge_mixers import *
from mixup_modules.feat_mixers import *
from mixup_modules.samplers import *
from mixup_modules.samplers.functional import get_sampling_pool
from mixup_modules.mixing_ratios import *
from torch_geometric.data import Data
from graph import Graph
from config import device
from typing import List

class GraphMixer(ABC):
    def __init__(self) -> None:
        pass
    
    @property
    def mode(self):
        pass
    
    @abstractmethod
    def mixup(g: Graph, 
              p: torch.LongTensor = None, 
              x: torch.LongTensor = None, 
              saliency: torch.LongTensor = None, 
              sample_ratio: float = 1.) -> Data:
        raise NotImplementedError("Mixer.mixup should be implemented.")
    
    def generate_batches(self, train_size: int, batch_size: int) -> List[torch.LongTensor]:
        train_indices = torch.arange(train_size).tolist()
        batches = [train_indices]
        if batch_size > 0:
            batches = []
            batch_num = int(train_size / batch_size) if int(train_size / batch_size) * batch_size >= train_size else int(train_size / batch_size) + 1
            for i in range(batch_num - 1):
                batches.append(train_indices[i*batch_size:(i+1)*batch_size])
            batches.append(train_indices[batch_size*(batch_num-1):])
        
        return batches

    def __call__(self, *args: torch.Any, **kwds: torch.Any) -> torch.Any:
        return self.mixup(*args, **kwds)
    
