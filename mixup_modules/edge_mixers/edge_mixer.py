from mixup_modules.objects import Subgraph, MixedIndexContainer
from abc import ABC, abstractmethod
import torch
from typing import Tuple
from torch_sparse import SparseTensor

class EdgeMixer(ABC):
    def __init__(self) -> None:
        self._mixed_index_container_cache = dict()
        pass

    @abstractmethod
    def edge_mixup(self, 
                    g_l: Subgraph,
                    g_r: Subgraph):
        raise NotImplemented("EdgeMixer.edge_mixup() should be implemented.")

    def _update_mixed_index_container_cache(self, 
                                            key: Tuple[int, int],
                                            mixed_index_container: MixedIndexContainer):
        
        self._mixed_index_container_cache[key] = mixed_index_container
    
    def _get_mixed_index_container_from_cache(self, 
                                              key: Tuple[int, int]) -> MixedIndexContainer:
        
        if key not in self._mixed_index_container_cache:
            raise ValueError(f"Mixed index container for key {key} not found in cache.")
        
        return self._mixed_index_container_cache[key]

    def _existence_check(self, key: Tuple[int, int]) -> bool:
        return key in self._mixed_index_container_cache
    
    def _to_sparse(self, edge_index: torch.LongTensor) -> SparseTensor:   
        if  edge_index[0].size(0) != edge_index[1].size(0):
            print("edge_index is not a square matrix, converting to sparse tensor...")
        return SparseTensor(row=edge_index[0], col=edge_index[1])
    
    def __call__(self,  
                g_l: Subgraph,
                g_r: Subgraph,
                node_index_l: int,
                node_index_r: int,
                cache: bool = True,
                sparse: bool = False) -> MixedIndexContainer:
        return self.edge_mixup(g_l, g_r, node_index_l, node_index_r, cache, sparse)


