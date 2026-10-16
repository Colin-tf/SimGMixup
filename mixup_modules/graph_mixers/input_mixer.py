from mixup_modules.graph_mixers.graph_mixer import *
from torch_sparse import SparseTensor
from mixup_modules.objects import MixedGraph, Subgraph
from typing import List
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import degree
from config import device

class InputMixer(GraphMixer):
    def __init__(self, 
                edge_mixer: EdgeMixer, 
                feat_mixer: FeatureMixer, 
                sampler: Sampler, 
                mixing_ratio: MixingRatio, 
                extractor: Extractor, 
                **kwds) -> None:

        super().__init__()      
        self.edge_mixer = edge_mixer
        self.feat_mixer = feat_mixer
        self.sampler = sampler
        self.mixing_ratio = mixing_ratio
        self.extractor = extractor
        self.kwds = kwds
    
    @property
    def mixup_mode(self):
        return "input"
    
    def mixup_node_task(self, g: Graph,
                        p: torch.LongTensor = None, 
                        h: torch.LongTensor = None,
                        grad: torch.LongTensor = None,
                        sample_ratio: float = 1.,
                        cache: bool = True):
        
        # Sampling Node Index for Mixup
        index_list_l, index_list_r = self.sampler(g, sample_ratio, p)
        if len(index_list_l) == 0 or len(index_list_r) == 0:
            return []
        # print(index_list_l, index_list_r)
        # Extract Subgraph
        extract_cache = False if sample_ratio < 1 or self.sampler.mode is not "LL" else True
        subgraph_list, mapping = self.extractor(g, torch.unique(torch.cat([index_list_l, index_list_r])), cache=extract_cache)
        # print(subgraph_list, mapping)
        x, y_tmp = g.x.clone(), g.y_tmp.clone()

        mixed_graphs_list = []
        # Lambda Generator
        # lam = self.mixing_ratio.compute()
        
        for node_index_l, node_index_r in zip(index_list_l, index_list_r):
            # Edge Mixup
            node_pair = (mapping[node_index_l.item()], mapping[node_index_r.item()])
            # mixed_index_container = self.edge_mixer(subgraph_list[node_pair[0]], subgraph_list[node_pair[1]], node_pair[0], node_pair[1], cache=cache, sparse=g.is_sparse)
            mixed_index_container = self.edge_mixer(subgraph_list[node_pair[0]], subgraph_list[node_pair[1]], node_pair[0], node_pair[1], cache=cache)
            
            # Lambda Generator
            lam = self.mixing_ratio.compute(p, x, grad, mixed_index_container)
            
            # Feature Mixup
            mixed_x = self.feat_mixer(x, mixed_index_container, lam)

            # Mixed Graph Generation
            mixed_graph = Data(x=mixed_x, 
                               edge_index=mixed_index_container.mixed_edge_index, 
                               y_l=y_tmp[node_index_l], 
                               y_r=y_tmp[node_index_r], 
                               lam=lam,
                               mapping=mixed_index_container.mapping, 
                               central_node_index=mixed_index_container.central_node_index_l,
                            #    is_input=True,
                            #    is_sparse=g.is_sparse
                               )
            mixed_graphs_list.append(mixed_graph)
        batch_size = self.kwds["batch_size"] if self.kwds["batch_size"] > 0 else len(mixed_graphs_list)
        loader = DataLoader(mixed_graphs_list, batch_size=batch_size, shuffle=True)
        
        self.mixing_ratio.clear()
        return loader
            
    def mixup(self, 
              g: Graph,
              p: torch.LongTensor = None, 
              h: torch.LongTensor = None, 
              grad: torch.LongTensor = None,
              sample_ratio: float = 1.,
              cache: bool = True,
              task: str = "node"
              ) -> Data:
        
        if task == "node":
            return self.mixup_node_task(g, p, h, grad, sample_ratio, cache)
        else:
            raise ValueError("Invalid task type!")