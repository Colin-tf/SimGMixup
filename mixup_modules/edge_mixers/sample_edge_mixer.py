from mixup_modules.edge_mixers.edge_mixer import *
from mixup_modules.objects import Subgraph, MixedIndexContainer
import torch
from torch_geometric.utils import dropout_edge

class SampleEdgeMixer(EdgeMixer):
    def __init__(self, p:float = 0.5) -> None:
        super().__init__()
        self.p = p

    def edge_mixup(self,
                    g_l: Subgraph,
                    g_r: Subgraph,
                    node_index_l: int = None,
                    node_index_r: int = None,
                    cache: bool = False,
                    sparse: bool = False) -> MixedIndexContainer:
        
        node_index_list_l, edge_index_l, mapping_l = g_l
        node_index_list_r, edge_index_r, mapping_r = g_r
        central_node_index_l, central_node_index_r = node_index_list_l[mapping_l], node_index_list_r[mapping_r]

        drop_edge_index_l = dropout_edge(edge_index_l, self.p)[0]
        drop_edge_index_r = dropout_edge(edge_index_r, 1 - self.p)[0]
        central_node_mask_r = drop_edge_index_r == mapping_r
        new_edge_index_r = drop_edge_index_r.clone()
        new_edge_index_r[central_node_mask_r] = mapping_l
        if drop_edge_index_l.numel() > 0:
            new_edge_index_r = drop_edge_index_r + drop_edge_index_l.max()
        mixed_edge_index = torch.cat([drop_edge_index_l, new_edge_index_r], dim=1)
        mixed_node_index = torch.cat([node_index_list_l, torch.cat([node_index_list_r[:mapping_r], node_index_list_r[mapping_r+1:]])])
        
        if sparse:
            # convert the mixed edge index to a sparse tensor
            mixed_edge_index = self._to_sparse(mixed_edge_index)
            
        return MixedIndexContainer(mixed_node_index=mixed_node_index, 
                                   mixed_edge_index=mixed_edge_index, 
                                   mapping=mapping_l, 
                                   central_node_index_l=central_node_index_l, 
                                   central_node_index_r=central_node_index_r,
                                   node_index_l=node_index_list_l,
                                   node_index_r=node_index_list_r)