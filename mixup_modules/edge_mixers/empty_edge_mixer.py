from mixup_modules.edge_mixers.edge_mixer import *
from mixup_modules.objects import Subgraph, MixedIndexContainer
import torch

class EmptyEdgeMixer(EdgeMixer):
    '''Only Return the Empty Edge Index'''
    def __init__(self) -> None:
        super().__init__()

    def edge_mixup(self,
                    g_l: Subgraph,
                    g_r: Subgraph) -> MixedIndexContainer:
        node_index_list_l, _, mapping_l = g_l
        node_index_list_r, _, mapping_r = g_r
        central_node_index_l, central_node_index_r = node_index_list_l[mapping_l], node_index_list_r[mapping_r]
        mixed_edge_index = torch.empty(size=(2, 0), dtype=torch.int64)
        mixed_node_index = torch.cat([node_index_list_l, torch.cat([node_index_list_r[:mapping_r], node_index_list_r[mapping_r+1:]])])

        return MixedIndexContainer(mixed_node_index=mixed_node_index, 
                                   mixed_edge_index=mixed_edge_index, 
                                   mapping=mapping_l, 
                                   central_node_index_l=central_node_index_l, 
                                   central_node_index_r=central_node_index_r,
                                   node_index_l=node_index_list_l,
                                   node_index_r=node_index_list_r)