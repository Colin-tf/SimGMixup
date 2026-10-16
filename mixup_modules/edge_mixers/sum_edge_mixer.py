from mixup_modules.edge_mixers.edge_mixer import *
from mixup_modules.objects import Subgraph, MixedIndexContainer
import torch

class SumEdgeMixer(EdgeMixer):
    def __init__(self) -> None:
        super().__init__()

    def edge_mixup(self,
                    g_l: Subgraph,
                    g_r: Subgraph,
                    node_index_l: int,
                    node_index_r: int,
                    cache: bool = True,
                    sparse: bool = False,
                ) -> MixedIndexContainer:
        
        node_pair = (node_index_l, node_index_r)
        
        if self._existence_check(node_pair) and cache:
            # if the edge mixup has been computed before, retrieve the result from cache
            return self._get_mixed_index_container_from_cache(node_pair)
        
        node_index_list_l, edge_index_l, mapping_l = g_l
        node_index_list_r, edge_index_r, mapping_r = g_r
        central_node_index_l, central_node_index_r = node_index_list_l[mapping_l], node_index_list_r[mapping_r]
        central_node_mask_r = edge_index_r == mapping_r
        new_edge_index_r = edge_index_r.clone()
        new_edge_index_r[central_node_mask_r] = mapping_l
        if edge_index_l.numel() > 0:
            # new_edge_index_r = edge_index_r + edge_index_l.max()
            new_edge_index_r = edge_index_r + edge_index_l.max()
        mixed_edge_index = torch.cat([edge_index_l, new_edge_index_r], dim=1)
        mixed_node_index = torch.cat([node_index_list_l, torch.cat([node_index_list_r[:mapping_r], node_index_list_r[mapping_r+1:]])])
        
        if sparse:
            # convert the mixed edge index to a sparse tensor
            mixed_edge_index = self._to_sparse(mixed_edge_index)
        if isinstance(mixed_edge_index, SparseTensor):
            if mixed_edge_index.size(0) != mixed_edge_index.size(1):
                print("Warning: the number of nodes in the mixed edge index does not match the number of nodes in the mixed node index. This may cause errors in downstream tasks.")
            
        mixed_index_container = MixedIndexContainer(
                                mixed_node_index=mixed_node_index, 
                                mixed_edge_index=mixed_edge_index, 
                                mapping=mapping_l, 
                                central_node_index_l=central_node_index_l, 
                                central_node_index_r=central_node_index_r,
                                node_index_l=node_index_list_l,
                                node_index_r=node_index_list_r)
        if cache:
            # cache the mixed index container for future use
            self._update_mixed_index_container_cache(node_pair, mixed_index_container)
            
        return mixed_index_container
        