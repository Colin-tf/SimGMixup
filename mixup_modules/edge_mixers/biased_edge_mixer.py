from mixup_modules.edge_mixers.edge_mixer import *
from mixup_modules.objects import Subgraph, MixedIndexContainer

class BiasedEdgeMixer(EdgeMixer):
    def __init__(self) -> None:
        super().__init__()

    def edge_mixup(self,
                    g_l: Subgraph,
                    g_r: Subgraph,
                    node_index_l: int,
                    node_index_r: int,
                    cache: bool = True,
                    sparse: bool = False,) -> MixedIndexContainer:
        
        node_pair = (node_index_l, node_index_r)
        
        if self._existence_check(node_pair) and cache:
            # if the edge mixup has been computed before, retrieve the result from cache
            return self._get_mixed_index_container_from_cache(node_pair)
        
        node_index_list_l, edge_index_l, mapping_l = g_l
        node_index_list_r, _, mapping_r = g_r
        central_node_index_l, central_node_index_r = node_index_list_l[mapping_l], node_index_list_r[mapping_r]

        if sparse:
            # convert the mixed edge index to a sparse tensor
            mixed_edge_index = self._to_sparse(mixed_edge_index)
            
        mixed_index_container = MixedIndexContainer(
                                    mixed_node_index=node_index_list_l, 
                                   mixed_edge_index=edge_index_l, 
                                   mapping=mapping_l, 
                                   central_node_index_l=central_node_index_l, 
                                   central_node_index_r=central_node_index_r,
                                   node_index_l=node_index_list_l,
                                   node_index_r=node_index_list_r)
        if cache:
            # cache the mixed index container for future use
            self._update_mixed_index_container_cache(node_pair, mixed_index_container)
            
        return mixed_index_container