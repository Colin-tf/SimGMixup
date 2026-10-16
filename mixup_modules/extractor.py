from torch_geometric.utils import k_hop_subgraph, contains_self_loops, add_self_loops
import os.path as osp
import torch
import os
from typing import Tuple, Dict
from config import device
from mixup_modules.objects import Subgraph
from graph import Graph

class Extractor:
    def __init__(self, to_folder: str, hop: int = 2):
        self.to_folder = to_folder
        self.hop = hop

        self._subgraph_dict = dict()
        self._subgraph_cache = None
        self._mapping_cache = None
        
    def extract_subgraph(self, 
                         g: Graph, 
                         node_indices: torch.LongTensor,
                         cache: bool = True
                         ) -> Tuple[Subgraph, Dict]:
        subgraph_mapping = dict()
        edge_index = g.edge_index
        
        if self._subgraph_cache is not None and cache:
            return self._subgraph_cache, self._mapping_cache
        else:
            subgraph_list = []
            for i in range(len(node_indices)):
                node = node_indices[i].item()
                if node in self._subgraph_dict:
                    subgraph = self._subgraph_dict[node]
                else:
                    try:
                        extracted_subgraph = k_hop_subgraph(node, self.hop, edge_index, relabel_nodes=True)
                        sub_node_index, sub_edge_index, mapping, _ = extracted_subgraph
                    except:
                        sub_node_index = torch.LongTensor([node]).to(device)
                        sub_edge_index = torch.LongTensor([[0], [0]]).to(device)
                        mapping = torch.LongTensor([0]).to(device)
                    subgraph = Subgraph(node_index=sub_node_index, edge_index=sub_edge_index, mapping=mapping)
                    
                subgraph_list.append(subgraph)
                subgraph_mapping[node] = i
            self._subgraph_cache = subgraph_list
            self._mapping_cache = subgraph_mapping

        return subgraph_list, subgraph_mapping
        
            
    def __call__(self, 
                edge_index: torch.LongTensor, 
                node_indices: torch.LongTensor,
                cache: bool = True
                ) -> Tuple[Subgraph, Dict]:
        return self.extract_subgraph(edge_index, node_indices, cache)