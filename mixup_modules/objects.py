from typing import NamedTuple, List
import torch
from torch_sparse import SparseTensor

class Subgraph(NamedTuple):
    """extracted subgraph"""
    node_index: torch.LongTensor # node indexs 
    # x: torch.FloatTensor # node features
    edge_index: torch.LongTensor # edge indexs 
    mapping: torch.LongTensor # mapping from the central node in subgraph to original graph
    # y: torch.LongTensor # label of (the central node in) subgraph 
    # num_nodes: int # number of nodes in subgraph

class MixedIndexContainer(NamedTuple):
    """index container prepared for mixup"""
    # x_l: torch.FloatTensor  # left node features
    # x_r: torch.FloatTensor  # right node features
    mixed_node_index: torch.LongTensor  # mixed node index
    mixed_edge_index: torch.LongTensor  # mixed edge index
    mapping: torch.LongTensor  # mapping from mixed to original node index
    central_node_index_l: torch.LongTensor  # left central node index
    central_node_index_r: torch.LongTensor  # right central node index
    node_index_l: torch.LongTensor  # left central node index
    node_index_r: torch.LongTensor  # right central node index
    
class MixedGraph(NamedTuple):
    """mixed graph"""
    x: torch.FloatTensor  # node features
    edge_index: torch.LongTensor  # edge index
    is_sparse: bool  # whether the graph is sparse or not
    sparse_edge_index: SparseTensor  # sparse edge index
    y_l: torch.LongTensor  # label of left graph (node)
    y_r: torch.LongTensor  # label of right graph (node)
    lam: torch.FloatTensor  # lambda value for mixup
    mapping: torch.LongTensor  # mapping from mixed to original node index
    central_node_index: torch.LongTensor  # central node index in mixed graph
    is_input: bool  # whether the graph is input graph or not