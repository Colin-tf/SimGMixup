import torch
from abc import ABC, abstractmethod
from torch_geometric.nn import global_mean_pool
from torch_sparse import SparseTensor
from config import device
import torch.nn.functional as F
import torch.nn as nn

class BaseGNN(torch.nn.Module, ABC):
    """Base class for GNNs."""
    def __init__(self, in_channels, hid_channels, out_channels, dropout, num_layers: int=2, task: str="node", hidden_layer_index:int=0, **kwds):
        super().__init__()  # Initialize the nn.Module correctly
        assert num_layers >= 2, "Number of layers must be at least 2"
        assert hidden_layer_index < num_layers, "Hidden layer index must be less than the number of layers minus 1"
        
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.num_layer = num_layers
        self.task = task
        self.hidden_layer_index = hidden_layer_index
        self.kwds = kwds
        
        self._conv_embedding_cache = None
        self._fcs_embedding_cache = None
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.mixup_bns = nn.ModuleList()
        self.fcs = None
        
        self.input_mode_mixup_forward_layers = None
        self.hidden_mode_mixup_forward_layers = None
        self.mixup_bns = None


    def build_mixup_forward_layers(self, mode:str = "input"):
        self.mixup_bns = nn.ModuleList()
        
        if mode == "input":
            self.input_mode_mixup_forward_layers = nn.ModuleList()
            for conv in self.convs:
                self.input_mode_mixup_forward_layers.append(conv)
            for bn in self.bns:
                self.mixup_bns.append(bn)
                
        elif mode == "hidden":
            self.hidden_mode_mixup_forward_layers = nn.ModuleList()
            
            if self.task == "node":
                for conv in self.convs[1:]:
                    self.hidden_mode_mixup_forward_layers.append(conv)
                    
            for bn in self.bns:
                self.mixup_bns.append(bn)


    def get_embedding(self) -> torch.FloatTensor:
        if self.task == "node":
            return self._conv_embedding_cache
        elif self.task == "graph":
            return self._fcs_embedding_cache
    
    def _reset_embedding_cache(self):
        self._conv_embedding_cache = None
        self._fcs_embedding_cache = None

    def mixup_forward(self, mixed_g, is_sparse: bool=False, mixup_mode: str="input") -> torch.FloatTensor:
        """mixup forward"""
        x = mixed_g.x
        edge_index = mixed_g.edge_index 
    
        if is_sparse and mixup_mode == "input":
            edge_index = SparseTensor(row=edge_index[0], col=edge_index[1])
        
        convs = self.input_mode_mixup_forward_layers if mixup_mode == "input" else self.hidden_mode_mixup_forward_layers
       
        if self.task == "graph" and mixup_mode == "hidden":
            x = self.mixup_bns[-1](x)
            return self.fcs(x)
        
        for i, conv in enumerate(convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            
            if i < len(convs) - 1:
                if self.task == "graph":
                    x = self.mixup_bns[i](x)

        if self.task == "graph":
            x = self.mixup_bns[-1](x)
            x = global_mean_pool(x, mixed_g.batch)
            x = self.fcs(x)
                
        return x
    
    def forward(self, g) -> torch.FloatTensor:
        self._reset_embedding_cache()
        
        x = g.x
        
        if hasattr(g, 'is_sparse'):
            edge_index = g.edge_index if not g.is_sparse else g.sparse_edge_index
        else:
            edge_index = g.edge_index
            
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            if self.task == "graph":
                x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if i == 0:
                self._conv_embedding_cache = x.clone()
        x = self.convs[-1](x, edge_index)

        if self.task == "graph":
            x = self.bns[-1](x)
            x = global_mean_pool(x, g.batch)
            self._fcs_embedding_cache = x.clone()
            x = self.fcs(x)
            
            
        return x