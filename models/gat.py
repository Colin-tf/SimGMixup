from models.gnn import BaseGNN
from torch_geometric.nn import GATConv
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch_geometric.nn import global_mean_pool
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d

class GAT(BaseGNN):
    def __init__(self, in_channels, hid_channels, out_channels, dropout, num_layers: int=2, task: str="node", **kwds):
        super().__init__(in_channels, hid_channels, out_channels, dropout, num_layers, task, **kwds)  # Correctly initialize the base class
        self.convs = nn.ModuleList()
        self.build()
        
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        # for mixup_bn in self.mixup_bns:
        #     mixup_bn.reset_parameters()
            
    def build(self):
        heads = 8
        self.convs.append(GATConv(self.in_channels, self.hid_channels, heads=heads))
        self.bns.append(BatchNorm1d(self.hid_channels * heads))
        # self.mixup_bns.append(BatchNorm1d(self.hid_channels * heads))

        if self.num_layer > 2:
            for _ in range(self.num_layer-2):
                self.convs.append(GATConv(self.hid_channels * heads, self.hid_channels * heads, heads=heads))
                self.bns.append(BatchNorm1d(self.hid_channels * heads))
                # self.mixup_bns.append(BatchNorm1d(self.hid_channels * heads))

        self.convs.append(GATConv(self.hid_channels * heads, self.out_channels if self.task == "node" else self.hid_channels, heads=1))
        
        if self.task == "graph":
            self.bns.append(BatchNorm1d(self.hid_channels))
            # self.mixup_bns.append(BatchNorm1d(self.hid_channels))
            self.fcs = Sequential(Linear(self.hid_channels, self.hid_channels), ReLU(), Linear(self.hid_channels, self.out_channels))
                 