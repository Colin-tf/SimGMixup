from models.gnn import BaseGNN
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d

class SAGE(BaseGNN):
    def __init__(self, in_channels, hid_channels, out_channels, dropout, num_layers: int=2, task: str="node", **kwds):
        super().__init__(in_channels, hid_channels, out_channels, dropout, num_layers, task, **kwds) 
        self.build()
        
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        # for mixup_bn in self.mixup_bns:
        #     mixup_bn.reset_parameters()
            
    def build(self):
        self.convs.append(SAGEConv(self.in_channels, self.hid_channels))
        self.bns.append(BatchNorm1d(self.hid_channels))
        # self.mixup_bns.append(BatchNorm1d(self.hid_channels))

        if self.num_layer > 2:
            for _ in range(self.num_layer-2):
                self.convs.append(SAGEConv(self.hid_channels, self.hid_channels))
                self.bns.append(BatchNorm1d(self.hid_channels))
                # self.mixup_bns.append(BatchNorm1d(self.hid_channels))

        self.convs.append(SAGEConv(self.hid_channels, self.out_channels if self.task == "node" else self.hid_channels))
        
        if self.task == "graph":
            self.bns.append(BatchNorm1d(self.hid_channels))
            self.mixup_bns.append(BatchNorm1d(self.hid_channels))
            # self.fcs = Sequential(Linear(self.hid_channels, self.hid_channels), ReLU(), Linear(self.hid_channels, self.out_channels))
                 
    
