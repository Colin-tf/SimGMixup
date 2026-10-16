import torch
from mixup_modules.feat_mixers import FeatureMixer
from mixup_modules.edge_mixers import MixedIndexContainer

class GGFeatMixer(FeatureMixer):
    def __init__(self):
        super().__init__()

    def feat_mixup(self, 
                   x: torch.FloatTensor, 
                   mixed_index_container: MixedIndexContainer, 
                   lam: torch.FloatTensor
                   ) -> torch.FloatTensor:
        x_l, x_r = x[mixed_index_container.node_index_l].mean(dim=0), x[mixed_index_container.node_index_r].mean(dim=0)
        mixed_x = lam * x_l + (1 - lam) * x_r
        if mixed_index_container.mixed_node_index is not None:
            mixed_graph_x = x[mixed_index_container.mixed_node_index]
            mixed_graph_x[mixed_index_container.mapping] = mixed_x
        else:
            mixed_graph_x = mixed_x
            
        
        return mixed_graph_x
