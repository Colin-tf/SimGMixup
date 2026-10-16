from .gnn import BaseGNN
from .gcn import GCN
from .gat import GAT
from .sage import SAGE
from .gin import GIN

__all__ = [
    'BaseGNN',
    'GCN',
    'GAT',
    'SAGE',
    'GIN'
]

classes = __all__
