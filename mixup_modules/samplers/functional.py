from .sampler import *
import torch

def get_sampling_pool(g: Graph, mode: str, pl_threshold: float=0.5, p: torch.FloatTensor=None) -> Tuple[List[torch.LongTensor], List[torch.LongTensor]]:
    if mode.lower() == "ll":
        index_l = g.train_mask.nonzero().squeeze()
        index_r = index_l.clone()
    else:
        unlabeled_index = g.unlabeled_mask.nonzero().squeeze()
        if p is not None:
            # 确定无标签节点的预测概率最大值和对应的类别
            prob_max, pseudo_labels = torch.max(p[unlabeled_index], dim=1)
            # 超过阈值的伪标签掩码
            mask = prob_max > pl_threshold
            # 更新这些节点的伪标签
            g.y_tmp[unlabeled_index[mask]] = pseudo_labels[mask]
            # 更新 index_r 为只包含被赋予伪标签的节点索引
            index_r = unlabeled_index[mask]
        else:
            index_r = torch.tensor([], dtype=torch.long)
        if mode.lower() == "lu":
            index_l = g.train_mask.nonzero().squeeze()
        elif mode.lower() == "uu":
            index_l = index_r.clone()
    
    return (index_l, index_r)