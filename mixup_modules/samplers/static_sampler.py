from mixup_modules.samplers.sampler import *
from .functional import get_sampling_pool

class StaticSampler(Sampler):
    def __init__(self,mode: str, pl_thresehold: float=0.5) -> None:
        super().__init__()
        self._cache = None
        self.mode = mode
        self.pl_thresehold = pl_thresehold

    def sample(self, g: Graph, sample_ratio: float, p: torch.FloatTensor=None) -> Tuple[torch.Tensor, torch.Tensor]:
        index_l, index_r = get_sampling_pool(g, self.mode, self.pl_thresehold, p=p)
        
        if self._cache is None:
            max_sample_size = min(len(index_l), len(index_r))
            sample_size = max_sample_size
            if sample_ratio > 0:
                sample_size = int(max_sample_size * sample_ratio)

            perm_index = torch.randperm(max_sample_size)[:sample_size]
            self._cache = perm_index
        else:
            perm_index = self._cache
            sample_size = len(perm_index)

        return index_l[perm_index], index_r[perm_index]