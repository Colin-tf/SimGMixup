from mixup_modules.mixing_ratios.mixing_ratio import *

class RandomMixingRatio(MixingRatio):
    def __init__(self, pair_wise: bool = False, init_lam: float=0.5, beta_alpha: float=0.5, reset_flag: bool = False,*args) -> None:
        super(RandomMixingRatio, self).__init__(pair_wise)
        self.beta_alpha = beta_alpha
        self.reset_flag = reset_flag

    def compute(self, p: torch.LongTensor=None, x: torch.LongTensor=None, grad: torch.LongTensor=None, mixed_index_container: MixedIndexContainer=None) -> torch.LongTensor:
        if self.reset_flag:
            lam = torch.distributions.Beta(self.beta_alpha, self.beta_alpha).sample()
            self.lam = lam
        else:
            lam = self.lam if self.lam is not None else torch.distributions.Beta(self.beta_alpha, self.beta_alpha).sample()
        
        return lam