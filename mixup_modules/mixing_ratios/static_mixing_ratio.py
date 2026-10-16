from mixup_modules.mixing_ratios.mixing_ratio import MixingRatio


class StaticMixingRatio(MixingRatio):
    """Fixed mixing ratio (default 0.5), used for controlled ablations."""

    def __init__(self, pair_wise: bool = False, init_lam: float = 0.5, *args) -> None:
        super().__init__(pair_wise)
        # 中文注释：静态比例直接保存 init_lam，每次调用返回同一个数值。
        self.lam = init_lam

    def compute(self, p=None, x=None, grad=None, mixed_index_container=None):
        return self.lam
