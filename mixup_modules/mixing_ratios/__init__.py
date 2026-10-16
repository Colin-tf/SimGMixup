from .mixing_ratio import MixingRatio
from .random_mixing_ratio import RandomMixingRatio
# from .context_mixing_ratio import ContextMixingRatio
from .static_mixing_ratio import StaticMixingRatio
# from .saliency_mixing_ratio import SaliencyMixingRatio

__all__ = [
    'MixingRatio',
    'RandomMixingRatio',
    # 'ContextMixingRatio',
    'StaticMixingRatio'
    # 'SaliencyMixingRatio'
]

classes = __all__
