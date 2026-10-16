import torch
from abc import ABC, abstractmethod
from typing import List, Tuple
from graph import Graph

import torch
from abc import ABC, abstractmethod
from typing import Tuple

class Sampler(ABC):
    def __init__(self, *args) -> None:
        super().__init__()
    
    @abstractmethod
    def sample(self, *args) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Sampler.sample should be implemented.")

    def __call__(self, *args: torch.Any) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sample(*args)
