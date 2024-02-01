import torch
from torch import Tensor
from enum import Enum, auto
from torch.nn import ModuleList
from typing import NamedTuple

from helpers.model_type import ModelType


class Pool(Enum):
    """
        an object for the different activation types
    """
    MEAN = auto()
    SUM = auto()

    @staticmethod
    def from_string(s: str):
        try:
            return Pool[s]
        except KeyError:
            raise ValueError()

    def forward(self, x: Tensor) -> Tensor:
        if self is Pool.MEAN:
            return torch.mean(x, dim=0)
        elif self is Pool.SUM:
            return torch.sum(x, dim=0)
        else:
            raise ValueError(f'Pool {self.name} not supported')


class ModelArgs(NamedTuple):
    model_type: ModelType
    num_layers: int
    in_dim: int
    hidden_dim: int
    out_dim: int
    pool: Pool

    def load_net(self) -> ModuleList:
        if self.model_type.encoder_decoder():
            dim_list = [self.hidden_dim] * (self.num_layers + 1)
        else:
            dim_list = [self.in_dim] + [self.hidden_dim] * (self.num_layers - 1) + [self.out_dim]
        component_list = [self.model_type.load_component(in_channels=in_dim_i, out_channels=out_dim_i)
                          for in_dim_i, out_dim_i in zip(dim_list[:-1], dim_list[1:])]
        return ModuleList(component_list)
