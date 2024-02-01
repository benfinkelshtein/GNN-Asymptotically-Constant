from enum import Enum, auto
import torch
from torch_geometric.data import Data
from torch_geometric.utils import erdos_renyi_graph, stochastic_blockmodel_graph
import math
from torch_geometric.transforms import BaseTransform
from typing import Optional

from helpers.utils import set_seed

ER_EDGE_PROB = 0.1

SBM_NUM_BLOCKS = 10
SBM_EDGE_PROBS = 0.6 * torch.eye(n=SBM_NUM_BLOCKS) + 0.1 * torch.ones(size=(SBM_NUM_BLOCKS, SBM_NUM_BLOCKS))


class DataSet(Enum):
    ER = auto()
    LogER = auto()
    InverseER = auto()
    SBM = auto()

    @staticmethod
    def from_string(s: str):
        try:
            return DataSet[s]
        except KeyError:
            raise ValueError()

    def load(self, graph_size: int, in_dim: int, seed: int, pos_enc_transform: Optional[BaseTransform]) -> Data:
        set_seed(seed=seed)

        x = torch.rand(size=(graph_size, in_dim))
        if self is DataSet.ER:
            edge_index = erdos_renyi_graph(graph_size, ER_EDGE_PROB)
        elif self is DataSet.LogER:
            edge_index = erdos_renyi_graph(graph_size, math.log(graph_size)/graph_size)
        elif self is DataSet.InverseER:
            edge_index = erdos_renyi_graph(graph_size, 1/(50 * graph_size))
        elif self is DataSet.SBM:
            assert graph_size % SBM_NUM_BLOCKS == 0
            block_size = int(graph_size / SBM_NUM_BLOCKS)
            edge_index = stochastic_blockmodel_graph(block_sizes=torch.tensor([block_size] * SBM_NUM_BLOCKS),
                                                     edge_probs=SBM_EDGE_PROBS)
        else:
            raise ValueError(f'DataSet {self.name} not supported in dataloader')
        data = Data(graph_size=graph_size, x=x, edge_index=edge_index)
        if pos_enc_transform is not None:
            data = pos_enc_transform(data=data)
            data.x = torch.cat((data.x, data.random_walk_pe), dim=1)
            delattr(data, 'random_walk_pe')
        return data
