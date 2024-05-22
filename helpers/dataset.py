from enum import Enum, auto
import torch
from torch_geometric.data import Data
from torch_geometric.utils import erdos_renyi_graph, stochastic_blockmodel_graph, barabasi_albert_graph
import math
from torch_geometric.transforms import BaseTransform
from typing import Optional
import os.path as osp
import pickle

from helpers.utils import set_seed
from helpers.constants import ROOT_DIR

ER_EDGE_PROB = 0.1

SBM_NUM_BLOCKS = 10
SBM_EDGE_PROBS = 0.6 * torch.eye(n=SBM_NUM_BLOCKS) + 0.1 * torch.ones(size=(SBM_NUM_BLOCKS, SBM_NUM_BLOCKS))
BA_NUM_EDGES = 5


class DataSetFamily(Enum):
    distribution = auto()
    tiger = auto()

    @staticmethod
    def from_string(s: str):
        try:
            return DataSetFamily[s]
        except KeyError:
            raise ValueError()


class DataSet(Enum):
    ER = auto()
    LogER = auto()
    InverseER = auto()
    SBM = auto()
    BA = auto()

    # real-world dataset
    Tiger1k = auto()
    Tiger5k = auto()
    Tiger10k = auto()
    Tiger25k = auto()
    Tiger90k = auto()

    @staticmethod
    def from_string(s: str):
        try:
            return DataSet[s]
        except KeyError:
            raise ValueError()

    def get_family(self) -> DataSetFamily:
        if self in [DataSet.Tiger1k, DataSet.Tiger5k, DataSet.Tiger10k, DataSet.Tiger25k, DataSet.Tiger90k]:
            return DataSetFamily.tiger
        elif self in [DataSet.ER, DataSet.LogER, DataSet.InverseER, DataSet.SBM, DataSet.BA]:
            return DataSetFamily.distribution
        else:
            raise ValueError(f'DataSet {self.name} not supported in dataloader')

    def load(self, num_nodes: int, in_dim: int, seed: int, pos_enc_transform: Optional[BaseTransform]) -> Data:
        set_seed(seed=seed)

        if self.get_family() is DataSetFamily.distribution:
            x = torch.rand(size=(num_nodes, in_dim))
            if self is DataSet.ER:
                edge_index = erdos_renyi_graph(num_nodes, ER_EDGE_PROB)
            elif self is DataSet.LogER:
                edge_index = erdos_renyi_graph(num_nodes, math.log(num_nodes)/num_nodes)
            elif self is DataSet.InverseER:
                edge_index = erdos_renyi_graph(num_nodes, 1/(50 * num_nodes))
            elif self is DataSet.SBM:
                assert num_nodes % SBM_NUM_BLOCKS == 0
                block_size = int(num_nodes / SBM_NUM_BLOCKS)
                edge_index = stochastic_blockmodel_graph(block_sizes=torch.tensor([block_size] * SBM_NUM_BLOCKS),
                                                         edge_probs=SBM_EDGE_PROBS)
            elif self is DataSet.BA:
                edge_index = barabasi_albert_graph(num_nodes=num_nodes, num_edges=BA_NUM_EDGES)
            else:
                raise ValueError(f'DataSet {self.name} not supported in dataloader')
        elif self.get_family() is DataSetFamily.tiger:
            tiger_num = self.name.split('Tiger')[1]
            file_path = osp.join(ROOT_DIR, 'tiger', tiger_num[:-1], f'Large_Tiger_Alaska_{tiger_num}.pkl')
            with open(file_path, "rb") as f:
                data = pickle.load(f)[0]
        else:
            raise ValueError(f'DataSet {self.name} not supported in dataloader')

        if self.get_family() is DataSetFamily.tiger:
            data.x = 30 * torch.rand(size=(data.x.shape[0], in_dim))
        else:
            data = Data(num_nodes=num_nodes, x=x, edge_index=edge_index)
        if pos_enc_transform is not None:
            data = pos_enc_transform(data=data)
            data.x = torch.cat((data.x, data.random_walk_pe), dim=1)
            delattr(data, 'random_walk_pe')
        return data
