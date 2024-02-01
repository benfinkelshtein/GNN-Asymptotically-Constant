from enum import Enum, auto
from torch_geometric.nn.conv import GATConv, GPSConv
from torch.nn import Module

from helpers.conv import GNNConv


class ModelType(Enum):
    """
        an object for the different core
    """
    SUM_GNN = auto()
    MEAN_GNN = auto()
    MAX_GNN = auto()

    GAT = auto()
    TRANSFORMER = auto()
    GPS = auto()

    @staticmethod
    def from_string(s: str):
        try:
            return ModelType[s]
        except KeyError:
            raise ValueError()

    def load_component(self, in_channels: int, out_channels: int) -> Module:
        if self is ModelType.SUM_GNN:
            return GNNConv(in_channels=in_channels, out_channels=out_channels, aggr='sum')
        elif self is ModelType.MEAN_GNN:
            return GNNConv(in_channels=in_channels, out_channels=out_channels, aggr='mean')
        elif self is ModelType.MAX_GNN:
            return GNNConv(in_channels=in_channels, out_channels=out_channels, aggr='max')
        elif self is ModelType.GAT:
            return GATConv(in_channels=in_channels, out_channels=out_channels)
        elif self is ModelType.TRANSFORMER:
            return GPSConv(channels=in_channels, conv=None, norm=None)
        elif self is ModelType.GPS:
            mean_gnn = GNNConv(in_channels=in_channels, out_channels=in_channels, aggr='mean')
            return GPSConv(channels=in_channels, conv=mean_gnn, norm=None)
        else:
            raise ValueError(f'model {self.name} not supported')

    def encoder_decoder(self) -> bool:
        return self in [ModelType.TRANSFORMER, ModelType.GPS]
