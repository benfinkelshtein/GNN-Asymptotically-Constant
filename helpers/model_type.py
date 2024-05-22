from enum import Enum, auto
from torch_geometric.nn.conv import GATConv, GPSConv, GCNConv
from torch.nn import Module

from helpers.conv import GNNConv


class ModelType(Enum):
    """
        an object for the different core
    """
    MEAN_GNN = auto()
    GAT = auto()
    GPS = auto()
    GCN = auto()

    @staticmethod
    def from_string(s: str):
        try:
            return ModelType[s]
        except KeyError:
            raise ValueError()

    def load_component(self, in_channels: int, out_channels: int) -> Module:
        if self is ModelType.MEAN_GNN:
            return GNNConv(in_channels=in_channels, out_channels=out_channels, aggr='mean')
        elif self is ModelType.GAT:
            return GATConv(in_channels=in_channels, out_channels=out_channels)
        elif self is ModelType.GPS:
            mean_gnn = GNNConv(in_channels=in_channels, out_channels=in_channels, aggr='mean')
            return GPSConv(channels=in_channels, conv=mean_gnn, norm=None)
        elif self is ModelType.GCN:
            return GCNConv(in_channels=in_channels, out_channels=out_channels)
        else:
            raise ValueError(f'model {self.name} not supported')

    def encoder_decoder(self) -> bool:
        return self is ModelType.GPS
