from torch import Tensor
from torch.nn import Module, Identity, Linear

from torch_geometric.typing import NoneType  # noqa
from torch_geometric.typing import Adj
import torch.nn.functional as F

from helpers.classes import ModelArgs


class Model(Module):
    def __init__(self, model_args: ModelArgs):
        """
        Create a model which represents the agent's policy.
        """
        super().__init__()
        if model_args.model_type.encoder_decoder():
            self.encoder = Linear(in_features=model_args.in_dim, out_features=model_args.hidden_dim)
            self.decoder = Linear(in_features=model_args.hidden_dim, out_features=model_args.out_dim)
        else:
            self.encoder = Identity()
            self.decoder = Identity()
        self.net = model_args.load_net()
        self.pool = model_args.pool

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        x = self.encoder(x)
        for layer in self.net[:-1]:
            x = layer(x=x, edge_index=edge_index)
            x = F.relu(x)
        x = self.net[-1](x=x, edge_index=edge_index)  # (graph_size, dim)
        x = self.pool.forward(x)
        x = self.decoder(x)
        return F.softmax(x, dim=-1)
