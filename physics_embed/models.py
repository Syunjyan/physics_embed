from collections import OrderedDict
from typing import Iterable

import torch
from torch import nn


class MLP(nn.Module):
    """Fully connected network used by all PINN baselines."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: Iterable[int] = (64, 64, 64, 64),
        activation=torch.tanh,
    ) -> None:
        super().__init__()
        widths = [input_dim, *hidden_layers, output_dim]
        layers = OrderedDict()
        for idx, (in_features, out_features) in enumerate(zip(widths[:-1], widths[1:])):
            layers[f"linear_{idx}"] = nn.Linear(in_features, out_features)

        self.layers = nn.ModuleDict(layers)
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        layer_count = len(self.layers)
        x = inputs
        for idx, layer in enumerate(self.layers.values()):
            x = layer(x)
            if idx < layer_count - 1:
                x = self.activation(x)
        return x

