import torch
import torch.nn as nn
from typing import List


class FeedForwardNet(nn.Module):
    """
    Module for the feedforward neural network (FNN)
    """

    def __init__(self, in_dim: int, out_dim: int, net_structure: List[int]) -> None:
        """
        :param in_dim: input dimension for the FNN
        :param out_dim: output dimension for the FNN
        :param net_structure: layer structure of the FNN
        """
        super(FeedForwardNet, self).__init__()
        structure = []
        self._in_dim = in_dim
        self._out_dim = out_dim
        for layers in net_structure:
            structure += [
                nn.Linear(in_features=self._in_dim, out_features=layers, bias=True),
                nn.ReLU(),
            ]
            self._in_dim = layers
        structure += [
            nn.Linear(in_features=self._in_dim, out_features=self._out_dim, bias=True)
        ]
        self.network = nn.Sequential(*structure)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        pass the input through the network and get the output
        :param x: input of the FNN
        :return: output of the FNN
        """
        output = self.network(x)
        return output

    def save(self, model_path: str) -> None:
        """
        save the model
        :param model_path: path for the saved model
        """
        torch.save(self.state_dict(), model_path)
