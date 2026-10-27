"""两层 MLP：2 → 8 → 1，共 33 个参数。"""
from torch import nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 8)
        self.activation = nn.ReLU()
        self.output = nn.Linear(8, 1)

    def forward(self, x):
        return self.output(self.activation(self.hidden(x)))
