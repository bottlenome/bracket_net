import torch
import torch.nn as nn
from .util import CommonModule

class Sample(CommonModule):
    def __init__(self, config):
        super().__init__(config)
        self.model = nn.Sequential(nn.Linear(32*32*3, 32*32*2),
                                   nn.BatchNorm1d(32*32*2),
                                   nn.LeakyReLU(),
                                   nn.Linear(32*32*2, 32*32*2),
                                   nn.BatchNorm1d(32*32*2),
                                   nn.LeakyReLU(),
                                   nn.Linear(32*32*2, 32*32*2),
                                   )
        self.remap = nn.Softmax(dim=-1)

    def forward(self, map_designs, start_maps, goal_maps):
        src = torch.cat([map_designs, start_maps, goal_maps], dim=1)
        src = src.view(src.shape[0], -1)
        out = self.model(src)
        out = self.remap(out)
        out = out.view(-1, 2, 32, 32)
        return out