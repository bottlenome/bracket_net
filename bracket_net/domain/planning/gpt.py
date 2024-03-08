from ...model import gpt, encoder

import pytorch_lightning as L
import torch
import torch.nn as nn
from .util import CommonModule, NaiveBase


class Naive(NaiveBase):
    def __init__(self, config):
        super().__init__(config, gpt.GPT)

 
class NNAstarLike(CommonModule):
    def __init__(self, config):
        super().__init__(config)
        d_vocab = config.gpt.d_vocab
        d_model = config.gpt.d_model
        self.d_vocab = d_vocab
        self.encode = nn.Conv2d(in_channels=3,
                                out_channels=d_model,
                                kernel_size=1)

        self.model = gpt.GPT(d_vocab,
                             encoder.PostionalEncodingFactory(
                                 "2d", height=32, width=32),
                             d_model=d_model,
                             n_head=config.gpt.n_head,
                             num_layers=config.gpt.num_layers,
                             dropout=config.gpt.dropout,
                             embed=self.encode)
        self.remap = nn.Softmax(dim=-1)

    def forward(self, map_designs, start_maps, goal_maps):
        src = torch.cat([map_designs, start_maps, goal_maps], dim=1)
        # batch, 3, 32, 32 -> 32*32, batch, 2
        out = self.model(src)
        # 32*32, batch, 2 -> 32*32, batch, 2
        out = self.remap(out)
        # 32*32, batch, 2 -> batch, 2, 32*32
        out = out.permute(1, 2, 0)
        out = out.view(-1, self.d_vocab, 32, 32)
        return out


if __name__ == '__main__':
    model = Naive()
    model = NNAstarLike()