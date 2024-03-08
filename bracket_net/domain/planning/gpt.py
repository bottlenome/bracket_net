from ...model import gpt, encoder

import pytorch_lightning as L
import torch
import torch.nn as nn
from .util import CommonModule


class Naive(CommonModule):
    def __init__(self, config):
        super().__init__(config)
        self.d_vocab = config.gpt.d_vocab
        self.d_model = config.gpt.d_model
        self.model = gpt.GPT(self.d_vocab,
                             encoder.PostionalEncodingFactory(
                                 "1d", max_len=32*32*4+4+1),
                             d_model=self.d_model,
                             nhead=config.gpt.nhead,
                             num_layers=config.gpt.num_layers,
                             dropout=config.gpt.dropout)

    def forward(self, map_designs, start_maps, goal_maps, out_trajs):
        # batch, 1, 32, 32 -> batch, 32 * 32
        start_maps = start_maps.view(start_maps.size(0), -1)
        goal_maps = goal_maps.view(goal_maps.size(0), -1)
        map_designs = map_designs.view(map_designs.size(0), -1)
        # concat problem_start, start_maps, goal_maps, map_designs,
        #        estimate_start, out_trajs, estimate_end
        src = torch.cat([self.problem_start, start_maps, goal_maps, map_designs,
                        self.estimate_start,
                        out_trajs.view(out_trajs.size(0), -1),
                        self.estimate_end], dim=1)
        # float to int
        src = src.to(torch.int64)
        # batch, seq -> seq, batch
        src = src.permute(1, 0)
        # seq, batch -> seq, batch, d_vocab
        out = self.model(src)
        # seq, batch, d_vocab -> batch, d_vocab, seq
        out = out.permute(1, 2, 0)
        return out
 
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
                             nhead=config.gpt.nhead,
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