from ...model import bracket_net, encoder

from .util import NaiveBase, CommonModule
import torch
import torch.nn as nn


class Naive(NaiveBase):
    def __init__(self, config):
        super().__init__(config, bracket_net.BracketNet)


class NNAstarLike(CommonModule):
    def __init__(self, config):
        super().__init__(config)
        d_vocab = config.gpt.d_vocab
        d_model = config.gpt.d_model
        self.d_vocab = d_vocab
        self.conition_encode = nn.Conv2d(in_channels=3,
                                         out_channels=1,
                                         kernel_size=1)
        self.seq_embedding = nn.Conv2d(in_channels=1,
                                       out_channels=128,
                                       kernel_size=1)

        self.model = bracket_net.BracketNet(d_vocab,
                                            encoder.PostionalEncodingFactory(
                                                "none", height=32, width=32),
                                            d_model=d_model,
                                            nhead=config.gpt.nhead,
                                            num_layers=config.gpt.num_layers,
                                            dropout=config.gpt.dropout,
                                            embed=self.seq_embedding)
        self.remap = nn.Softmax(dim=-1)

    def forward(self, map_designs, start_maps, goal_maps, out_trajs):
        # batch, 1, 32, 32 -> batch, 3, 32, 32
        src = torch.cat([map_designs, start_maps, goal_maps], dim=1)
        # batch, 3, 32, 32 -> batch, 1, 32, 32
        encoded = self.conition_encode(src)
        # batch, 1, 32, 32 -> batch, 32 * 32
        encoded = encoded.view(encoded.size(0), -1)
        # problem_start: batch, 1
        # concat problem_start, encoded,
        #        estimate_start, out_trajs, estimate_end
        src1 = torch.cat([self.problem_start, encoded,
                          self.estimate_start,
                          out_trajs.view(out_trajs.size(0), -1),
                          self.estimate_end], dim=1)
        # batch, seq -> seq, batch
        src1 = src1.permute(1, 0)
        # seq, batch -> seq, batch, d_vocab
        out = self.model(src1)
        # seq, batch, d_vocab -> batch, d_vocab, seq
        out = out.permute(1, 2, 0)
        return out