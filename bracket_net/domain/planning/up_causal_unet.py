import pytorch_lightning as L

import torch
import torch.nn as nn
import torch.nn.functional as F
from ...model import up_causal_unet
from .util import CommonModule


class Naive(CommonModule):
    def __init__(self, config, max_len=4100):
        super().__init__(config)
        self.model = up_causal_unet.StackedUnet(config.gpt.d_vocab,
                                                config.gpt.d_model,
                                                config.gpt.num_layers,
                                                max_len=max_len)
        self.max_len = max_len
        self.d_vocab = config.gpt.d_vocab + 1


    def forward(self, map_designs, start_maps, goal_maps, out_trajs):
        # batch, 1, 32, 32 -> batch, 32 * 32
        start_maps = start_maps.view(start_maps.size(0), -1)
        goal_maps = goal_maps.view(goal_maps.size(0), -1)
        map_designs = map_designs.view(map_designs.size(0), -1)
        # concat problem_start, start_maps, goal_maps, map_designs,
        #        estimate_start, out_trajs, estimate_end
        if out_trajs is not None:
            src = torch.cat([self.problem_start,
                            start_maps, goal_maps, map_designs,
                            self.estimate_start,
                            out_trajs.view(out_trajs.size(0), -1),
                            self.estimate_end], dim=1)
        else:
            src = torch.cat([self.problem_start,
                            start_maps, goal_maps, map_designs,
                            self.estimate_start], dim=1)
        # float to int
        src = src.to(torch.int64)
        # batch, seq -> batch, d_vocal, seq
        out = self.model(src)
        return out


if __name__ == '__main__':
    class GPT:
        d_vocab = 6
        d_model = 128
        num_layers = 4
    class Params:
        lr = 0.001
        batch_size = 10
        enable_entropy_loss = False
    class Model:
        type = "1d"
    class Config:
        gpt = GPT()
        params = Params()
        model = Model()
    config = Config()
    max_len=4100
    model = Naive(config, max_len=max_len)
    map_designs = torch.randint(0, 6, (10, 1, 32, 32))
    start_maps = torch.randint(0, 6, (10, 1, 32, 32))
    goal_maps = torch.randint(0, 6, (10, 1, 32, 32))
    out_trajs = torch.randint(0, 6, (10, 1, 32, 32))
    y = model(map_designs, start_maps, goal_maps, out_trajs)
    print(y.shape)