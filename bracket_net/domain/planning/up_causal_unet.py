import pytorch_lightning as L

import torch
import torch.nn as nn
import torch.nn.functional as F
from ...model import up_causal_unet
from .util import CommonModule


class StackedUnet(nn.Module):
    def __init__(self, config, max_len=4100):
        super().__init__()
        self.embed = nn.Embedding(config.gpt.d_vocab + 1, config.gpt.d_model)
        self.max_len = max_len
        self.unets = nn.ModuleList()
        for i in range(config.gpt.num_layers):
            self.unets.add_module(
                f'up_causal_unet_{i}',
                up_causal_unet.UpCasualUnet(config.gpt.d_model, max_len))
        self.unembed = torch.nn.Linear(config.gpt.d_model, config.gpt.d_vocab + 1)

    
    def forward(self, x):
        len = x.size(1)
        x = self.embed(x)
        # [batch, seq, dim] -> [batch, dim, seq]
        x = x.permute(0, 2, 1)
        for net in self.unets:
            if len < self.max_len:
                x = net(F.pad(x[:, :, :len], (0, self.max_len - len)))
            else:
                x = net(x[:, :, :self.max_len])
        x = x[:, :, :self.max_len]
        # [batch, dim, seq] -> [batch, seq, dim]
        x = x.permute(0, 2, 1)
        x = self.unembed(x)
        # [batch, seq, dim] -> [batch, dim, seq]
        x = x.permute(0, 2, 1)
        return x


class Naive(CommonModule):
    def __init__(self, config, max_len=4100):
        super().__init__(config)
        self.model = StackedUnet(config, max_len=max_len)
        self.max_len = max_len


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
    class Config:
        d_vocab = 6
        d_model = 128
        num_layers = 4
    config = Config()
    max_len=4100
    model = Naive(config, max_len=max_len)
    x = torch.randint(0, 6, (100, max_len))
    y = model(x)
    print(y.shape)