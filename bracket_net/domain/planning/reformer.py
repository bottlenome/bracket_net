import torch
import torch.nn as nn
from reformer_pytorch import Reformer
from .util import NaiveBase



class ReformerInterface(nn.Module):
    def __init__(self, d_vocab, pos_encoder, d_model=128, n_head=4,
                 num_layers=6, dropout=0.0,
                 embed=None):
        super().__init__()
        if embed is None:
            # int(d_vocab+1) to float(d_model)
            self.embed = torch.nn.Embedding(d_vocab + 1, d_model)
        else:
            self.embed = embed
        self.pos_encoder = pos_encoder(d_model, dropout)
        self.model = Reformer(
            dim=d_model,
            depth=num_layers,
            heads=n_head,
            lsh_dropout=dropout,
            causal=True
        )
        self.unembed = torch.nn.Linear(d_model, d_vocab)

    def forward(self, src):
        src = self.embed(src)
        src = self.pos_encoder(src)
        len_seq = src.shape[1]
        multification = int(len_seq / (self.model.bucket_size * 2))
        padding = 2 * self.model.bucket_size * (multification + 1) - len_seq
        src = torch.cat([src, torch.zeros(src.shape[0], padding, src.shape[2]).to(src.device)], dim=1)
        out = self.model(src)
        out = out[:, :len_seq, :]
        out = self.unembed(out)
        return out


class Naive(NaiveBase):
    def __init__(self, config):
        super().__init__(config, ReformerInterface, seq_batch_convert=False)


if __name__ == '__main__':
    class Params:
        def __init__(self):
            self.lr = 0.1
            self.batch_size = 10
            self.enable_entropy_loss = False

    class Model:
        def __init__(self):
            self.type = "1d"

    class GPTParam:
        def __init__(self):
            self.d_vocab = 5
            self.d_model = 128
            self.dropout = 0.1
            self.n_head = 4
            self.num_layers = 4

    class Config:
        def __init__(self):
            self.params = Params()
            self.model = Model()
            self.gpt = GPTParam()
            self.embedding = None

    model = Naive(Config())
    map_design = torch.randint(0, 5, (10, 32, 32))
    start_map = torch.randint(0, 5, (10, 32, 32))
    goal_map = torch.randint(0, 5, (10, 32, 32))
    opt_traj = torch.randint(0, 5, (10, 32, 32))
    out = model(map_design, start_map, goal_map, opt_traj)
    print(out.shape)