from ...model import gpt

import pytorch_lightning as L
import torch
import torch.nn as nn


class Naive(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        # self.map = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1)
        d_vocab = config.gpt.d_vocab
        self.model = gpt.GPT(d_vocab,
                             gpt.PostionalEncodingFactory(
                                 "1d", max_len=32*32*3),
                             d_model=config.gpt.d_vocab,
                             nhead=config.gpt.nhead,
                             num_layers=config.gpt.num_layers,
                             dropout=config.gpt.dropout)
        self.remap = nn.Softmax(dim=-1)
        self.lr = 0.001

    def forward(self, map_designs, start_maps, goal_maps):
        # batch, 3, 32, 32
        src = torch.cat([map_designs, start_maps, goal_maps], dim=1)
        # float to int
        src = src.to(torch.int64)
        # batch, 1, 32, 32
        # src = self.map(src)
        # batch, 32*32
        src = src.view(src.shape[0], -1)
        # map_seq, batch
        src = src.permute(1, 0)
        # 32*32, batch, 2
        out = self.model(src)
        # 32*32, batch, 2 -> 32*32, batch, 2
        out = self.remap(out)
        # 32*32, batch, 2 -> 32, 32, batch, 2
        out = out.view(32, 32, -1, self.d_model)
        # 32, 32, batch, 2 -> batch, 2, 32, 32
        out = out.permute(2, 3, 0, 1)
        return out

    def configure_optimizers(self):
        return torch.optim.RMSprop(self.model.parameters(),
                                   self.lr)

    def training_step(self, train_batch, batch_idx):
        map_designs, start_maps, goal_maps, out_trajs = opt_trajs = train_batch
        outputs = self.forward(map_designs, start_maps, goal_maps)
        loss = nn.CrossEntropyLoss()(outputs, out_trajs)
        self.log("metrics/train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        map_designs, start_maps, goal_maps, out_trajs = opt_trajs = val_batch
        outputs = self.forward(map_designs, start_maps, goal_maps)
        loss = nn.CrossEntropyLoss()(outputs, out_trajs)
        self.log("metrics/val_loss", loss)
        accu = (outputs.argmax(dim=1) == out_trajs).float().mean()
        self.log("metrics/val_accu", accu)
        return loss


class NNAstarLike(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        d_vocab = config.gpt.d_vocab
        d_model = config.gpt.d_model
        self.d_vocab = d_vocab
        self.encode = nn.Conv2d(in_channels=3, out_channels=d_model, kernel_size=1)
        
        self.model = gpt.GPT(d_vocab,
                             gpt.PostionalEncodingFactory(
                                 "2d", height=32, width=32),
                             d_model=d_model,
                             nhead=config.gpt.nhead,
                             num_layers=config.gpt.num_layers,
                             dropout=config.gpt.dropout,
                             embed=self.encode)
        self.remap = nn.Softmax(dim=-1)
        self.lr = config.params.lr

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

    def configure_optimizers(self):
        return torch.optim.RMSprop(self.model.parameters(),
                                   self.lr)

    def training_step(self, train_batch, batch_idx):
        map_designs, start_maps, goal_maps, out_trajs = train_batch
        outputs = self.forward(map_designs, start_maps, goal_maps)
        out_trajs = out_trajs.view(out_trajs.size(0),
                                   out_trajs.size(2),
                                   out_trajs.size(3))
        out_trajs = out_trajs.to(torch.int64)
        loss = nn.CrossEntropyLoss()(outputs, out_trajs)
        self.log("metrics/train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        map_designs, start_maps, goal_maps, out_trajs = val_batch
        outputs = self.forward(map_designs, start_maps, goal_maps)
        out_trajs = out_trajs.view(out_trajs.size(0),
                                   out_trajs.size(2),
                                   out_trajs.size(3))
        out_trajs = out_trajs.to(torch.int64)
        loss = nn.CrossEntropyLoss()(outputs, out_trajs)
        self.log("metrics/val_loss", loss)
        self.log("metrics/h_mean", 0)
        return loss


if __name__ == '__main__':
    model = Naive()
    model = NNAstarLike()