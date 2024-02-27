from ...model import gpt

import pytorch_lightning as L
import torch
import torch.nn as nn


class Naive(L.LightningModule):
    def __init__(self):
        super().__init__()
        # self.map = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1)
        self.model = gpt.GPT(10, 32*32*3)
        self.remap = nn.Sequential(
                nn.Linear(32 * 32 * 3 * 10, 32 * 32), nn.ReLU())
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
        # 32*32, batch, 10
        out = self.model(src)
        # batch, 32 * 32, 10
        out = out.permute(1, 0, 2)
        # batch, 32 * 32 * 10
        out = out.reshape(out.shape[0], -1)
        # batch, 32 * 32
        out = self.remap(out)
        out = out.view(-1, 1, 32, 32)
        return out

    def configure_optimizers(self):
        return torch.optim.RMSprop(self.model.parameters(),
                                   self.lr)

    def training_step(self, train_batch, batch_idx):
        map_designs, start_maps, goal_maps, out_trajs = opt_trajs = train_batch
        outputs = self.forward(map_designs, start_maps, goal_maps)
        loss = nn.MSELoss()(outputs, out_trajs)
        self.log("metrics/train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        map_designs, start_maps, goal_maps, out_trajs = opt_trajs = val_batch
        outputs = self.forward(map_designs, start_maps, goal_maps)
        loss = nn.MSELoss()(outputs, out_trajs)
        self.log("metrics/val_loss", loss)
        self.log("metrics/h_mean", 0)
        return loss


class NNALike(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.map = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1)
        self.model = gpt.GPT(10, 32*32*3)
        self.remap = nn.Sequential(
                nn.Linear(32 * 32 * 3 * 10, 32 * 32), nn.ReLU())
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
        # 32*32, batch, 10
        out = self.model(src)
        # batch, 32 * 32, 10
        out = out.permute(1, 0, 2)
        # batch, 32 * 32 * 10
        out = out.reshape(out.shape[0], -1)
        # batch, 32 * 32
        out = self.remap(out)
        out = out.view(-1, 1, 32, 32)
        return out

    def configure_optimizers(self):
        return torch.optim.RMSprop(self.model.parameters(),
                                   self.lr)

    def training_step(self, train_batch, batch_idx):
        map_designs, start_maps, goal_maps, out_trajs = opt_trajs = train_batch
        outputs = self.forward(map_designs, start_maps, goal_maps)
        loss = nn.MSELoss()(outputs, out_trajs)
        self.log("metrics/train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        map_designs, start_maps, goal_maps, out_trajs = opt_trajs = val_batch
        outputs = self.forward(map_designs, start_maps, goal_maps)
        loss = nn.MSELoss()(outputs, out_trajs)
        self.log("metrics/val_loss", loss)
        self.log("metrics/h_mean", 0)
        return loss


if __name__ == '__main__':
    model = Naive()
