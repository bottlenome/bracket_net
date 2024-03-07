import pytorch_lightning as L
import torch
import torch.nn as nn
from neural_astar.planner.astar import VanillaAstar
from .util import get_p_opt

class Sample(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(32*32*3, 32*32*2),
                                   nn.ReLU(),
                                   nn.Linear(32*32*2, 32*32*2))
        self.remap = nn.Softmax(dim=-1)
        self.lr = config.params.lr
        self.vanilla_astar = VanillaAstar()

    def forward(self, map_designs, start_maps, goal_maps):
        src = torch.cat([map_designs, start_maps, goal_maps], dim=1)
        src = src.view(src.shape[0], -1)
        out = self.model(src)
        out = self.remap(out)
        out = out.view(-1, 2, 32, 32)
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
        accu = (outputs.argmax(dim=1) == out_trajs).float().mean()
        self.log("metrics/val_accu", accu)
        path = outputs.argmax(dim=1)
        path = path.view(-1, 1, path.size(1), path.size(2))
        p_opt = get_p_opt(self.vanilla_astar,
                          map_designs, start_maps, goal_maps, path)
        self.log("metrics/p_opt", accu)
        self.log("metrics/p_exp", 0)
        self.log("metrics/h_mean", 0)
        return loss