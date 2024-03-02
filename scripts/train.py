"""Training Neural A*
Author: Ryo Yonetani
Affiliation: OSX
"""
from __future__ import annotations

import os

import hydra
import pytorch_lightning as pl
import torch
from neural_astar.planner import NeuralAstar
from neural_astar.utils.data import create_dataloader
from neural_astar.utils.training import PlannerModule, set_global_seeds
from pytorch_lightning.callbacks import ModelCheckpoint

import pytorch_lightning as L
import torch.nn as nn
import bracket_net.domain.planning.gpt as gpt


class Sample(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(32*32*3, 32*32),
                                   nn.ReLU(),
                                   nn.Linear(32*32, 32*32))
        self.lr = 0.001

    def forward(self, map_designs, start_maps, goal_maps):
        src = torch.cat([map_designs, start_maps, goal_maps], dim=1)
        src = src.view(src.shape[0], -1)
        out = self.model(src)
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


@hydra.main(config_path="config", config_name="train")
def main(config):

    # dataloaders
    set_global_seeds(config.seed)
    train_loader = create_dataloader(
        config.dataset + ".npz", "train",
        config.params.batch_size, shuffle=True
    )
    val_loader = create_dataloader(
        config.dataset + ".npz", "valid",
        config.params.batch_size, shuffle=False
    )
    neural_astar = NeuralAstar(
        encoder_input=config.encoder.input,
        encoder_arch=config.encoder.arch,
        encoder_depth=config.encoder.depth,
        learn_obstacles=False,
        Tmax=config.Tmax,
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="metrics/h_mean", save_weights_only=True, mode="max"
    )

    # module = PlannerModule(neural_astar, config)
    # module = Sample()
    # module = gpt.Naive()
    module = gpt.NNAstarLike()
    logdir = f"{config.logdir}/{os.path.basename(config.dataset)}"
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        log_every_n_steps=1,
        default_root_dir=logdir,
        max_epochs=config.params.num_epochs,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(module, train_loader, val_loader)


if __name__ == "__main__":
    main()
