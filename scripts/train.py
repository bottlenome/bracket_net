from __future__ import annotations

import os

import hydra
import pytorch_lightning as pl
import torch
from neural_astar.utils.data import MazeDataset
import torch.utils.data as data

from neural_astar.planner import NeuralAstar
from neural_astar.utils.training import PlannerModule, set_global_seeds

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profilers import PyTorchProfiler

import pytorch_lightning as L
import torch.nn as nn
import bracket_net.domain.planning.gpt as gpt
import bracket_net.domain.planning.bracket_net as bracket_net
import bracket_net.domain.planning.sample as sample

import random
import numpy
import torch.utils.data as data


def create_dataloader(
    filename: str,
    split: str,
    batch_size: int,
    num_starts: int = 1,
    shuffle: bool = False,
    magnification: int = 1
) -> data.DataLoader:
    dataset = AugumentedMazeDataset(filename, split, num_starts, magnification)
    return data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2,
        pin_memory=True
    )


# Apply masks to the correct answers in the data set to augment the data.
class AugumentedMazeDataset(data.Dataset):
    def __init__(self, filename, split, num_starts=1, magnification=1):
        super().__init__()
        self.dataset = MazeDataset(filename, split, num_starts=num_starts)
        if split == "test":
            self.magnification = 1
        else:
            self.magnification = magnification
        self.map = {}
        self.ignore_index = 5

    def __getitem__(self, index):
        if self.magnification == 1:
            return self.dataset[index]
        else:
            dataset = self.dataset[index % len(self.dataset)]
            map_design = dataset[0]
            start_map = dataset[1]
            goal_map = dataset[2]
            opt_traj = dataset[3]
            if index < len(self.dataset):
                return map_design, start_map, goal_map, opt_traj
            else:
                if index not in self.map:
                    self.map[index] = random.randint(1, 1024)
                length = self.map[index]
                masked_opt_traj = opt_traj[:].reshape(1, -1)
                masked_opt_traj[0, length:] = self.ignore_index
                masked_opt_traj = masked_opt_traj.reshape(1, 32, 32)
                return map_design, start_map, goal_map, masked_opt_traj

    def __len__(self):
        return len(self.dataset) * self.magnification


@hydra.main(config_path="config", config_name="train")
def main(config):

    # dataloaders
    set_global_seeds(config.seed)
    train_loader = create_dataloader(
        config.dataset + ".npz", "train",
        config.params.batch_size, shuffle=True,
        magnification=config.data.magnification
    )
    val_loader = create_dataloader(
        config.dataset + ".npz", "valid",
        config.params.batch_size, shuffle=False,
        magnification=config.data.magnification
    )
    test_loader = create_dataloader(
        config.dataset + ".npz", "test",
        config.params.batch_size, shuffle=False
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="metrics/h_mean", save_weights_only=True, mode="max"
    )

    if config.model.name == "neural_astar":
        neural_astar = NeuralAstar(
            encoder_input=config.encoder.input,
            encoder_arch=config.encoder.arch,
            encoder_depth=config.encoder.depth,
            learn_obstacles=False,
            Tmax=config.Tmax,
        )
        module = PlannerModule(neural_astar, config)
        name = f"{config.model.name}-{config.encoder.arch}"
        name += f"-{config.encoder.depth}"
    elif config.model.name == "gpt-naive":
        module = gpt.Naive(config)
        name = f"{config.model.name}-{config.gpt.d_model}"
        name += f"-{config.gpt.n_head}-{config.gpt.num_layers}"
    elif config.model.name == "gpt-nnastarlike":
        module = gpt.NNAstarLike(config)
        name = f"{config.model.name}-{config.gpt.d_model}"
        name += f"-{config.gpt.n_head}-{config.gpt.num_layers}"
    elif config.model.name == "bracket-naive":
        module = bracket_net.Naive(config)
        name = f"{config.model.name}-{config.gpt.d_model}"
        name += f"-{config.gpt.n_head}-{config.gpt.num_layers}"
    elif config.model.name == "bracket-nnastarlike":
        module = bracket_net.NNAstarLike(config)
        name = f"{config.model.name}-{config.gpt.d_model}"
        name += f"-{config.gpt.n_head}-{config.gpt.num_layers}"
    elif config.model.name == "sample":
        module = sample.Sample(config)
        name = f"{config.model.name}"
    else:
        raise ValueError(f"Unknown model: {config.model.name}")
    # logdir = f"{config.logdir}/{os.path.basename(config.dataset)}"
    wandb_logger = WandbLogger(name=name,
                               project=config.project,
                               log_model=True)
    # profiler = PyTorchProfiler()
    profiler = None
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        log_every_n_steps=1,
        logger=wandb_logger,
        max_epochs=config.params.num_epochs,
        callbacks=[checkpoint_callback],
        profiler=profiler
    )
    """
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        trainer.fit(module, train_loader, val_loader)
        print(prof.key_averages().table(sort_by="self_cpu_time_total"))
        prof.export_chrome_trace("./trace.json")
    """
    trainer.fit(module, train_loader, val_loader)

    trainer.test(module, test_loader)

    wandb_logger.finalize("success")


if __name__ == "__main__":
    main()
