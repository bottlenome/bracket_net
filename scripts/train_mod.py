from __future__ import annotations

import hydra
import pytorch_lightning as pl
import torch

from bracket_net.data.mod import create_dataloader


@hydra.main(config_path="config", config_name="train_mod")
def main(config):
    train_loader, val_loader = create_dataloader(config.params.p,
                                                 config.params.batch_size)

    if config.model.name == "bracket-naive":
        from bracket_net.domain.mod.bracket_net import Naive
        model = Naive(config)
    else:
        raise ValueError(f"Unknown model name {config.model.name}")

    profiler = None
    wandb_logger = pl.loggers.WandbLogger(
            project=config.project, name=config.model.name, log_model=True)
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        log_every_n_steps=1,
        logger=wandb_logger,
        max_epochs=config.params.num_epochs,
        profiler=profiler
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
