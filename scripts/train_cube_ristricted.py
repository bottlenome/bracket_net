from __future__ import annotations

import hydra
import pytorch_lightning as pl
import torch

from bracket_net.data.cube import create_dataloader

class TestEveryNSteps(pl.Callback):
    def __init__(self, test_dataloader, test_every_n_steps: int = 1000):
        super().__init__()
        self.test_every_n_steps = test_every_n_steps
        self.test_dataloader = test_dataloader

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.test_every_n_steps == 0:
            current_logger = trainer.logger
            trainer.logger = None
            # trainer.test(pl_module, dataloaders=self.test_dataloader)
            trainer.logger = current_logger

@hydra.main(config_path="config", config_name="train_cube_restricted")
def main(config):
    train_loader, val_loader,test_loader = create_dataloader(
                                                 config.data.name,
                                                 config.data.val_test_rate,
                                                 config.params.batch_size,
                                                 config.data.size_max)
    test_callback = TestEveryNSteps(test_loader, test_every_n_steps=1000)

    if config.data.name == "StateDistanceLoader":
        from bracket_net.domain.cube.linear import DistanceEstimator
        model = DistanceEstimator(config)
    elif config.data.name == "StateNextActionLoader":
        from bracket_net.domain.cube.linear import PolicyEstimator
        model = PolicyEstimator(config)
    elif config.data.name == "RewardStateActionLoader":
        pass
    elif config.data.name == "StateActionLoader":
        pass
    elif config.data.name == "RubicDFSLoader":
        pass
    else:
        raise ValueError(f"Unknown model name {config.model.name}")

    profiler = None
    wandb_logger = pl.loggers.WandbLogger(
            project=config.project, name=config.data.name, log_model=True)
    trainer = pl.Trainer(
        callbacks=[test_callback],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        log_every_n_steps=1,
        logger=wandb_logger,
        max_epochs=config.params.num_epochs,
        profiler=profiler
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
