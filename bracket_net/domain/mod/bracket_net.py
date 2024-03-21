from ...model import bracket_net, encoder

import pytorch_lightning as pl
import torch
from torch import nn


class Naive(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        d_vocab = config.params.p
        d_model = config.params.d_model
        pos_encoder = encoder.PostionalEncodingFactory(
                "1d", d_model=d_model, max_len=10)
        embed = None
        self.model = bracket_net.BracketNet(
                d_vocab=d_vocab, pos_encoder=pos_encoder,
                d_model=d_model, n_head=1, num_layers=1,
                dropout=0.0, embed=embed, mode=config.model.mode)
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = config.params.lr

    def forward(self, x):
        # batch, seq -> seq, batch
        x = x.permute(1, 0)
        y = self.model(x)
        # seq, batch, d_model -> batch, d_model, seq
        y = y.permute(1, 2, 0)
        return y

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('metrics/train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('metrics/val_loss', loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('metrics/test_loss', loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=1.0,
                betas=(0.9, 0.98))
        scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=1000, gamma=0.1)
        return optimizer
