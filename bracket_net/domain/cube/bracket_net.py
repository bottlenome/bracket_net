from ...model import bracket_net, encoder

import pytorch_lightning as pl
import torch
from torch import nn


class Naive(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        d_vocab = config.params.d_vocab
        d_model = config.params.d_model
        pos_encoder = encoder.PostionalEncodingFactory(
                "1d", d_model=d_model, max_len=config.data.seq_max)
        n_head = config.params.n_head
        num_layers = config.params.num_layers
        dropout = config.params.dropout
        embed = None
        self.model = bracket_net.BracketNet(
                d_vocab=d_vocab, pos_encoder=pos_encoder,
                d_model=d_model, n_head=n_head, num_layers=num_layers,
                dropout=dropout, embed=embed, mode=config.model.mode,
                seq_max=config.data.seq_max)
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
        # concat src and tgt
        x = torch.cat(batch, dim=1)
        y_hat = self(x)
        # padding x by ingoring the first token
        x_pad = torch.cat([x[:, 1:], torch.zeros_like(x[:, 0:1])-100], dim=1)
        loss = self.loss_fn(y_hat, x_pad)
        self.log('metrics/train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = torch.cat(batch, dim=1)
        y_hat = self(x)
        x_pad = torch.cat([x[:, 1:], torch.zeros_like(x[:, 0:1])-100], dim=1)
        loss = self.loss_fn(y_hat, x_pad)
        self.log('metrics/val_loss', loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x = torch.cat(batch, dim=1)
        y_hat = self(x)
        x_pad = torch.cat([x[:, 1:], torch.zeros_like(x[:, 0:1])-100], dim=1)
        loss = self.loss_fn(y_hat, x_pad)
        self.log('metrics/test_loss', loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.lr)
        return optimizer
