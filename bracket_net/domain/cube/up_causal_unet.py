from ...model.up_causal_unet import StackedUnet
import torch
import pytorch_lightning as pl


class Naive(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = StackedUnet(config.params.d_vocab,
                                 config.params.d_model,
                                 config.params.num_layers,
                                 max_len=config.data.seq_max)
        self.lr = config.params.lr
        self.seq_max = config.data.seq_max

    def forward(self, x):
        y = self.model(x)
        return y

    def loss_fn(self, y_hat, x_pad):
        return torch.nn.functional.cross_entropy(y_hat, x_pad)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.001)

    def training_step(self, batch, batch_idx):
        # concat src and tgt
        x = torch.cat(batch, dim=1)
        y_hat = self(x)
        # padding x by seq_max
        x_pad = torch.cat([x[:, 1:], torch.zeros_like(x[:, 0:1+y_hat.size(-1) - x.size(-1)])-100], dim=1)
        loss = self.loss_fn(y_hat, x_pad)
        self.log('metrics/train/loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = torch.cat(batch, dim=1)
        y_hat = self(x)
        x_pad = torch.cat([x[:, 1:], torch.zeros_like(x[:, 0:1+y_hat.size(-1) - x.size(-1)])-100], dim=1)
        loss = self.loss_fn(y_hat, x_pad)
        self.log('metrics/val/loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x = torch.cat(batch, dim=1)
        y_hat = self(x)
        x_pad = torch.cat([x[:, 1:], torch.zeros_like(x[:, 0:1+y_hat.size(-1) - x.size(-1)])-100], dim=1)
        loss = self.loss_fn(y_hat, x_pad)
        self.log('metrics/test/loss', loss, prog_bar=False)
        return loss