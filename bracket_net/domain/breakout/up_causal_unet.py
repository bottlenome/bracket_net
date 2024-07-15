from ...model.up_causal_unet import StackedUnet
import torch
import torch.nn as nn
import pytorch_lightning as pl


class PositionalEmbedding(nn.Module):
    def __init__(self, max_timestamp, context_length, d_model):
        super().__init__()
        self.max_timestamp = max_timestamp
        self.context_length = context_length
        self.d_model = d_model

        self.absolute_pos_embed = torch.nn.Parameter(
            torch.zeros(self.max_timestamp + 1, self.d_model)
        )

        self.relative_pos_embed = torch.nn.Parameter(
            torch.zeros(self.context_length * 3 + 1, self.d_model)
        )

    def forward(self, x, timesteps):
        _, seq_len, _ = x.size()
        absolute_pos_embed = self.absolute_pos_embed[timesteps].view(-1, 1, self.d_model)
        relative_pos_embed = self.relative_pos_embed[:seq_len]
        return x + (absolute_pos_embed + relative_pos_embed)


class StatesEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.state_embedding = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*7*7, config.params.d_model),
            nn.Tanh()
        )
    
    def forward(self, states):
        batch_size = states.size(0)
        seq_len = states.size(1)
        states = states.view(batch_size * seq_len, *states.size()[2:])
        y = self.state_embedding(states)
        y = y.view(batch_size, seq_len, -1)
        return y

class Decision(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.state_shape = (84, 84, 4)

        d_model = config.params.d_model
        d_action = config.params.d_action
        self.rtgs_embedding = nn.Sequential(
            nn.Linear(1, d_model),
            nn.Tanh()
        )
        self.state_embedding = StatesEmbedding(config)
        self.action_embedding = nn.Sequential(
            nn.Embedding(d_action, d_model),
            nn.Tanh()
        )
        self.positional_embedding = PositionalEmbedding(config.data.max_timestamp, config.data.seq_max, d_model)
        self.dropout = nn.Dropout(config.params.dropout)

        self.decoder = StackedUnet(config.params.d_vocab,
                                   config.params.d_model,
                                   config.params.num_layers,
                                   max_len=config.data.seq_max,
                                   enable_embed=False)
        self.layer_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, config.params.d_action, bias=False)

    def forward(self, rtgs, states, actions, timesteps):
        batch_size = states.size(0)
        seq_len = states.size(1)

        rtgs = self.rtgs_embedding(rtgs)
        states = self.state_embedding(states)
        actions = self.action_embedding(actions)

        rtgs = rtgs.view(batch_size, seq_len, 1, -1)
        states = states.view(batch_size, seq_len, 1, -1)
        actions = actions.view(batch_size, seq_len, 1, -1)

        tokens = torch.cat([rtgs, states, actions], dim=2)
        tokens = tokens.view(batch_size, 3*seq_len, -1)

        x = self.positional_embedding(tokens, timesteps)
        x = self.dropout(x)
        y = self.decoder(x)
        y = self.layer_norm(y)
        logits = self.head(y)
        return logits[:, 1::3]

class DecisionLike(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = Decision(config)
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
        states, actions, rtgs, timesteps = batch
        y = self.model(rtgs, states, actions, timesteps)
        y = y[:, :actions.size(1)]
        y_hat = y.permute(0, 2, 1)
        actions = actions.squeeze(2)
        loss = self.loss_fn(y_hat, actions)
        self.log('metrics/train/loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        states, actions, rtgs, timesteps = batch
        y = self.model(rtgs, states, actions, timesteps)
        y = y[:, :actions.size(1)]
        y_hat = y.permute(0, 2, 1)
        actions = actions.squeeze(2)
        loss = self.loss_fn(y_hat, actions)
        self.log('metrics/val/loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        states, actions, rtgs, timesteps = batch
        y = self.model(rtgs, states, actions, timesteps)
        y = y[:, :actions.size(1)]
        y_hat = y.permute(0, 2, 1)
        actions = actions.squeeze(2)
        loss = self.loss_fn(y_hat, actions)
        self.log('metrics/test/loss', loss, prog_bar=False)
        return loss

if __name__ == '__main__':
    class Params():
        d_vocab = 100
        d_model = 128
        d_action = 4
        num_layers = 10
        lr = 0.001
        dropout = 0.1
    class Data():
        seq_max = 16*3
        max_timestamp = 100
    class Config():
        params = Params()
        data = Data()
    
    config = Config()
    model = Decision(config)
    states = torch.randn(10, 16, 4, 84, 84)
    y = model.state_embedding(states)
    print(y.shape)
    rtgs = torch.randn(10, 16, 1)
    y = model.rtgs_embedding(rtgs)
    print(y.shape)
    actions = torch.randint(0, 4, (10, 16))
    y = model.action_embedding(actions)
    print(y.shape)
    timesteps = torch.randint(0, 100, (10, ))
    x = torch.randn(10, 16, 128)
    y = model.positional_embedding(x, timesteps)
    print(y.shape)
    y = model(rtgs, states, actions, timesteps)
    print(y.shape)
