import torch
import torch.nn as nn
import numpy as np

def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)

class DiffusionModule(nn.Module):
    def __init__(self, n_timesteps=1000):
        super().__init__()
        self.n_timesteps = n_timesteps
        self.betas = cosine_beta_schedule(n_timesteps)
        alphas = 1 - self.betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        self.sqrt_alphas_cumprod = nn.Parameter(torch.sqrt(alphas_cumprod),
                                                requires_grad=False)
        self.sqrt_one_minus_alphas_cumprod = nn.Parameter(
            torch.sqrt(1 - alphas_cumprod),
            requires_grad=False)

    def extract(self, a, t, x_shape):
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    def add_noise(self, embed):
        batch_size = embed.shape[0]
        time_steps = torch.randint(0, self.n_timesteps, (batch_size,), device=embed.device)
        noise = torch.randn_like(embed)
        sqrt_alpha_cumprod_t = self.extract(self.sqrt_alphas_cumprod, time_steps, embed.shape)
        sqrt_one_minus_alpha_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, time_steps, embed.shape)
        return sqrt_alpha_cumprod_t * embed + sqrt_one_minus_alpha_cumprod_t * noise


class LinearDiffusion(nn.Module):
    def __init__(self, input_size=24, embed_size=32, hidden_size=128,
                 num_hidden=4, output_size=9, dropout_prob=0.1,
                 n_timesteps=1000):
        super().__init__()

        self.input_embedding = nn.Embedding(input_size, embed_size)
        self.output_embedding = nn.Embedding(output_size, hidden_size)
        self.output_size = output_size

        # 入力層の定義
        self.input_layer = nn.Linear(embed_size * input_size, hidden_size)
        self.input_batch_norm = nn.BatchNorm1d(hidden_size)

        self.num_hidden = num_hidden
        # 隠れ層の定義
        self.hidden_layers = nn.ModuleList()
        self.hidden_batch_norms = nn.ModuleList()
        for _ in range(self.num_hidden - 1):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
            self.hidden_batch_norms.append(nn.BatchNorm1d(hidden_size))

        # Dropout層の定義
        self.dropout = nn.Dropout(p=dropout_prob)

        # 活性化関数の定義
        self.relu = nn.ReLU()

        self.diffusion = DiffusionModule(n_timesteps)

    def loss(self, x, y):
        y_embed = self.output_embedding(y)
        y_noisy = self.diffusion.add_noise(y_embed)

        x = self(x)

        loss = torch.nn.functional.mse_loss(x, y_noisy)

        return loss

    def predict(self, x):
        x = self(x)
        batch_size = x.size(0)
        # make 0 to self.output_size - 1 array
        indexs = torch.arange(self.output_size, device=x.device)
        indexs = self.output_embedding(indexs)
        indexs = indexs.unsqueeze(0)
        x = x.unsqueeze(1)
        similarity = torch.nn.functional.cosine_similarity(x, indexs, dim=-1)
        return similarity.argmax(dim=-1)

    def forward(self, x):
        x = self.input_embedding(x)
        x = x.view(x.size(0), -1)
        x = self.input_layer(x)
        x = self.input_batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)

        for i in range(len(self.hidden_layers)):
            identity = x
            x = self.hidden_layers[i](x)
            x = self.hidden_batch_norms[i](x)
            x = self.relu(x)
            x = x + identity 
            x = self.relu(x)
            x = self.dropout(x)

        return x


if __name__ == '__main__':
    model = LinearDiffusion()
    batch_size = 3
    state_size = 24
    x = torch.zeros((batch_size, state_size), dtype=torch.int64)
    ret = model(x)
    print(ret.shape)
    y = model.predict(x)
    print(y.shape)
    loss = model.loss(x, y, torch.tensor([0, 1, 2]))
    print(loss.shape)