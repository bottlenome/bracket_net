from ...model.diffusion import LinearDiffusion
import torch
import pytorch_lightning as pl
from .utils import DebugLogger


def simulate_policy(initial_state, model):
    from ...data.cube import face_str2int, face_int2str
    from ...data.cube_model.enums import Move
    from ...data.cube_model.coord import CoordCube

    p = DebugLogger.get_instance()
    state_string = face_int2str(initial_state)
    co_cube = CoordCube.from_string(state_string)
    device = next(model.parameters()).device
    for _ in range(12):
        if co_cube.is_solved():
            p.rint("solved")
            return 1
        state_input = torch.tensor(face_str2int(state_string)).unsqueeze(0)
        state_input = state_input.to(device)
        p.rint(f"  src:{state_input}")
        move = model.predict(state_input)
        try:
            m = Move(move.item())
            p.rint(f"  move:{m}")
            p.rint(co_cube)
        except ValueError:
            p.rint(f"invalid move:{move}")
            return 0
        co_cube.move(m)
        state_string = co_cube.to_string()
        p.rint(f"  dst:{state_string}")
    p.rint("reach max steps")
    return 0


class PolicyEstimator(pl.LightningModule):
    def __init__(self, config):
        from ...data.cube_model.enums import Move, Color, Facelet
        super().__init__()
        self.model = LinearDiffusion(input_size=len(Facelet),
                                     embed_size=config.params.d_model,
                                     hidden_size=config.params.d_model,
                                     num_hidden=config.params.num_layers,
                                     output_size=len(Move),
                                     dropout_prob=config.params.dropout)
        self.lr = config.params.lr
        self.loss_weight = torch.nn.Parameter(torch.tensor(
            [4.5448347952551926e-05, 7.890168849613381e-05,
             0.00010198878123406426, 0.00010530749789385004,
             0.00013090718680455556, 0.00018446781036709093,
             0.0001859427296392711, 0.0001836884643644379,
             0.0002463661000246366], dtype=torch.float32),
            requires_grad=False)

    def forward(self, x):
        y = self.model(x)
        return y

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        x, y = batch
        batch_size = x.size(0)
        timesteps = torch.randint(0, self.model.n_timesteps, (batch_size,), device=x.device)
        loss = self.model.loss(x, y, timesteps)
        self.log('metrics/train/loss', loss, prog_bar=True)
        if batch_idx % 100 == 0:
            self.log_gradients()
        return loss


    def log_gradients(self):
        p = DebugLogger.get_instance()
        for name, param in self.named_parameters():
            if param.grad is not None:
                p.rint(f'grad_{name}_mean', param.grad.mean())
                p.rint(f'grad_{name}_std', param.grad.std())


    def validation_step(self, batch, batch_idx):
        x, y = batch
        batch_size = x.size(0)
        timesteps = torch.randint(0, self.model.n_timesteps, (batch_size,), device=x.device)
        loss = self.model.loss(x, y, timesteps)
        self.log('metrics/val/loss', loss, prog_bar=True)
        if batch_idx == 0:
            total = 0.
            try_num = 10
            for i in range(try_num):
                solved = simulate_policy(x[i], self.model)
                total += solved
            self.log('metrics/val/solved', total / try_num, prog_bar=True)

        return loss


    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('metrics/test/loss', loss, prog_bar=False)
        return loss


if __name__ == '__main__':
    class Config:
        params = type("Params", (), {"lr": 1e-3, "d_model": 128, "num_layers": 6, "dropout": 0.1})
    
    config = Config()
    model = PolicyEstimator(config)