from ...model.linear import Linear
import torch
import pytorch_lightning as pl
from .utils import DebugLogger


def simulate(initial_state, model):
    from ...data.cube import face_str2int, face_int2str
    from ...data.cube_model.enums import Move
    from ...data.cube_model.coord import CoordCube
    from copy import deepcopy

    p = DebugLogger.get_instance()

    state_string = face_int2str(initial_state)
    co_cube = CoordCube.from_string(state_string)
    for _ in range(12):
        if co_cube.is_solved():
            return 1
        min_distance = 1000
        min_move = None
        for m in Move:
            after_state = deepcopy(co_cube)
            after_state.move(m)
            state_str = after_state.to_string()
            state_tensor = torch.tensor(face_str2int(state_str), dtype=torch.float32)
            state_tensor = state_tensor.to(next(model.parameters()).device)
            distance = model(state_tensor.unsqueeze(0) / 7.)
            p.rint(f" move:{m} distance:{torch.nn.functional.softmax(distance, dim=-1)}")
            distance = distance.argmax().item()
            if distance < min_distance:
                min_distance = distance
                min_move = m
        if min_move is not None:
            co_cube.move(min_move)
            p.rint(f"move:{min_move} distance:{min_distance}")
            p.rint(co_cube)
        else:
            raise ValueError("No move found")
    p.rint("reach max steps")
    return 0

def simulate_policy(initial_state, model):
    from ...data.cube import face_str2int, face_int2str
    from ...data.cube_model.enums import Move
    from ...data.cube_model.coord import CoordCube

    p = DebugLogger.get_instance()
    state_string = face_int2str(initial_state)
    co_cube = CoordCube.from_string(state_string)
    for _ in range(12):
        if co_cube.is_solved():
            p.rint("solved")
            return 1
        # state_input = torch.tensor(face_str2int(state_string), dtype=torch.float32).unsqueeze(0) / 7.
        state_input = torch.tensor(face_str2int(state_string)).unsqueeze(0)
        state_input = state_input.to(next(model.parameters()).device)
        p.rint(f"  src:{state_input}")
        out = model(state_input)
        p.rint(f"  out:{out}")
        move = out.argmax(dim=-1)
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


class DistanceEstimator(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = Linear(embed_size=config.params.d_model,
                            hidden_size=config.params.d_model,
                            num_hidden=config.params.num_layers,
                            dropout_prob=config.params.dropout,
                            output_size=12,
                            enable_positional_embedding=False)
        self.lr = config.params.lr
        self.loss_weight = torch.nn.Parameter(torch.tensor(
            [1, 0.2, 0.07142857142857142, 0.022222222222222223,
             0.008695652173913044, 0.002840909090909091,
             0.000723589001447178, 0.00018162005085361425,
             5.1607575992155646e-05, 2.459298608036988e-05,
             6.988120195667365e-05, 0.006622516556291391], dtype=torch.float32),
             requires_grad=False)

    def forward(self, x):
        x = x / 7.
        y = self.model(x)
        return y

    def loss_fn(self, y, x):
        return torch.nn.functional.cross_entropy(y, x, weight=self.loss_weight)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        T = 2.0
        loss = self.loss_fn(y_hat / 2, y)
        self.log('metrics/train/loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        p = DebugLogger.get_instance()
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('metrics/val/loss', loss, prog_bar=True)
        if batch_idx == 0:
            total = 0.
            try_num = 10
            for i in range(try_num):
                p.rint(f"#try to solve distance:{int(y[i].item())} problem")
                solved = simulate(x[i], self.model)
                total += solved
            self.log('metrics/val/solved', total / try_num, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        # self.log('metrics/test/loss', loss, prog_bar=False)
        return loss


class PolicyEstimator(pl.LightningModule):
    def __init__(self, config):
        from ...data.cube_model.enums import Move, Color
        super().__init__()
        self.model = Linear(embed_size=config.params.d_model,
                            hidden_size=config.params.d_model,
                            num_hidden=config.params.num_layers,
                            output_size=len(Move),
                            dropout_prob=config.params.dropout,
                            enable_positional_embedding=False,
                            input_num=len(Color) + 1)
        self.lr = config.params.lr
        self.loss_weight = torch.nn.Parameter(torch.tensor(
            [4.5448347952551926e-05, 7.890168849613381e-05,
             0.00010198878123406426, 0.00010530749789385004,
             0.00013090718680455556, 0.00018446781036709093,
             0.0001859427296392711, 0.0001836884643644379,
             0.0002463661000246366], dtype=torch.float32),
            requires_grad=False)

    def forward(self, x):
        # x = x / 7.
        # x -= 1
        y = self.model(x)
        return y

    def loss_fn(self, y, x):
        return torch.nn.functional.cross_entropy(y, x, weight=self.loss_weight)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
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
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
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
    from ...data.cube import face_str2int

    class Mock:
        def __init__(self):
            self.count = 0

        def __call__(self, x):
            if self.count == 0:
                # Move.U1 is the best move
                self.count += 1
                return torch.tensor([0.1])
            else:
                return torch.tensor([1.0])

        def parameters(self):
            yield torch.tensor([1.0])

    model = Mock()
    state = 'UUUURRRRFFFFDDDDLLLLBBBB'
    state_tensor = torch.tensor(face_str2int(state), dtype=torch.float32)
    print(simulate(state_tensor, model))

    from ...data.cube_model.enums import Move
    from ...data.cube_model.coord import CoordCube

    cc = CoordCube()
    cc.move(Move.U3)
    state_tensor = torch.tensor(face_str2int(cc.to_string()), dtype=torch.float32)
    print(simulate(state_tensor, model))

    class MockPolicy:
        def __init__(self):
            self.count = 0

        def __call__(self, x):
            if self.count == 0:
                # make one hot array for Move.U1
                vector = torch.zeros(1, len(Move))
                vector[0][Move.U1.value + 1] = 1
                self.count += 1
                return vector
            else:
                vector = torch.zeros(1, len(Move))
                vector[0][Move.U2.value + 1] = 1
                return vector

        def parameters(self):
            yield torch.tensor([1.0])
    
    model = MockPolicy()
    cc = CoordCube()
    cc.move(Move.U3)
    state_tensor = torch.tensor(face_str2int(cc.to_string()), dtype=torch.float32)
    print(simulate_policy(state_tensor, model))