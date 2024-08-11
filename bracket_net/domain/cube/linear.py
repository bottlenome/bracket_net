from ...model.linear import Linear
import torch
import pytorch_lightning as pl

def simulate(initial_state, model):
    from ...data.cube import face_str2int, face_int2str
    from ...data.cube_model.enums import Move
    from ...data.cube_model.coord import CoordCube
    from copy import deepcopy

    state_string = face_int2str(initial_state)
    co_cube = CoordCube.from_string(state_string)
    for _ in range(10):
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
            distance = model(state_tensor.unsqueeze(0) / 7.).item()
            if distance < min_distance:
                min_distance = distance
                min_move = m
        if min_move is not None:
            co_cube.move(min_move)
        else:
            raise ValueError("No move found")
    return 0

def simulate_policy(initial_state, model):
    from ...data.cube import face_str2int, face_int2str
    from ...data.cube_model.enums import Move
    from ...data.cube_model.coord import CoordCube

    state_string = face_int2str(initial_state)
    co_cube = CoordCube.from_string(state_string)
    for _ in range(10):
        if co_cube.is_solved():
            return 1
        state_input = torch.tensor(face_str2int(state_string), dtype=torch.float32).unsqueeze(0) / 7.
        state_input = state_input.to(next(model.parameters()).device)
        move = model(state_input).argmax()
        try:
            m = Move(move.item() - 1) # 0 invalid move is not used
        except ValueError:
            return 0
        co_cube.move(m)
        state_string = co_cube.to_string()
    return 0


class DistanceEstimator(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = Linear(num_hidden=config.params.num_layers,
                            dropout_prob=config.params.dropout)
        self.lr = config.params.lr

    def forward(self, x):
        x = x / 7.
        y = self.model(x)
        return y

    def loss_fn(self, y, x):
        return torch.nn.functional.mse_loss(y, x)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = y.unsqueeze(-1) / 10.
        loss = self.loss_fn(y_hat, y)
        self.log('metrics/train/loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = y.unsqueeze(-1) / 10.
        loss = self.loss_fn(y_hat, y)
        self.log('metrics/val/loss', loss, prog_bar=True)
        if batch_idx == 0:
            total = 0.
            try_num = 10
            for i in range(try_num):
                solved = simulate(x[i], self.model)
                total += solved
            self.log('metrics/val/solved', total / try_num, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = y.unsqueeze(-1) / 10.
        loss = self.loss_fn(y_hat, y)
        # self.log('metrics/test/loss', loss, prog_bar=False)
        return loss


class PolicyEstimator(pl.LightningModule):
    def __init__(self, config):
        from ...data.cube_model.enums import Move
        super().__init__()
        self.model = Linear(num_hidden=config.params.num_layers,
                            output_size=len(Move) + 1,
                            dropout_prob=config.params.dropout)
        self.lr = config.params.lr

    def forward(self, x):
        x = x / 7.
        y = self.model(x)
        return y

    def loss_fn(self, y, x):
        return torch.nn.functional.cross_entropy(y, x)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('metrics/train/loss', loss, prog_bar=True)
        return loss

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