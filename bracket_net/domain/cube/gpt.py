from ...model.gpt import GPTBlock
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


def get_single_rtgs(reward, device):
    return torch.zeros(1, 1, 1, dtype=torch.int64, device=device) + reward

def get_single_state(state_str, device):
    from ...data.cube import face_str2int
    state_int = face_str2int(state_str)
    state_tensor = torch.tensor(state_int, dtype=torch.int64, device=device)
    return state_tensor.unsqueeze(0).unsqueeze(0)

def simulate(initial_state, model):
    from ...data.cube import face_int2str
    from ...data.cube_model.enums import Move
    from ...data.cube_model.coord import CoordCube

    device = next(model.parameters()).device

    reward = 110
    move_reward = -1
    state_string = face_int2str(initial_state)
    co_cube = CoordCube.from_string(state_string)
    for i in range(10):
        if co_cube.is_solved():
            return 1
        if i == 0:
            rtgs = get_single_rtgs(reward, device)
            state = get_single_state(co_cube.to_string(), device)
            action = None
            timestep = torch.zeros(1, 1, 1, dtype=torch.int64, device=device)
        else:
            rtgs = torch.cat([rtgs, get_single_rtgs(reward, device)], dim=1)
            state = torch.cat(
                [state, get_single_state(co_cube.to_string(), device)], dim=1)
            # action and timestep is no need to update
        y = model.estimate(rtgs, state, action, timestep)
        action = y[:, 1::3, :].argmax(dim=-1)
        try:
            co_cube.move(Move(action[:, -1].item() - 1))
        except Exception as e:
            return 0
        reward += move_reward

    return 0


class DecisionFormer(pl.LightningModule):
    def __init__(self, config):
        from ...data.cube_model.enums import Move, Color, Facelet
        super().__init__()
        face_color_size = len(Color) + 1
        face_let_size = len(Facelet)
        move_size = len(Move) + 1
        self.state_encoder = nn.Sequential(
            nn.Embedding(face_color_size, config.params.d_model),
            nn.Flatten(start_dim=2), # batch, seq, state, d_model -> batch, seq, state * d_model
            nn.ReLU(),
            nn.Linear(face_let_size * config.params.d_model, config.params.d_model),
            nn.ReLU(),
            nn.Linear(config.params.d_model, config.params.d_model),
            nn.Tanh())
        self.action_encoder = nn.Sequential(
            nn.Embedding(move_size, config.params.d_model),
            nn.Flatten(start_dim=2), # batch, seq, 1, d_model -> batch, seq, d_model
            nn.ReLU(),
            nn.Linear(config.params.d_model, config.params.d_model),
            nn.ReLU(),
            nn.Linear(config.params.d_model, config.params.d_model),
            nn.Tanh())
        rtgs_size = 1
        self.rtgs_encoder = nn.Sequential(
            nn.Linear(rtgs_size, config.params.d_model),
            nn.Tanh()
        )
        self.position_embedding = nn.Parameter(torch.zeros(1, (config.params.max_len * 3) + 1, config.params.d_model))
        max_timestamp = 10
        self.global_position_embedding = nn.Parameter(torch.zeros(1, max_timestamp + 1, config.params.d_model))

        self.gpt_block = GPTBlock(config.params.d_model, config.params.n_head,
                                  config.params.num_layers, config.params.dropout, batch_first=True)

        self.dropout = nn.Dropout(config.params.dropout)
        self.layer_norm = nn.LayerNorm(config.params.d_model)
        self.head = nn.Linear(config.params.d_model, move_size)

        self.lr = config.params.lr

    def forward(self, state, action, timesteps, rtgs = None):
        """
        Args:
            state (torch.Tensor): [batch, seq, state]
            action (torch.Tensor): [batch, seq, 1]
            timesteps (torch.Tensor): [batch, 1, 1]
            rtgs (torch.Tensor): [batch, seq]
        """
        if rtgs is not None:
            rtgs = rtgs.float()
            rtgs = rtgs.unsqueeze(-1)
            rtgs_encoded = self.rtgs_encoder(rtgs)
        state_encoded = self.state_encoder(state)
        action_encoded = self.action_encoder(action)

        batch_size = state.size(0) # batch, seq, state
        embedding_dim = state_encoded.size(2)
        action_encoded_pad = F.pad(action_encoded, (0, 0, 0, 1))
        if rtgs is not None:
            token_embedding = torch.cat([rtgs_encoded, state_encoded, action_encoded_pad], dim=2)
        else:
            token_embedding = torch.cat([state_encoded, action_encoded_pad], dim=2)
        token_embedding = token_embedding.view(batch_size, -1, embedding_dim)
        token_embedding = token_embedding[:, :-2, :]

        global_position_embedding = self.global_position_embedding.expand(batch_size, -1, -1)
        timesteps_expand = timesteps.expand(-1, -1, embedding_dim)
        position_embedding = (torch.gather(global_position_embedding, 1, timesteps_expand)
                              + self.position_embedding[:, :token_embedding.size(1), :])

        x = self.dropout(token_embedding + position_embedding)
        x = self.layer_norm(self.gpt_block(x))
        y = self.head(x)

        return y

    def estimate(self, rtgs, state, action, timesteps):
        batch_size = state.size(0) # batch, seq, state

        rtgs = rtgs.float()
        rtgs_encoded = self.rtgs_encoder(rtgs)
        state_encoded = self.state_encoder(state)
        if action is not None:
            action_encoded = self.action_encoder(action)
            action_encoded = F.pad(action_encoded, (0, 0, 0, 1))
            token_embedding = torch.cat([rtgs_encoded, state_encoded, action_encoded], dim=2)
        else:
            token_embedding = torch.cat([rtgs_encoded, state_encoded], dim=2)
        embedding_dim = state_encoded.size(2)
        token_embedding = token_embedding.view(batch_size, -1, embedding_dim)
        if action is not None:
            token_embedding = token_embedding[:, :-1, :] # remove padded action

        global_position_embedding = self.global_position_embedding.expand(batch_size, -1, -1)
        timesteps_expand = timesteps.expand(-1, -1, embedding_dim)
        position_embedding = (torch.gather(global_position_embedding, 1, timesteps_expand)
                              + self.position_embedding[:, :token_embedding.size(1), :])

        x = self.dropout(token_embedding + position_embedding)
        x = self.layer_norm(self.gpt_block(x))
        y = self.head(x)

        return y

    def loss_fn(self, y, x):
        return torch.nn.functional.cross_entropy(y, x)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def step(self, batch, batch_idx):
        rtgs, state, action = batch
        batch_size = rtgs.size(0)
        timestep = torch.zeros(batch_size, 1, 1, dtype=torch.int64, device=rtgs.device)
        y_hat = self(state, action, timestep, rtgs)
        action_hat = y_hat[:, 1::3, :].reshape(-1, y_hat.size(-1))
        action = action.view(-1)
        loss = self.loss_fn(action_hat, action)
        return loss


    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log('metrics/train/loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log('metrics/val/loss', loss, prog_bar=True)
        if batch_idx == 0:
            total = 0.
            try_num = 10
            for i in range(try_num):
                _, state, _ = batch
                solved = simulate(state[i, 0], self)
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


if __name__ == '__main__':
    class Params:
        d_model = 128
        n_head = 4
        num_layers = 6
        dropout = 0.1
        max_len = 11
        lr = 0.01
    class Config:
        params = Params()

    config = Config()
    model = DecisionFormer(config)
    max_len = config.params.max_len
    state = torch.zeros(10, max_len, 24, dtype=torch.int64)
    action = torch.zeros(10, max_len - 1, 1, dtype=torch.int64)
    timestep = torch.zeros(10, 1, 1, dtype=torch.int64)
    output = model(state, action, timestep)
    print(output.shape)

    rtgs = torch.zeros(10, max_len, dtype=torch.float32)
    output = model(state, action, timestep, rtgs)
    print(output.shape)

    loss = model.step((rtgs, state, action), 0)
    print(loss)

    from ...data.cube import face_str2int
    class Mock:
        def __init__(self):
            pass
    state = 'UUUURRRRFFFFDDDDLLLLBBBB'
    state_tensor = torch.tensor(face_str2int(state), dtype=torch.int64)
    ret = simulate(state_tensor, model)
    print(ret)

    from ...data.cube_model.coord import CoordCube
    from ...data.cube_model.enums import Move
    class MockDecision:
        def __init__(self):
            self.count = 0
            self.device = torch.device("cpu")

        def __call__(self, rtgs, state, action, timestep):
            action = torch.zeros(1, 1, len(Move) + 1, dtype=torch.float32)
            if self.count == 0:
                self.count += 1
                action[:, :, Move.U1.value + 1] = 1
            elif self.count == 1:
                self.count += 1
                action[:, :, Move.R1 + 1] = 1
            else:
                action[:, :, Move.U2.value + 1] = 1
            d_model = action.size(-1)
            output = torch.cat([action, action, action], dim=2).view(1, -1, d_model)
            return output

        def estimate(self, rtgs, state, action, timestep):
            return self(rtgs, state, action, timestep)

        def parameters(self):
            yield torch.tensor([1.0])

    model = MockDecision()
    cc = CoordCube()
    cc.move(Move.R3)
    cc.move(Move.U3)
    state_tensor = torch.tensor(face_str2int(cc.to_string()), dtype=torch.int64)
    print(simulate(state_tensor, model))