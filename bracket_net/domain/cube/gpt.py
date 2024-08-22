from ...model.gpt import GPTBlock
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from .utils import DebugLogger


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

    p = DebugLogger.get_instance()
    device = next(model.parameters()).device

    reward = 110
    move_reward = -1
    state_string = face_int2str(initial_state)
    co_cube = CoordCube.from_string(state_string)
    for i in range(12):
        if co_cube.is_solved():
            p.rint("solved")
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
        y = model.estimate(state, action, timestep, rtgs=rtgs)
        action = y[:, 1::3, :]
        p.rint(f"action_prob:{action[:, -1]}")
        action = action.argmax(dim=-1)
        p.rint(f"move:{action[:, -1].item()}")
        try:
            co_cube.move(Move(action[:, -1].item()))
        except Exception as e:
            p.rint("invalid move")
            return 0
        reward += move_reward
    p.rint("reach max steps")
    return 0


def simulate_with_state_action(initial_state, model):
    from ...data.cube import face_int2str
    from ...data.cube_model.enums import Move
    from ...data.cube_model.coord import CoordCube

    p = DebugLogger.get_instance()
    device = next(model.parameters()).device
    state_string = face_int2str(initial_state)
    co_cube = CoordCube.from_string(state_string)
    for i in range(12):
        if co_cube.is_solved():
            p.rint("solved")
            return 1
        if i == 0:
            state = get_single_state(co_cube.to_string(), device)
            action = None
            timestep = torch.zeros(1, 1, 1, dtype=torch.int64, device=device)
        else:
            state = torch.cat(
                [state, get_single_state(co_cube.to_string(), device)], dim=1)
            # action and timestep is no need to update
        y = model.estimate(state, action, timestep)
        action = y[:, 0::2, :].argmax(dim=-1)
        p.rint(f"move:{action[:, -1].item()}")
        try:
            co_cube.move(Move(action[:, -1].item()))
        except Exception as e:
            p.rint("invalid move")
            return 0
    p.rint("reach max steps")
    return 0


def simulate_dfs(initial_state, model):
    from ...data.cube_model.enums import Move
    from ...data.cube_model.coord import CoordCube

    p = DebugLogger.get_instance()
    co_cube = CoordCube()
    co_cube.cornperm = int(initial_state[0].item())
    co_cube.corntwist = int(initial_state[1].item())
    device = next(model.parameters()).device
    initial_state = initial_state.unsqueeze(0).unsqueeze(0)
    initial_state = initial_state / 5040.
    state = initial_state
    done = torch.zeros(1, 1, dtype=torch.int64, device=device)
    stack = None
    histories = None
    timestep = torch.zeros(1, 1, 1, dtype=torch.int64, device=device)
    for _ in range(40*4):
        done, stack, histories, state_hat = model(state, done, stack, histories, timestep)
        state = torch.cat([initial_state, state_hat], dim=1)
        done = done.argmax(dim=-1)
        if done[0, -1] == 1:
            p.rint("done found")
            p.rint(f"histories.shape:{histories.shape}")
            if histories.size(1) < 1:
                p.rint("invalid histories")
                return 0
            moves = histories[0].argmax(dim=-1)
            for move in moves:
                p.rint(f"move:{move.item()}")
                if co_cube.is_solved():
                    p.rint("solved")
                    return 1
                try:
                    co_cube.move(Move(move.item()))
                except Exception as e:
                    p.rint("invalid move")
                    return 0
    p.rint("reach max steps")
    return 0


class BaseDecisionFormer(pl.LightningModule):
    def __init__(self, config):
        from ...data.cube_model.enums import Move, Color, Facelet
        super().__init__()
        face_color_size = len(Color) + 1
        face_let_size = len(Facelet)
        move_size = len(Move)
        self.move_size = move_size
        self.state_encoder = nn.Sequential(
            nn.Embedding(face_color_size, config.params.d_model),
            nn.Flatten(start_dim=2), # batch, seq, state, d_model -> batch, seq, state * d_model
            nn.ReLU(),
            nn.Linear(face_let_size * config.params.d_model, config.params.d_model),
            nn.ReLU(),
            nn.Linear(config.params.d_model, config.params.d_model),
            nn.Tanh())
        self.action_encoder = nn.Sequential(
            nn.Embedding(move_size + 1, config.params.d_model, padding_idx=move_size),
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
        max_timestamp = config.params.max_len - 1
        self.global_position_embedding = nn.Parameter(torch.zeros(1, max_timestamp + 1, config.params.d_model))

        self.gpt_block = GPTBlock(config.params.d_model, config.params.n_head,
                                  config.params.num_layers, config.params.dropout, batch_first=True)

        self.dropout = nn.Dropout(config.params.dropout)
        self.layer_norm = nn.LayerNorm(config.params.d_model)
        self.head = nn.Linear(config.params.d_model, move_size)

        self.lr = config.params.lr

    def expand_token(self, token):
        return token

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
        token_embedding = self.expand_token(token_embedding)

        global_position_embedding = self.global_position_embedding.expand(batch_size, -1, -1)
        timesteps_expand = timesteps.expand(-1, -1, embedding_dim)
        position_embedding = (torch.gather(global_position_embedding, 1, timesteps_expand)
                              + self.position_embedding[:, :token_embedding.size(1), :])

        x = self.dropout(token_embedding + position_embedding)
        x = self.layer_norm(self.gpt_block(x))
        y = self.head(x)

        return y

    def estimate(self, state, action, timesteps, rtgs=None):
        batch_size = state.size(0) # batch, seq, state

        if rtgs is not None:
            rtgs = rtgs.float()
            rtgs_encoded = self.rtgs_encoder(rtgs)
        state_encoded = self.state_encoder(state)
        if action is not None:
            action_encoded = self.action_encoder(action)
            action_encoded = F.pad(action_encoded, (0, 0, 0, 1))
            if rtgs is not None:
                token_embedding = torch.cat([rtgs_encoded, state_encoded, action_encoded], dim=2)
            else:
                token_embedding = torch.cat([state_encoded, action_encoded], dim=2)
        else:
            if rtgs is not None:
                token_embedding = torch.cat([rtgs_encoded, state_encoded], dim=2)
            else:
                token_embedding = state_encoded
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
        return torch.nn.functional.cross_entropy(y, x, ignore_index=self.move_size)

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


class DecisionFormer(BaseDecisionFormer):
    def __init__(self, config):
        super().__init__(config)


class StateActionDecisionFormer(BaseDecisionFormer):
    def __init__(self, config):
        super().__init__(config)
        del self.rtgs_encoder

    def step(self, batch, batch_idx):
        state, action = batch
        batch_size = state.size(0)
        timestep = torch.zeros(batch_size, 1, 1, dtype=torch.int64, device=state.device)
        y_hat = self(state, action, timestep)
        action_hat = y_hat[:, 0::2, :].reshape(-1, y_hat.size(-1))
        action = action.view(-1)
        loss = self.loss_fn(action_hat, action)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log('metrics/val/loss', loss, prog_bar=True)
        if batch_idx == 0:
            total = 0.
            try_num = 10
            for i in range(try_num):
                state, _ = batch
                solved = simulate_with_state_action(state[i, 0], self)
                total += solved
            self.log('metrics/val/solved', total / try_num, prog_bar=True)
        return loss


class MemoryStateActionDecisionFormer(StateActionDecisionFormer):
    def __init__(self, config):
        super().__init__(config)
        self.memory_encoder = nn.Parameter(torch.zeros(1, 1, config.params.d_model).random_(0, 1))

    def expand_token(self, token):
        batch_size = token.size(0)
        memory = self.memory_encoder.expand(batch_size, 1, -1)
        return torch.cat([memory, token], dim=1)

    def forward(self, state, action, timesteps):
        ret = super().forward(state, action, timesteps)
        return ret[:, 1:, :]


class DFSDecisionFormer(BaseDecisionFormer):
    def __init__(self, config):
        from ...data.cube import DFS_STACK_WIDTH, DFS_STACK_HEIGHT
        super().__init__(config)
        self.state_encoder = nn.Sequential(
            nn.Linear(2, config.params.d_model),
            nn.ReLU(),
            nn.Linear(config.params.d_model, config.params.d_model),
            nn.ReLU(),
            nn.Linear(config.params.d_model, config.params.d_model),
            nn.Tanh())
        self.done_encoder = nn.Sequential(
            nn.Embedding(2 + 1, config.params.d_model),  # True, False, padding
            nn.ReLU(),
            nn.Linear(config.params.d_model, config.params.d_model),
            nn.ReLU(),
            nn.Linear(config.params.d_model, config.params.d_model),
            nn.Tanh()
        )
        self.stacks_encoder = nn.Sequential(
            nn.Flatten(start_dim=2),
            nn.Linear(DFS_STACK_WIDTH * DFS_STACK_HEIGHT, config.params.d_model),
            nn.ReLU(),
            nn.Linear(config.params.d_model, config.params.d_model),
            nn.ReLU(),
            nn.Linear(config.params.d_model, config.params.d_model),
            nn.Tanh()
        )
        self.history_size = 11
        self.history_encoder = nn.Sequential(
            nn.Linear(self.history_size, config.params.d_model),
            nn.ReLU(),
            nn.Linear(config.params.d_model, config.params.d_model),
            nn.ReLU(),
            nn.Linear(config.params.d_model, config.params.d_model),
            nn.Tanh()
        )

        self.position_embedding = nn.Parameter(torch.zeros(1, (40 * 4) + 1, config.params.d_model))

        self.state_head = nn.Linear(config.params.d_model, 2)
        self.done_head  = nn.Linear(config.params.d_model, 2)
        self.stack_head = nn.Linear(config.params.d_model, DFS_STACK_WIDTH * DFS_STACK_HEIGHT)
        self.history_head = nn.Linear(config.params.d_model, self.history_size)

        self.lr = config.params.lr

        del self.head
        del self.rtgs_encoder
        del self.action_encoder

    def forward(self, state, done, stack, histories, timesteps):
        """
        Args:
            state (torch.Tensor): [batch, seq, 2]
            done (torch.Tensor): [batch, seq] or [batch, seq, 1]
            stack (torch.Tensor): [batch, seq, DFS_STACK_WIDTH, DFS_STACK_HEIGHT]
            histories (torch.Tensor): [batch, seq, 11]
        """
        state_encoded = self.state_encoder(state)
        batch_size = state.size(0)
        embedding_dim = state_encoded.size(-1)
        shift = None
        if done is not None:
            done_encoded = self.done_encoder(done)
        if stack is not None:
            stack_encoded = self.stacks_encoder(stack)
        if histories is not None:
            if histories.size(2) < self.history_size:
                histories = F.pad(histories, (0, self.history_size - histories.size(2)))
            history_encoded = self.history_encoder(histories)

        if done is None and stack is None and histories is None:
            seq = 1
            done_encoded = torch.zeros(batch_size, seq, embedding_dim, dtype=torch.int64, device=state.device)
            stack_encoded = torch.zeros(batch_size, seq, embedding_dim, dtype=torch.float32, device=state.device)
            history_encoded = torch.zeros(batch_size, seq, embedding_dim, dtype=torch.float32, device=state.device)
            shift = -3
        elif stack is None and histories is None:
            seq = 1
            stack_encoded = torch.zeros(batch_size, seq, embedding_dim, dtype=torch.float32, device=state.device)
            history_encoded = torch.zeros(batch_size, seq, embedding_dim, dtype=torch.float32, device=state.device)
            shift = -2
        elif histories is None:
            seq = 1
            history_encoded = torch.zeros(batch_size, seq, embedding_dim, dtype=torch.float32, device=state.device)
            shift = -1

        if state_encoded.size(1) > done_encoded.size(1):
            done_encoded = F.pad(done_encoded, (0, 0, 0, state_encoded.size(1) - done_encoded.size(1)))
            stack_encoded = F.pad(stack_encoded, (0, 0, 0, state_encoded.size(1) - stack_encoded.size(1)))
            history_encoded = F.pad(history_encoded, (0, 0, 0, state_encoded.size(1) - history_encoded.size(1)))
            shift = -3
        elif done_encoded.size(1) > stack_encoded.size(1):
            stack_encoded = F.pad(stack_encoded, (0, 0, 0, state_encoded.size(1) - stack_encoded.size(1)))
            history_encoded = F.pad(history_encoded, (0, 0, 0, done_encoded.size(1) - history_encoded.size(1)))
            shift = -2
        elif stack_encoded.size(1) > history_encoded.size(1):
            history_encoded = F.pad(history_encoded, (0, 0, 0, stack_encoded.size(1) - history_encoded.size(1)))
            shift = -1

        token_embedding = torch.cat([state_encoded, done_encoded, stack_encoded, history_encoded], dim=2)
        token_embedding = token_embedding.view(batch_size, -1, embedding_dim)
        if shift is not None:
            token_embedding = token_embedding[:, :shift, :]

        global_position_embedding = self.global_position_embedding.expand(batch_size, -1, -1)
        timesteps_expand = timesteps.expand(-1, -1, embedding_dim)
        position_embedding = (torch.gather(global_position_embedding, 1, timesteps_expand)
                                + self.position_embedding[:, :token_embedding.size(1), :])

        x = self.dropout(token_embedding + position_embedding)
        x = self.layer_norm(self.gpt_block(x))
        done_hat = self.done_head(x[:, 0::4, :])
        stack_hat = self.stack_head(x[:, 1::4, :])
        history_hat = self.history_head(x[:, 2::4, :])
        state_hat = self.state_head(x[:, 3::4, :])

        return done_hat, stack_hat, history_hat, state_hat

    def loss_fn(self, y, x):
        return torch.nn.functional.mse_loss(y, x)

    def done_loss_fn(self, y, x):
        y = y.view(-1, 2)
        x = x.view(-1)
        return torch.nn.functional.cross_entropy(y, x, ignore_index=2)

    def step(self, batch, batch_idx, step_name):
        from ...data.cube import DFS_STACK_WIDTH, DFS_STACK_HEIGHT
        state, done, stack, histories = batch
        batch_size = state.size(0)
        timestep = torch.zeros(batch_size, 1, 1, dtype=torch.int64, device=state.device)
        state = state / 5040.
        stack = stack / 5040.
        done_hat, stack_hat, histories_hat, state_hat = self(state, done, stack, histories, timestep)

        done_hat = done_hat.squeeze(-1)
        done_loss = self.done_loss_fn(done_hat, done)
        stack_hat = stack_hat.view(batch_size, -1, DFS_STACK_WIDTH, DFS_STACK_HEIGHT)
        stack_loss = self.loss_fn(stack_hat, stack)
        histories_hat = histories_hat[:, :, :histories.size(2)]
        histories_loss = self.loss_fn(histories_hat, histories)
        state_loss = self.loss_fn(state_hat, state)
        loss = done_loss + stack_loss + histories_loss + state_loss

        self.log(f'metrics/{step_name}/done_loss', done_loss, prog_bar=False)
        self.log(f'metrics/{step_name}/stack_loss', stack_loss, prog_bar=False)
        self.log(f'metrics/{step_name}/histories_loss', histories_loss, prog_bar=False)
        self.log(f'metrics/{step_name}/state_loss', state_loss, prog_bar=False)
        self.log(f'metrics/{step_name}/loss', loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            total = 0.
            try_num = 10
            for i in range(try_num):
                state, _, _, _ = batch
                solved = simulate_dfs(state[i, 0], self)
                total += solved
            self.log('metrics/val/solved', total / try_num, prog_bar=True)
        return self.step(batch, batch_idx, 'val')


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

        def __call__(self, state, action, timestep, rtgs):
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

        def estimate(self, state, action, timestep, rtgs=None):
            return self(state, action, timestep, rtgs)

        def parameters(self):
            yield torch.tensor([1.0])

    model = MockDecision()
    cc = CoordCube()
    cc.move(Move.R3)
    cc.move(Move.U3)
    state_tensor = torch.tensor(face_str2int(cc.to_string()), dtype=torch.int64)
    print(simulate(state_tensor, model))

    print("StateActionDecisionFormer")
    model = StateActionDecisionFormer(config)
    state = torch.zeros(10, max_len, 24, dtype=torch.int64)
    action = torch.zeros(10, max_len - 1, 1, dtype=torch.int64)
    output = model(state, action, timestep)
    print(output.shape)

    print("MemoryStateActionDecisionFormer")
    model = MemoryStateActionDecisionFormer(config)
    state = torch.zeros(10, max_len, 24, dtype=torch.int64)
    action = torch.zeros(10, max_len - 1, 1, dtype=torch.int64)
    output = model(state, action, timestep)
    print(output.shape)

    print("DFSDecisionFormer")
    model = DFSDecisionFormer(config)
    state = torch.zeros(10, 1, 2, dtype=torch.float32)
    done = None
    stack = None
    histories = None
    done_hat, stack_hat, histories_hat, state_hat = model(state, done, stack, histories, timestep)
    print("done", done_hat.shape)
    print("stack", stack_hat.shape)
    print("histories", histories_hat.shape)
    print("state", state_hat.shape)
    print()

    done = torch.zeros(10, 1, dtype=torch.int64)
    done_hat, stack_hat, histories_hat, state_hat = model(state, done, stack, histories, timestep)
    print("done", done_hat.shape)
    print("stack", stack_hat.shape)
    print("histories", histories_hat.shape)
    print("state", state_hat.shape)
    print()

    from ...data.cube import DFS_STACK_WIDTH, DFS_STACK_HEIGHT
    stack = torch.zeros(10, 1, DFS_STACK_WIDTH, DFS_STACK_HEIGHT, dtype=torch.float32)
    done_hat, stack_hat, histories_hat, state_hat = model(state, done, stack, histories, timestep)
    print("done", done_hat.shape)
    print("stack", stack_hat.shape)
    print("histories", histories_hat.shape)
    print("state", state_hat.shape)
    print()

    histories = torch.zeros(10, 1, 11, dtype=torch.float32)
    done_hat, stack_hat, histories_hat, state_hat = model(state, done, stack, histories, timestep)
    print("done", done_hat.shape)
    print("stack", stack_hat.shape)
    print("histories", histories_hat.shape)
    print("state", state_hat.shape)
    print()

    state = torch.zeros(10, 2, 2, dtype=torch.float32)
    done_hat, stack_hat, histories_hat, state_hat = model(state, done, stack, histories, timestep)
    print("done", done_hat.shape)
    print("stack", stack_hat.shape)
    print("histories", histories_hat.shape)
    print("state", state_hat.shape)
    print()

    done = torch.zeros(10, 2, dtype=torch.int64)
    done_hat, stack_hat, histories_hat, state_hat = model(state, done, stack, histories, timestep)
    print("done", done_hat.shape)
    print("stack", stack_hat.shape)
    print("histories", histories_hat.shape)
    print("state", state_hat.shape)
    print()

    stack = torch.zeros(10, 2, DFS_STACK_WIDTH, DFS_STACK_HEIGHT, dtype=torch.float32)
    done_hat, stack_hat, histories_hat, state_hat = model(state, done, stack, histories, timestep)
    print("done", done_hat.shape)
    print("stack", stack_hat.shape)
    print("histories", histories_hat.shape)
    print("state", state_hat.shape)
    print()

    histories = torch.zeros(10, 2, 11, dtype=torch.float32)
    done_hat, stack_hat, histories_hat, state_hat = model(state, done, stack, histories, timestep)
    print("done", done_hat.shape)
    print("stack", stack_hat.shape)
    print("histories", histories_hat.shape)
    print("state", state_hat.shape)
    print()

    print("simulate_dfs")
    state_tensor = torch.zeros(2, dtype=torch.float)
    ret = simulate_dfs(state_tensor, model)
    print(ret)