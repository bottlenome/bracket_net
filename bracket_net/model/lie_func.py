import torch
import torch.nn as nn
import abc


class LieFuncBase(nn.Module):
    def __init__(self, bracket, d_model, n_head, dim):
        super().__init__()
        self.bracket = bracket
        self.d_model = d_model
        self.n_head = n_head
        self.dim = dim

    @abc.abstractmethod
    def lie_func(self, c, x, head_id):
        # c, x: [batch, 1, dim]
        raise NotImplementedError()

    def call_1d(self, context, src, src_1):
        c, y = self.lie_func(context, src, src_1, 0)
        return c, y

    def call_nd(self, context, src):
        # batch, d_model to batch, n_head, dim
        srcs = src.view(-1, self.n_head, self.dim)
        y_list = []
        contexts = context.view(-1, self.n_head, self.dim)
        c_list = []

        for head_id in range(self.n_head):
            c, y = self.lie_func(
                    contexts[:, head_id],
                    srcs[:, head_id],
                    head_id)
            c_list.append(c)
            y_list.append(y)

        c = torch.stack(c_list, dim=1)
        y = torch.stack(y_list, dim=1)
        return c, y

    def __call__(self, context, src, src_1):
        if self.n_head == 1:
            return self.call_1d(context, src, src_1)
        else:
            return self.call_nd(context, src)

    def initialize_context(self, src):
        return torch.zeros_like(src[0])


class LieFuncBasic(LieFuncBase):
    def lie_func(self, c, x_i, x_i_1, head_id):
        context = c + x_i_1 + self.bracket(x_i_1, x_i, head_id)
        y = context
        return context, y


class LieFuncBasicOptimized(nn.Module):
    def __init__(self, bracket, d_model, n_head, dim):
        super().__init__()
        self.activate = nn.ReLU()
        # convert batch, d_model, seq -> batch, d_model, seq
        self.bracket = nn.Conv1d(in_channels=d_model,
                                 out_channels=d_model,
                                 kernel_size=2,
                                 padding=1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        vec = self.activate(self.bracket(x))
        pos = x + vec[:, :, :-1]
        # integral out
        context = torch.cumsum(pos, dim=2)
        y = context
        y = self.norm(y.permute(0, 2, 1)).permute(0, 2, 1)
        return y


class LieFuncWithoutContext(LieFuncBase):
    def lie_func(self, c, x_i, x_i_1, head_id):
        y = x_i_1 + self.bracket(x_i_1, x_i, head_id)
        return c, y


class LieFuncWithoutContextOptimized(nn.Module):
    def __init__(self, bracket, d_model, n_head, dim):
        super().__init__()
        self.activate = nn.ReLU()
        self.bracket = nn.Conv1d(in_channels=d_model,
                                 out_channels=d_model,
                                 kernel_size=2,
                                 padding=1)

    def forward(self, x):
        vec = self.activate(self.bracket(x))
        pos = x + vec[:, :, :-1]
        y = pos
        return y


class LieFuncContextForget(LieFuncBase):
    def __init__(self, bracket, d_model, n_head, dim):
        super().__init__(bracket, d_model, n_head, dim)
        self.alpha = nn.Parameter(torch.tensor(0.9))

    def lie_func(self, c, x, head_id):
        context = self.alpha * c + self.bracket(c, x, head_id)
        y = x + c
        return context, y


class LieFuncContextForgetMix(LieFuncBase):
    def __init__(self, bracket, d_model, n_head, dim):
        super().__init__(bracket, d_model, n_head, dim)
        self.alpha = nn.Parameter(torch.tensor(0.9))
        self.beta = nn.Parameter(torch.tensor(0.9))

    def lie_func(self, c, x, head_id):
        context = self.alpha * c + self.beta * x + self.bracket(c, x, head_id)
        y = context
        return context, y


class LieFuncBracketRule(LieFuncBase):
    def __init__(self, bracket, d_model, n_head, dim):
        super().__init__(bracket, d_model, n_head, dim)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def lie_func(self, c, x_i, x_i_1, head_id):
        context = self.bracket(
                c.clone().detach(), self.bracket(x_i, x_i_1, head_id), head_id)
        y = self.alpha * x_i_1 + (1 - self.alpha) * context
        return context, y

    def initialize_context(self, src):
        return torch.ones_like(src[0])


class LieFuncWithoutContextBracket(LieFuncBase):
    def __init__(self, bracket, d_model, n_head, dim):
        super().__init__(bracket, d_model, n_head, dim)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def lie_func(self, c, x, head_id):
        c = x
        return c, (self.alpha * x +
                   (1 - self.alpha) * self.bracket(c, x, head_id))


class LieFuncVectorCondition(LieFuncBase):
    def lie_func(self, c, x, head_id):
        context = x
        y = x
        v = self.bracket(c, x, head_id)
        # c + v == x : bracket_product becomes direction vector caclucation
        # c = x[i]
        # v_2 = self.bracket(x[i], x[i+2])
        # 2*v = v_2
        zero_condition = c + v - x
        return context, y + zero_condition


class LieFucWithFixedContext2DWeightOptimized(nn.Module):
    def __init__(self, bracket, d_model, n_head, dim, seq_len=1024):
        super().__init__()
        self.bracket = bracket
        self.d_model = d_model
        self.n_head = n_head
        self.dim = dim
        self.seq_len = seq_len
        self.problem_len = 1 + seq_len * 3 + 1
        self.answer_len = seq_len + 2
        self.seq_max = self.problem_len + self.answer_len
        self.weight = nn.Parameter(
            torch.randn(n_head, self.seq_max, self.seq_max))
        self.activate = nn.ReLU()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        weight = self.weight.unsqueeze(1).expand(-1, self.dim, -1, -1).reshape(self.d_model, self.seq_max, -1)
        w = torch.tril(weight, diagonal=-1)
        # dot(x[b, d, :], w[d, i, :]) -> c[b, d, i]
        c = self.activate(torch.einsum("bdw, diw -> bdi", x, w[:, :x.size(2), :x.size(2)]))
        # assert(c[0, 0, 0].item() < 0.001)
        # assert(c[0, 0, 1].item() - x[0, 0, 0].item() * w[0, 1, 0].item())
        vecs = []
        for i in range(self.n_head):
            index = i * self.dim
            vec = self.bracket(c[:, index:index+self.dim], x[:, index:index+self.dim], i)
            vecs.append(vec)
        y = x + torch.cat(vecs, dim=1)
        y = self.norm(y.permute(0, 2, 1)).permute(0, 2, 1)
        return y

class LieFucWithFixedContext1DWeightOptimized(nn.Module):
    def __init__(self, bracket, d_model, n_head, dim, seq_len=1024):
        super().__init__()
        self.bracket = bracket
        self.d_model = d_model
        self.n_head = n_head
        self.dim = dim
        self.seq_len = seq_len
        self.problem_len = 1 + seq_len * 3 + 1
        self.answer_len = seq_len + 2
        seq_max = self.problem_len + self.answer_len
        self.weight = nn.Parameter(torch.randn(n_head, seq_max))
        self.activate = nn.ReLU()
        self.norm = nn.LayerNorm(d_model)

    # x: [batch, d_model, seq]
    def forward(self, x):
        # [n_head, seq_max] -> [n_head, dim, seq_max]
        weight = self.weight.unsqueeze(1).expand(-1, self.dim, -1)
        weight = weight.reshape(self.d_model, -1).unsqueeze(2).expand(-1, -1, x.size(2))
        w = torch.tril(weight, diagonal=-1)
        # dot(x[b, d, :], w[d, i, :]) -> c[b, d, i]
        c = self.activate(torch.einsum("bdw, diw -> bdi", x, w[:, :x.size(2), :x.size(2)]))
        # assert(c[0, 0, 0].item() < 0.001)
        # assert(c[0, 0, 1].item() - x[0, 0, 0].item() * w[0, 1, 0].item())
        vecs = []
        for i in range(self.n_head):
            index = i * self.dim
            vec = self.bracket(c[:, index:index+self.dim], x[:, index:index+self.dim], i)
            vecs.append(vec)
        y = x + torch.cat(vecs, dim=1)
        y = self.norm(y.permute(0, 2, 1)).permute(0, 2, 1)
        return y


class LieFucWithBracketWeightOptimized(nn.Module):
    def __init__(self, bracket, d_model, n_head, dim):
        super().__init__()
        self.bracket = bracket
        self.d_model = d_model
        self.n_head = n_head
        self.dim = dim
        self.activate = nn.ReLU()

    def forward(self, x):
        batch, dim, seq = x.shape
        h, w = torch.tril_indices(seq, seq)
        x_h = x[:, :, h]
        x_w = x[:, :, w]
        y = torch.cat([x_w, x_h], dim=1)
        y_out = torch.zeros(batch, dim*2, seq, seq, device=x.device)
        y_out[:, :, h, w] = y

        # get bracket weight and apply
        y_out = y_out.permute(0, 2, 3, 1).contiguous().view(batch, seq, seq, dim*2)
        weight = self.bracket.bracket_products[0](y_out)
        weight = weight.permute(0, 3, 1, 2)
        assert(weight.shape == (batch, dim, seq, seq))
        assert(weight[0, 0, 0, 1].item() < 0.001)
        assert(weight[0, 0, 1, 2].item() < 0.001)
        weight = weight.norm(dim=1, keepdim=True)
        assert(weight[0, 0, 0, :].sum().item() < 1)
        weight = 1 - weight

        # dot(x[b, d, :], w[batch, dim, i, :]) -> c[b, d, i]
        c = torch.einsum("bdw, bdwi -> bdi", x, weight)

        return c

class LieFucWithFixedContextWeightOptimized(nn.Module):
    def __init__(self, bracket, d_model, n_head, dim, seq_len=1024):
        super().__init__()
        self.bracket = bracket
        self.d_model = d_model
        self.n_head = n_head
        self.dim = dim
        self.seq_len = seq_len
        self.problem_len = 1 + seq_len * 3 + 1
        self.answer_len = seq_len + 2
        seq_max = self.problem_len + self.answer_len
        self.weight = nn.Parameter(
            torch.randn(dim, seq_max, seq_max))
        self.activate = nn.ReLU()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        w = torch.tril(self.weight, diagonal=-1)
        # dot(x[0, 0, :], w[0, i, :]) -> c[:, :, i]
        c = self.activate(torch.einsum("bdw, diw -> bdi", x, w[:, :x.size(2), :x.size(2)]))
        # assert(c[0, 0, 0].item() < 0.001)
        # assert(c[0, 0, 1].item() - x[0, 0, 0].item() * w[0, 1, 0].item())
        y = x + self.bracket(c, x, 0)
        y = self.norm(y.permute(0, 2, 1)).permute(0, 2, 1)
        return y


class LieFuncBeamSearchOptimized(nn.Module):
    def __init__(self, bracket, d_model, n_head, dim, seq_len=1024):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.dim = dim
        self.seq_len = seq_len
        self.problem_len = 1 + seq_len * 3 + 1
        self.answer_len = seq_len + 2
        self.seq_max = self.problem_len + self.answer_len
        self.weight = nn.Parameter(
            torch.randn(n_head, self.seq_max, self.seq_max))
        self.activate = nn.ReLU()
        self.norm = nn.LayerNorm(d_model)

        self.max_turn = 7
        self.action_size = 8
        self.beam_size = 10
        self.context2state_func = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model))
        self.get_next_action_func = nn.Sequential(
            nn.Linear(d_model + d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, self.action_size),
            nn.Softmax(dim=-1))
        self.estimate_goal_func = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model))
        self.action = nn.Parameter(
            torch.randn(self.action_size, d_model))

        self.debug_count = 0

    def context2state(self, context):
        """
            Args: context: [*, d_model]
            Returns: [*, d_model]
        """
        return self.context2state_func(context)

    def get_next_action_prob(self, states, goal):
        """
            Args: states: [batch, seq, beam_size, d_model]
                  goal: [batch, seq, d_model]
            Returns: [batch, seq, beam_size, action_size]
        """
        # [batch, seq, beam_size, d_model] -> [batch, seq, beam_size, action_size]
        goal_expand = goal.expand(-1, -1, states.size(2), -1)
        action_prob = self.get_next_action_func(
            torch.cat([states, goal_expand], dim=-1))
        return action_prob

    def estimate_goal(self, contexts):
        return self.estimate_goal_func(contexts)

    def determine_index(self, states_prob, action_prob, beam_size_next):
        """
            Args: states_prob: [batch, seq, beam_size_now]
                  action_prob: [batch, seq, beam_size_now, action_size]
                  beam_size_next: int
            Returns: beam_index: [batch, seq, beam_size_next]
                     action_index: [batch, seq, beam_size_next]
                     states_prob: [batch, seq, beam_size_next]
        """
        combined_prob = states_prob.unsqueeze(-1).expand(-1, -1, -1, self.action_size) * action_prob
        combined_prob = combined_prob.reshape(combined_prob.size(0), combined_prob.size(1), -1)
        states_prob, combined_index = combined_prob.topk(beam_size_next, dim=-1)
        # [batch, seq, beam_size_next]
        beam_index = combined_index // self.action_size
        action_index = combined_index % self.action_size
        return beam_index, action_index, states_prob
    
    def update_states(self, states, beam_index, action_index, beam_size_next):
        """
            Args: states: [batch, seq, beam_size_now, d_model]
                  beam_index: [batch, seq, beam_size_next]
                  action_index: [batch, seq, beam_size_next]
                  beam_size_next: int
            Returns: [batch, seq, beam_size_next, d_model]
        """
        batch = states.size(0)
        seq = states.size(1)
        d_model = states.size(3)
        states_base = states.gather(dim=2, index=beam_index.unsqueeze(-1).expand(-1, -1, -1, d_model))
        # choose action by index
        # for i in range(beam_size_next):
        #     ret[:, :, i, :] = ret[:, :, i, :] + self.action[action_index[:, i]]
        action = self.action[action_index]
        return self.norm(states_base + action)

    def update_history(self, history, beam_index, action_index, turn):
        """
            Args: history: [batch, seq, beam_size, max_turn]
                  beam_index: [batch, seq, beam_size_next]
                  action_index: [batch, seq, beam_size_next]
                  turn: int
            Returns: [batch, seq, beam_size, max_turn]
        """
        max_turn = history.size(3)
        next_history = torch.zeros_like(history)
        # for i in range(beam_size):
        #     next_history[:, :, i, :turn] = history[:, :, beam_index[:, i], :turn]
        #     next_history[:, :, i, turn] = action_index[:, i]
        next_history = history.gather(dim=2, index=beam_index.unsqueeze(-1).expand(-1, -1, -1, max_turn))
        next_history[:, :, :, turn] = action_index
        return next_history

    def get_next_state(self, states: torch.Tensor,
                             states_prob: torch.Tensor,
                             action_prob: torch.Tensor,
                             history: torch.Tensor,
                             turn: int,
                             beam_size: int):
        """
            Args: states: [batch, seq, beam_size_now, d_model]
                  states_prob: [batch, seq, beam_size_now]
                  action_prob: [batch, seq, beam_size_now, action_size]
                  history: [batch, seq, beam_size, max_turn]
                  turn: int
                  beam_size: int
            Returns: next_states: [batch, seq, beam_size_next, d_model]
                     states_prob: [batch, seq, beam_size_next]
                     history: [batch, seq, beam_size, max_turn]
        """
        beam_size_now = states.size(2)
        beam_size_next = beam_size_now * self.action_size
        if beam_size_next > beam_size:
            beam_size_next = beam_size
        beam_index, action_index, states_prob = self.determine_index(states_prob, action_prob, beam_size_next)
        next_states = self.update_states(states, beam_index, action_index, beam_size_next)
        history = self.update_history(history, beam_index, action_index, turn)
        return next_states, states_prob, history


    def reached(self, states, goal):
        """ Get mask of unreached goal

            Args: states: [batch, seq, beam_size, d_model]
                  goal: [batch, seq, d_model]
            Returns: [batch, seq, beam_size]
        """
        unreached = (states - goal.unsqueeze(2)).abs().sum(dim=-1) < 1.0e-10
        assert(unreached.shape == (states.size(0), states.size(1), states.size(2)))
        return unreached

    # x: [batch, d_model, seq]
    # ret: [batch, d_model, seq]
    def forward(self, x):
        # seq2context
        w = torch.tril(self.weight[0, :x.size(2), :x.size(2)], diagonal=0)
        c = self.activate(x @ w)
        # [batch, d_model, seq] -> [batch, seq, 1, d_model]
        states = self.context2state(c.permute(0, 2, 1).unsqueeze(2))
        # save for last stage
        initial_states = states.clone()
        # Initialize state probability as 1
        # [batch, seq, 1]
        states_prob = torch.ones((states.size(0), states.size(1), states.size(2)), device=states.device, requires_grad=False)

        goal = self.estimate_goal(c.permute(0, 2, 1).unsqueeze(2))

        # history: [batch, seq, beam_size, max_turn]
        history = torch.zeros((states.size(0), states.size(1), self.beam_size, self.max_turn), device=states.device, requires_grad=False, dtype=torch.uint8)

        for turn in range(self.max_turn):
            # Get action probability from state
            # ([batch, seq, beam_size, d_model], [batch, 1, beam_size, d_model])
            # -> [batch, seq, beam_size, prob]
            action_prob = self.get_next_action_prob(states, goal)
            states, states_prob, history = self.get_next_state(states, states_prob, action_prob, history, turn, self.beam_size)
            # unreached = self.reached(states, goal)
            # if self.unreached.sum().item() == 0:
            #     break
        # return most probable next state
        history_best = states_prob.argmax(dim=-1)
        beam_index = history_best.unsqueeze(-1)
        # [batch, seq]
        # action_index = torch.zeros((states.size(0), states.size(1)), device=states.device, dtype=torch.long)
        # for batch in range(states.size(0)):
        #     for seq in range(states.size(1)):
        #         action_index[batch, seq] = history[batch, seq, history_best[batch, seq], 0]
        # action_index = action_index.unsqueeze(-1)
        # print(action_index)
        indices = history_best.unsqueeze(-1)
        action_index = history[:, :, :, 0].gather(dim=2, index=indices).type(torch.long)
        next_states = self.update_states(initial_states, beam_index, action_index, 1)
        # debug
        if self.debug_count % 100 == 0:
            print("history", history[0, 100])
            print("states_prob", states_prob[0, 100])
            print("history_best", history_best[0, 100])
            print("action_index", action_index[0, 100])
        self.debug_count += 1

        return next_states.squeeze(2).permute(0, 2, 1)


class LieFuncFactory():
    def __init__(self, bracket, d_model, n_head, dim):
        self.map = {
                "base": LieFuncBasic,
                "base_optimized": LieFuncBasicOptimized,
                "1_without_context": LieFuncWithoutContext,
                "1_without_context_optimized": LieFuncWithoutContextOptimized,
                "2_context_forget": LieFuncContextForget,
                "3_context_forget": LieFuncContextForgetMix,
                "4_bracket_rule": LieFuncBracketRule,
                "5_without_context": LieFuncWithoutContextBracket,
                "6_vector_condition": LieFuncVectorCondition,
                "7_fixed_context_2d_weight_optimized": LieFucWithFixedContext2DWeightOptimized,
                "8_fixed_context_1d_weight_optimized": LieFucWithFixedContext1DWeightOptimized,
                "9_bracket_weight_optimized": LieFucWithBracketWeightOptimized,
                "10_fixed_context_weight_optimized": LieFucWithFixedContextWeightOptimized,
                "11_beam_search_optimized": LieFuncBeamSearchOptimized,
                }
        self.bracket = bracket
        self.d_model = d_model
        self.n_head = n_head
        self.dim = dim

    def get(self, mode):
        try:
            return self.map[mode](self.bracket, self.d_model,
                                  self.n_head, self.dim)
        except KeyError:
            raise NotImplementedError(f"mode {mode} is not implemented")


if __name__ == '__main__':
    def fake(a, b):
        return None
    func = LieFuncFactory(fake, 128, 4, 32).get("base")

    batch = 2
    d_model = 16
    n_head = 1
    seq = 11
    beam_search = LieFuncBeamSearchOptimized(fake, d_model=d_model, n_head=n_head, dim=d_model)
    src = torch.randn((batch, d_model, seq))
    dst = beam_search(src)
    assert(src.shape == dst.shape)
    dst.sum().backward()

    src = src.permute(0, 2, 1).unsqueeze(2)
    dst = beam_search.context2state(src)
    assert(src.shape == dst.shape)

    goal = beam_search.estimate_goal(src)
    assert(goal.shape == src.shape)

    action_prob = beam_search.get_next_action_prob(src, goal)
    assert(action_prob.shape == (batch, seq, 1, beam_search.action_size))
    print("action_prob", action_prob[0, 0, 0])

    states_prob = torch.ones((batch, seq, 1), device=src.device)
    action_prob = torch.ones((batch, seq, 1, beam_search.action_size), device=src.device) / beam_search.action_size
    beam_size_next = beam_search.action_size
    beam_index, action_index, states_prob = beam_search.determine_index(states_prob, action_prob, beam_size_next)
    assert(states_prob.shape == (batch, seq, beam_size_next))
    assert(states_prob.sum().item() == batch * seq)
    assert(beam_index.shape == (batch, seq, beam_size_next))
    print(beam_index[0, 0])
    print(action_index[0, 0])

    action_prob[:, :, :, 0] = 0
    action_prob[:, :, :, 1] = 2./8
    beam_index, action_index, states_prob = beam_search.determine_index(states_prob, action_prob, beam_size_next)
    print("action index of action prob 0.2 at index 1")
    print(action_index[0, 0])

    beam_index = torch.zeros((batch, seq, beam_size_next), device=src.device, dtype=torch.long)
    next_states = beam_search.update_states(src, beam_index, action_index, beam_size_next)
    assert(next_states.shape == (batch, seq, beam_size_next, d_model))
    print("next_states", next_states[0, 0, 0])

    history = torch.zeros((batch, seq, beam_search.beam_size, beam_search.max_turn), device=src.device, requires_grad=False)
    history = beam_search.update_history(history, beam_index, action_index, 0)
    print("history", history[0, 0, 0])


