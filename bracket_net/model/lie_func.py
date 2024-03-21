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
        context = c + x_i + self.bracket(x_i_1, x_i, head_id)
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

    def forward(self, x):
        vec = self.activate(self.bracket(x))
        pos = x + vec[:, :, 1:]
        # integral out
        context = torch.cumsum(pos, dim=2)
        y = context
        return y


class LieFuncWithoutContext(LieFuncBase):
    def lie_func(self, c, x, head_id):
        context = x
        y = x + self.bracket(c, x, head_id)
        return context, y


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
        pos = x + vec[:, :, 1:]
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

    def lie_func(self, c, x, head_id):
        context = self.bracket(
                c.clone().detach(), self.bracket(c, x, head_id), head_id)
        y = self.alpha * x + (1 - self.alpha) * context
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
                "6_vector_condition": LieFuncVectorCondition}
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
