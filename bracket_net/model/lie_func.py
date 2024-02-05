import torch
import torch.nn as nn
import abc


class LieFuncBase():
    def __init__(self, bracket, d_model, n_head, dim):
        self.bracket = bracket
        self.d_model = d_model
        self.n_head = n_head
        self.dim = dim
        self.layer_norm = nn.LayerNorm(d_model)

    @abc.abstractmethod
    def lie_func(self, c, x, head_id):
        # c, x: [batch, 1, dim]
        raise NotImplementedError()

    # src: [batch, d_model]
    def __call__(self, src, context):
        # batch, d_model to batch, n_head, dim
        srcs = src.view(-1, self.n_head, self.dim)
        y = torch.zeros_like(srcs)
        contexts = context.view(-1, self.n_head, self.dim)
        c = torch.zeros_like(contexts)

        for head_id in range(self.n_head):
            c[:, head_id], y[:, head_id] = self.lie_func(
                    srcs[:, head_id],
                    contexts[:, head_id],
                    head_id)

        c = c.view(-1, self.d_model)
        c = self.layer_norm(c)
        y = y.view(-1, self.d_model)
        y = self.layer_norm(y)
        return c, y

    def initialize_context(self, src):
        return torch.zeros_like(src[0])


class LieFuncBasic(LieFuncBase):
    def lie_func(self, c, x, head_id):
        context = x
        y = self.bracket(c, x, head_id)
        return context, y


class LieFuncWithoutContext(LieFuncBase):
    def lie_func(self, c, x, head_id):
        context = x
        y = x + self.bracket(c, x, head_id)
        return context, y


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
        self.map = {"base": LieFuncBasic,
                    "1_without_context": LieFuncWithoutContext,
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
