import torch
import torch.nn as nn
import torch.nn.functional as F
import abc


class BracketNet(nn.Module):
    def __init__(self,
                 d_model=128, n_head=4):
        super().__init__()
        self.bracket_product = nn.Linear(d_model*2, d_model)
        self.activate = nn.GELU()

        assert(d_model % n_head == 0)
        dim = int(d_model / n_head)

        def bracket(a, b):
            return self.activate(
                    self.bracket_product(torch.cat([a, b], dim=1)))
        self.bracket = bracket
        self.target_functions = nn.ModuleList()
        for i in range(n_head):
            self.target_functions.append(
                    nn.Sequential(nn.Linear(dim, dim * 2),
                                  self.activate,
                                  nn.Linear(dim * 2, dim)))
        self.layer_norm = nn.LayerNorm(d_model)
        self.d_model = d_model
        self.n_head = n_head
        self.dim = dim

    def forward(self, src):
        ret = []
        conditions_results = {0: 0, 1: 0, 2: 0}
        for i in range(src.shape[0]):
            # batch, d_model to batch, n_head, dim
            srcs = src[i].view(-1, self.n_head, self.dim)
            r = torch.zeros_like(srcs).cuda()
            for j in range(self.n_head):
                r[:, j] = self.activate(
                        self.target_functions[j](srcs[:, j])) + srcs[:, j]
            r = r.view(-1, self.d_model)
            r = self.layer_norm(r)
            context = r
            ret.append(r.clone())
        ret = torch.stack(ret)
        return ret, conditions_results


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


class LieNet(nn.Module):
    def __init__(self, d_model=128, n_head=4, mode="base"):
        super().__init__()
        self.activate = nn.GELU()

        assert(d_model % n_head == 0)
        dim = int(d_model / n_head)
        self.bracket_products = []
        for i in range(n_head):
            self.bracket_products.append(nn.Linear(dim*2, dim))

        def bracket(a, b, i):
            return self.activate(
                    self.bracket_products[i](torch.cat([a, b], dim=1)))

        self.bracket = bracket
        self.lie_func = LieFuncFactory(bracket, d_model, n_head, dim).get(mode)

        self.d_model = d_model
        self.n_head = n_head
        self.dim = dim

    def forward(self, src):
        context = self.lie_func.initialize_context(src)
        ret = []
        for i in range(src.shape[0]):
            context, r = self.lie_func(src[i], context)
            """
            if i >= 2:
                a = src[i - 2]
                b = src[i - 1]
                c = src[i]
                jacobi_identity = (self.bracket(a, self.bracket(b, c)) +
                                   self.bracket(b, self.bracket(c, a)) +
                                   self.bracket(c, self.bracket(a, b)))
                r += jacobi_identity
            """
            ret.append(r.clone())
        ret = torch.stack(ret)
        return ret


if __name__ == '__main__':
    def fake(a, b):
        return None
    Func = LieFuncFactory(fake, 128, 4, 32).get("base")
    model = BracketNet()
    model = LieNet()
    src = torch.ones((3, 10, 128))
    model(src)
    model = LieNet(mode="1_without_context")
    model(src)
    model = LieNet(mode="2_context_forget")
    model(src)
    model = LieNet(mode="3_context_forget")
    model(src)
    model = LieNet(mode="4_bracket_rule")
    model(src)
    model = LieNet(mode="5_without_context")
    model(src)
    model = LieNet(mode="6_vector_condition")
    model(src)
