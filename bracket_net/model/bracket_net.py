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
    def __init__(self, bracket):
        self.bracket = bracket

    @abc.abstractmethod
    def lie_func(self, i, src, context):
        raise NotImplementedError()

    def __call__(self, i, src, context):
        return self.lie_func(i, src, context)

    def initialize_context(self, src):
        return torch.zeros_like(src[0])


class LieFuncBasic(LieFuncBase):
    def lie_func(self, i, src, context):
        if i == 0:
            x = torch.zeros_like(src[0])
            y = src[0]
        else:
            x = src[i - 1]
            y = src[i]
        context = context + x + self.bracket(x, y)
        return context, context


class LieFuncWithoutContext(LieFuncBase):
    def lie_func(self, i, src, context):
        if i == 0:
            x = torch.zeros_like(src[0])
            y = src[0]
        else:
            x = src[i - 1]
            y = src[i]
        return context, y + self.bracket(x, y)


class LieFuncContextForget(LieFuncBase):
    def __init__(self, bracket):
        super().__init__(bracket)
        self.alpha = nn.Parameter(torch.tensor(0.9))

    def lie_func(self, i, src, context):
        if i == 0:
            x = torch.zeros_like(src[0])
            y = src[0]
        else:
            x = src[i - 1]
            y = src[i]
        context = self.alpha * context + self.bracket(x, y)
        r = y + context
        return context, r


class LieFuncContextForgetMix(LieFuncBase):
    def __init__(self, bracket):
        super().__init__(bracket)
        self.alpha = nn.Parameter(torch.tensor(0.9))
        self.beta = nn.Parameter(torch.tensor(0.9))

    def lie_func(self, i, src, context):
        if i == 0:
            x = torch.zeros_like(src[0])
            y = src[0]
        else:
            x = src[i - 1]
            y = src[i]
        context = self.alpha * context + self.beta * y + self.bracket(x, y)
        r = context
        return context, r


class LieFuncBracketRule(LieFuncBase):
    def __init__(self, bracket):
        super().__init__(bracket)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def lie_func(self, i, src, context):
        if i == 0:
            x = torch.ones_like(src[0])
            y = src[0]
        else:
            x = src[i - 1]
            y = src[i]
        context = self.bracket(context.clone().detach(), self.bracket(x, y))
        r = self.alpha * y + (1 - self.alpha) * context
        return context, r

    def initialize_context(src):
        return torch.ones_like(src[0])


class LieFuncWithoutContextBracket(LieFuncBase):
    def __init__(self, bracket):
        super().__init__(bracket)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def lie_func(self, i, src, context):
        if i == 0:
            x = torch.zeros_like(src[0])
            y = src[0]
        else:
            x = src[i - 1]
            y = src[i]
        return context, self.alpha * y + (1 - self.alpha) * self.bracket(x, y)


class LieFuncVectorCondition(LieFuncBase):
    def lie_func(self, i, src, context):
        x = src[0]
        t = src[-1]
        v = self.bracket(x, t)
        if i == 0:
            ret = x + v
        else:
            ret = 2 * x + 2 * v - context
        context = x + 2 * v
        return context, ret


class LieFuncFactory():
    def __init__(self, bracket):
        self.map = {"base": LieFuncBasic,
                    "1_without_context": LieFuncWithoutContext,
                    "2_context_forget": LieFuncContextForget,
                    "3_context_forget": LieFuncContextForgetMix,
                    "4_bracket_rule": LieFuncBracketRule,
                    "5_without_context": LieFuncWithoutContextBracket,
                    "6_vector_condition": LieFuncVectorCondition}
        self.bracket = bracket

    def get(self, mode):
        try:
            return self.map[mode](self.bracket)
        except KeyError:
            raise NotImplementedError(f"mode {mode} is not implemented")


class LieNet(nn.Module):
    def __init__(self, d_model=128, n_head=4, mode="base"):
        super().__init__()
        self.activate = nn.GELU()
        self.bracket_product = nn.Linear(d_model*2, d_model)

        assert(d_model % n_head == 0)
        dim = int(d_model / n_head)

        def bracket(a, b):
            return self.activate(
                    self.bracket_product(torch.cat([a, b], dim=1)))

        self.bracket = bracket
        self.lie_func = LieFuncFactory(bracket).get(mode)

        self.layer_norm = nn.LayerNorm(d_model)
        self.d_model = d_model
        self.n_head = n_head
        self.dim = dim

    def forward(self, src):
        context = self.lie_func.initialize_context(src)
        ret = []
        for i in range(src.shape[0]):
            context, r = self.lie_func(i, src, context)
            if i >= 2:
                a = src[i - 2]
                b = src[i - 1]
                c = src[i]
                jacobi_identity = (self.bracket(a, self.bracket(b, c)) +
                                   self.bracket(b, self.bracket(c, a)) +
                                   self.bracket(c, self.bracket(a, b)))
                r += jacobi_identity
            ret.append(r.clone())
        ret = torch.stack(ret)
        return ret


if __name__ == '__main__':
    def fake(a, b):
        return None
    Func = LieFuncFactory(fake).get("base")
    model = BracketNet()
    model = LieNet()
    src = torch.ones((10, 3, 128))
    model(src)
