import torch
import torch.nn as nn
from lie_func import LieFuncFactory


class BracketNet(nn.Module):
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
    model = BracketNet()
    src = torch.ones((3, 10, 128))
    model(src)
    model = BracketNet(mode="1_without_context")
    model(src)
    model = BracketNet(mode="2_context_forget")
    model(src)
    model = BracketNet(mode="3_context_forget")
    model(src)
    model = BracketNet(mode="4_bracket_rule")
    model(src)
    model = BracketNet(mode="5_without_context")
    model(src)
    model = BracketNet(mode="6_vector_condition")
    model(src)
