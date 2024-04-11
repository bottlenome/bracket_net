import torch
import torch.nn as nn
from .lie_func import LieFuncFactory


class BracketFunc(nn.Module):
    def __init__(self, d_model=128, n_head=4, mode="base"):
        super().__init__()
        assert(d_model % n_head == 0)

        dim = int(d_model / n_head)
        self.bracket_products = nn.ModuleList()
        self.activate = nn.ReLU()
        for _ in range(n_head):
            self.bracket_products.append(nn.Linear(dim*2, dim))

        class Bracket():
            def __init__(self, activate, bracket_products):
                self.activate = activate
                self.bracket_products = bracket_products

            def __call__(self, a, b, i):
                input = torch.cat([a, b], dim=1)
                # check input is batch, d_model or batch, d_model, seq
                if len(input.shape) == 2:
                    return self.activate(self.bracket_products[i](input))
                elif len(input.shape) == 3:
                    ret = self.activate(
                        self.bracket_products[i](input.permute(0, 2, 1)))
                    return ret.permute(0, 2, 1)
                else:
                    raise ValueError("Invalid input shape")

        bracket = Bracket(self.activate, self.bracket_products)

        """
        def bracket(a, b, i):
            input = torch.cat([a, b], dim=1)
            # check input is batch, d_model or batch, d_model, seq
            if len(input.shape) == 2:
                return self.activate(self.bracket_products[i](input))
            elif len(input.shape) == 3:
                ret = self.activate(
                    self.bracket_products[i](input.permute(0, 2, 1)))
                return ret.permute(0, 2, 1)
            else:
                raise ValueError("Invalid input shape")
        """
        self.bracket = bracket
        self.lie_func = LieFuncFactory(bracket, d_model, n_head, dim).get(mode)

        self.d_model = d_model
        self.n_head = n_head
        self.dim = dim
        self.mode_optimized = (mode.find("optimized") != -1)

    def forward(self, src):
        if self.mode_optimized:
            # seq, batch, d_model -> batch, d_model, seq
            src = src.permute(1, 2, 0)
            out = self.lie_func(src)
            # batch, d_model, seq -> seq, batch, d_model
            out = out.permute(2, 0, 1)
            return out

        context = self.lie_func.initialize_context(src)
        ret = [context.clone()]
        for i in range(src.shape[0] - 1):
            context, r = self.lie_func(context, src[i], src[i+1])
            ret.append(r.clone())
        ret = torch.stack(ret)
        return ret


class BracketNet(nn.Module):
    def __init__(self, d_vocab, pos_encoder,
                 d_model=128, n_head=4,
                 num_layers=6, dropout=0.0,
                 embed=None, mode="base"):
        super().__init__()
        self.d_vocab = d_vocab
        self.d_model = d_vocab
        self.n_head = n_head
        self.dim = int(d_vocab / n_head)
        # self.activate = nn.GELU()
        self.activate = nn.ReLU()

        if embed is None:
            self.embed = torch.nn.Embedding(d_vocab + 1, d_model)
        else:
            self.embed = embed
        self.pos_encoder = pos_encoder(d_model, dropout)
        self.map = nn.Linear(d_model, d_model)
        self.bracket_funcs = nn.Sequential()
        for i in range(num_layers):
            self.bracket_funcs.add_module(
                f"bracket_func{i}", BracketFunc(d_model, n_head, mode))
        self.unembed = torch.nn.Linear(d_model, d_vocab)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src = self.embed(src)
        src = self.pos_encoder(src)
        src = self.activate(self.map(src))
        out = self.bracket_funcs(src)
        out = self.unembed(out)
        return out


if __name__ == '__main__':
    # batch, map
    data = torch.ones(10, 32*32, dtype=torch.int64)
    data = data.permute(1, 0)
    from encoder import PostionalEncodingFactory
    encoding = PostionalEncodingFactory("1d", max_len=32*32)
    model = BracketNet(113, encoding)
    out = model(data)
    print(out.shape)
    model = BracketNet(113, encoding, mode="1_without_context")
    out = model(data)
    print(out.shape)
    model = BracketNet(113, encoding, mode="2_context_forget")
    out = model(data)
    print(out.shape)
    model = BracketNet(113, encoding, mode="3_context_forget")
    out = model(data)
    print(out.shape)
    model = BracketNet(113, encoding, mode="4_bracket_rule")
    out = model(data)
    print(out.shape)
    model = BracketNet(113, encoding, mode="5_without_context")
    out = model(data)
    print(out.shape)
    model = BracketNet(113, encoding, mode="6_vector_condition")
    out = model(data)
    print(out.shape)
