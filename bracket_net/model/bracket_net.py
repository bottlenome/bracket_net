import torch
import torch.nn as nn
from .lie_func import LieFuncFactory


class BracketFunc(nn.Module):
    def __init__(self, d_model=128, n_head=4, mode="base"):
        super().__init__()
        assert(d_model % n_head == 0)

        dim = int(d_model / n_head)
        self.bracket_products = nn.ModuleList()
        for _ in range(n_head):
            self.bracket_products.append(nn.Linear(dim*2, dim))

        def bracket(a, b, i):
            return self.bracket_products[i](torch.cat([a, b], dim=1))
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
            return out[:-1]


        context = self.lie_func.initialize_context(src)
        ret = []
        for i in range(src.shape[0]):
            context, r = self.lie_func(context, src[i])
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
        self.activate = nn.GELU()

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
        # src = self.activate(self.map(src))
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
