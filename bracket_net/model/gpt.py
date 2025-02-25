import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class GPTBlock(nn.Module):
    def __init__(self, d_model, n_head, num_layers, dropout, norm=None, batch_first=False):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(
                d_model, n_head,
                dim_feedforward=d_model*4, dropout=dropout, batch_first=batch_first)
        self.transformer_encoder = TransformerEncoder(
                encoder_layer, num_layers, norm=norm)
        self.batch_first = batch_first

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        if self.batch_first:
            seq_len = src.size(1)
        else:
            seq_len = src.size(0)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(src.device)
        out = self.transformer_encoder(src, mask=mask, is_causal=True)
        return out


class GPT(nn.Module):
    def __init__(self, d_vocab, pos_encoder, d_model=128, n_head=4,
                 num_layers=6, dropout=0.0,
                 embed=None, batch_first=False):
        super().__init__()
        if embed is None:
            # int(d_vocab+1) to float(d_model)
            self.embed = torch.nn.Embedding(d_vocab + 1, d_model)
        else:
            self.embed = embed
        self.pos_encoder = pos_encoder(d_model, dropout)
        self.gpt_block = GPTBlock(d_model, n_head, num_layers, dropout,
                                  batch_first=batch_first)
        self.unembed = torch.nn.Linear(d_model, d_vocab)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src = self.embed(src)
        src = self.pos_encoder(src)
        out = self.gpt_block(src)
        out = self.unembed(out)
        return out


if __name__ == '__main__':
    from encoder import PostionalEncodingFactory
    encoding = PostionalEncodingFactory("1d", max_len=32*32)
    model = GPT(113, encoding)
    # batch, map
    data = torch.zeros(10, 32 * 32, dtype=torch.int64)
    # map, batch
    data = data.permute(1, 0)
    out = model(data)
    print(out.shape)

    model = GPTBlock(128, 4, 6, 0.1)
    data = torch.zeros(32 * 32, 10, 128)
    out = model(data)
    print(out.shape)
