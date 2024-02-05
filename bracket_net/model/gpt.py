import torch
import torch.nn as nn
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 10):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
                torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class GPT(nn.Module):
    def __init__(self, d_vocab, d_model=128, nhead=4,
                 num_layers=6, dropout=0.0):
        super().__init__()
        self.embed = torch.nn.Embedding(d_vocab + 1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = TransformerEncoderLayer(
                d_model, nhead,
                dim_feedforward=d_model*4, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(
                encoder_layer, num_layers, norm=None)
        self.unembed = torch.nn.Linear(d_model, d_vocab)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src = self.embed(src)
        src = self.pos_encoder(src)
        out = self.transformer_encoder(src)
        out = self.unembed(out)
        return out


if __name__ == '__main__':
    model = GPT(113)
