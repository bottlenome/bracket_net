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


class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model: int, height: int, width: int, dropout=0.1):
        super(PositionalEncoding2D, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a 2D position encoding
        pe = torch.zeros(height, width, d_model)
        # Compute the positional encodings once in log space.
        y_position = torch.arange(0, height).unsqueeze(1)
        x_position = torch.arange(0, width).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, :, 0::2] = torch.sin(
            x_position * div_term) + torch.sin(y_position * div_term)
        pe[:, :, 1::2] = torch.cos(
            x_position * div_term) + torch.cos(y_position * div_term)
        # Shape: [1, d_model, height, width]
        pe = pe.unsqueeze(0).permute(0, 3, 1, 2)
        self.register_buffer('pe', pe)
        self.d_model = d_model

    def forward(self, x):
        x = x + self.pe[:, :, :x.size(2), :x.size(3)]
        # batch, c, h, w -> batch, c, seq
        x = x.view(x.shape[0], self.d_model, -1)
        x = self.dropout(x)
        # seq, batch, d_model
        return x.permute(2, 0, 1)


class PostionalEncodingFactory():
    def __init__(self, model_type, **kwargs):
        if model_type == "1d":
            self.model = PositionalEncoding
        elif model_type == "2d":
            self.model = PositionalEncoding2D
        else:
            print(f"invalid model_type:{model_type}")
            assert(False)
        self.model_type = model_type
        self.kwargs = kwargs

    def __call__(self, d_model, dropout):
        if self.model_type == "1d":
            return self.model(d_model=d_model,
                              max_len=self.kwargs["max_len"],
                              dropout=dropout)
        elif self.model_type == "2d":
            return self.model(d_model=d_model,
                              height=self.kwargs["height"],
                              width=self.kwargs["width"],
                              dropout=dropout)
        assert(False)


class GPT(nn.Module):
    def __init__(self, d_vocab, pos_encoder, d_model=128, nhead=4,
                 num_layers=6, dropout=0.0,
                 embed=None):
        super().__init__()
        if embed is None:
            # int(d_vocab+1) to float(d_model)
            self.embed = torch.nn.Embedding(d_vocab + 1, d_model)
        else:
            self.embed = embed
        self.pos_encoder = pos_encoder(d_model, dropout)
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
    encoding = PostionalEncodingFactory("1d", max_len=32*32)
    model = GPT(113, encoding, 32* 32)
    # batch, map
    data = torch.zeros(10, 32 * 32, dtype=torch.int64)
    # map, batch
    data = data.permute(1, 0)
    out = model(data)
    print(out.shape)
