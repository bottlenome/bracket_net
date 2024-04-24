import torch
import torch.nn as nn
import math


class DownLayer(nn.Module):
    def __init__(self, n_channel, out_channel=None):
        super().__init__()
        if out_channel is None:
            out_channel = 2 * n_channel
        self.conv = nn.Conv1d(n_channel, out_channel, kernel_size=2, stride=2)
        self.norm = nn.BatchNorm1d(out_channel)
        self.activate = nn.LeakyReLU()
    
    def forward(self, x):
        skip = x
        x = self.conv(x)
        x = self.norm(x)
        x = self.activate(x)
        return x, skip

class UpLayer(nn.Module):
    def __init__(self, n_channel, out_channel=None):
        super().__init__()
        if out_channel is None:
            out_channel = n_channel // 2
        self.deconv = nn.ConvTranspose1d(n_channel, out_channel, kernel_size=2, stride=2)
        self.norm = nn.BatchNorm1d(out_channel)
        self.activate = nn.LeakyReLU()
        self.pad = nn.ConstantPad1d((1, 0), 0)
    
    def forward(self, x, skip):
        x = self.deconv(x)
        x = self.norm(x)
        x = self.activate(x)
        x = self.pad(x)[:, :, :-1]
        # [batch, channels, length], [batch, channels, length]
        x = x + skip
        return x


class UpCasualUnet(nn.Module):
    def __init__(self, in_channels, seq_len):
        super().__init__()
        self.n_layers = math.ceil(math.log2(seq_len))
        self.map_len = int(math.pow(2, self.n_layers))
        self.input_pad = nn.ConstantPad1d((0, self.map_len - seq_len), 0)
        self.down_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.down_layers.append(DownLayer(in_channels, in_channels))
            # self.down_layers.append(DownLayer(in_channels))
            # in_channels *= 2
        self.up_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.up_layers.append(UpLayer(in_channels, in_channels))
            # self.up_layers.append(UpLayer(in_channels))
            # in_channels //= 2
        
    def forward(self, x):
        skips = []
        x = self.input_pad(x)
        for layer in self.down_layers:
            x, skip = layer(x)
            skips.append(skip)
        # [batch, channels, 1]
        assert(x.shape[2] == 1)
        for layer, skip in zip(self.up_layers, reversed(skips)):
            x = layer(x, skip)
        return x

if __name__ == '__main__':
    model = UpCasualUnet(16, 1023)
    assert(model.n_layers == 10)
    x = torch.randn(10, 16, 1023)
    y = model(x)
    print(y.shape)