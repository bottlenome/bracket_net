import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, input_size=24, embed_size=128, hidden_size=128,
                 num_hidden=4, output_size=1, dropout_prob=0.1,
                 enable_positional_embedding=True, input_num=None):
        super().__init__()

        # 位置エンベディングの定義
        self.enable_positional_embedding = enable_positional_embedding
        if enable_positional_embedding:
            self.position_embedding = nn.Embedding(input_size, embed_size)

        # 入力層の定義
        if enable_positional_embedding:
            self.input_layer = nn.Linear(2 * embed_size * input_size, hidden_size)
        else:
            if input_num is not None:
                assert(hidden_size % 8 == 0)
                self.input_layer = nn.Sequential(
                    nn.Embedding(input_num, int(hidden_size / 8)),
                    nn.Flatten(),
                    nn.Linear(hidden_size * 3, hidden_size)
                )
            else:
                self.input_layer = nn.Linear(input_size, hidden_size)
        self.input_batch_norm = nn.BatchNorm1d(hidden_size)

        self.num_hidden = num_hidden
        # 隠れ層の定義
        self.hidden_layers = nn.ModuleList()
        self.hidden_batch_norms = nn.ModuleList()
        for _ in range(self.num_hidden - 1):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
            self.hidden_batch_norms.append(nn.BatchNorm1d(hidden_size))

        # 出力層の定義
        self.output_layer = nn.Linear(hidden_size, output_size)

        # Dropout層の定義
        self.dropout = nn.Dropout(p=dropout_prob)

        # 活性化関数の定義
        self.relu = nn.ReLU()


    def forward(self, x):
        batch_size = x.size(0)

        # 位置エンベディングの取得
        if self.enable_positional_embedding:
            position_indices = torch.arange(0, x.size(1), device=x.device).unsqueeze(0).repeat(batch_size, 1)
            position_embeds = self.position_embedding(position_indices)

            # 入力特徴量と位置エンベディングの結合
            x = x.unsqueeze(2).expand(-1, -1, position_embeds.size(2))
            x = torch.cat((x, position_embeds), dim=2)
            x = x.view(batch_size, -1)

        # 入力層を通過
        x = self.input_batch_norm(self.input_layer(x))
        x = self.dropout(self.relu(x))

        # 隠れ層を通過
        for i in range(len(self.hidden_layers)):
            identity = x
            x = self.hidden_batch_norms[i](self.hidden_layers[i](x))
            x = identity + self.relu(x)
            x = self.relu(x)
            x = self.dropout(x)

        # 出力層を通過
        x = self.output_layer(x)
        return x


if __name__ == '__main__':
    model = Linear()
    data = torch.zeros(10, 24, dtype=torch.int64)
    out = model(data)
