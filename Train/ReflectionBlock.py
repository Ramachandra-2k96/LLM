import torch.nn as nn

class ReflectionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.ReLU(),
            nn.Linear(config.intermediate_size, config.hidden_size)
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = nn.LayerNorm(config.hidden_size)

    def forward(self, x):
        x_reflected = self.encoder(x)
        x = self.layer_norm(x + self.dropout(x_reflected))
        return x