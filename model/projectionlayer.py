import torch.nn as nn
import torch.nn.functional as F

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, projection_dim=4096, hidden_dim=2048, dropout_rate=0.1):
        super(ProjectionHead, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.hidden_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.output_layer = nn.Linear(hidden_dim, projection_dim)
        self.output_bn = nn.LayerNorm(projection_dim)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layer(x)
        x = self.output_layer(x)
        x = self.output_bn(x)
        x = F.normalize(x, p=2, dim=1)
        return x
