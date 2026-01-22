import torch.nn as nn

class NFM(nn.Module):
    def __init__(self, num_features, embed_dim=64):
        super().__init__()
        self.embedding = nn.Linear(num_features, embed_dim)

    def forward(self, x):
        return self.embedding(x)
