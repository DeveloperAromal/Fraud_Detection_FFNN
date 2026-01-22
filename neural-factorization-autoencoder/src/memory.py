import torch

class Memory:
    def __init__(self, size=128):
        self.size = size
        self.memory = []

    def update(self, z):
        if len(self.memory) >= self.size:
            self.memory.pop(0)
        self.memory.append(z.detach())

    def score(self, z):
        if len(self.memory) == 0:
            return torch.tensor(0.0)
        mem = torch.stack(self.memory)
        return torch.min(torch.norm(mem - z, dim=1))
