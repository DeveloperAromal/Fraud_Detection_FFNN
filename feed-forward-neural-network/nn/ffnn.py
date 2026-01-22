import torch.nn as nn
from config.nn_config import NNCONFIG


class FeedForwardNN(nn.Module):
    
    def __init__(self):
        super(FeedForwardNN, self).__init__()
        
        layers = []
        input_dim = NNCONFIG["input_size"]
        
        
        for hidden in NNCONFIG["hidden_layers"]:
            
            layers.append(nn.Linear(input_dim, hidden))
            layers.append(nn.BatchNorm1d(hidden))
            
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            
            input_dim = hidden
        
        layers.append(nn.Linear(input_dim, NNCONFIG["output_size"]))
        layers.append(nn.Sigmoid())
        
        self.net = nn.Sequential(*layers)
        
            
        
    def forward(self, x):
        return self.net(x)
    
    