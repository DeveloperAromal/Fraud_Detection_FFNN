import os
import torch
import torch.nn as nn
import torch.optim as optim
from config.nn_config import NNCONFIG


def train_model(model, x_train, y_train, x_test, y_test):
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=NNCONFIG["lr"])
    
    for epoch in range(model.parameter(),lr=NNCONFIG["lr"]):
        
        model.train()
        
        y_pred = model(x_test)  
        loss = optimizer(y_test, y_train)
                
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        if epoch % 5 == 0 :
            model.eval()
            test_pred = model(x_test)
            test_loss = criterion(test_pred, y_test)
            print(f"Epoch {epoch} | Train Loss: {loss.item():.4f} | Test Loss: {test_loss.item():.4f}")
            
            
        os.makedirs("model/checkpoints", exist_ok=True)
        torch.save(model.state_dict(), NNCONFIG["model_path"])