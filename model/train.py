import os
import torch
import pickle
import torch.nn as nn
import torch.optim as optim
from config.nn_config import NNCONFIG

def train_model(model, X_train, y_train, X_val, y_val, X_test, y_test):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=NNCONFIG["lr"])
    epochs = NNCONFIG["epochs"]

    x_train = X_train.float()
    y_train = y_train.float() 
    
    x_val = X_val.float()
    y_val = y_val.float() 
    
    x_test = X_test.float()
    y_test = y_test.float()

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val)
            val_loss = criterion(val_pred, y_val)

        if epoch % 5 == 0:
            print(f"Epoch {epoch} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}")

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_state = model.state_dict()

    model.load_state_dict(best_state)

    with torch.no_grad():
        test_pred = model(x_test)
        test_loss = criterion(test_pred, y_test)

    print(f"\nFinal Test Loss: {test_loss.item():.4f}")

    os.makedirs("model/checkpoints", exist_ok=True)
    torch.save(model.state_dict(), NNCONFIG["model_path"])
    print(f"Final Model saved to: {NNCONFIG['model_path']}")

