import os
import torch
import torch.nn as nn
import torch.optim as optim
from config.nn_config import NNCONFIG

def train_model(model, X_train, y_train, X_val, y_val, X_test, y_test, fold):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=NNCONFIG["lr"])
    epochs = NNCONFIG["epochs"]

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    x_train, x_val, x_test = X_train.float(), X_val.float(), X_test.float()
    y_train, y_val, y_test = y_train.float(), y_val.float(), y_test.float()

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val)
            val_loss = criterion(val_logits, y_val)

        scheduler.step(val_loss)  

        if epoch % 5 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"Fold {fold} | Epoch {epoch:03d} | "
                f"Train: {loss.item():.4f} | Val: {val_loss.item():.4f} | LR: {current_lr:.6f}"
            )

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_state = model.state_dict()

    model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        test_logits = model(x_test)
        test_loss = criterion(test_logits, y_test)

    print(f"Fold {fold} | Final Test Loss: {test_loss.item():.4f}")

    os.makedirs("model/checkpoints", exist_ok=True)
    save_path = f"model/checkpoints/fold_{fold}.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Saved model → {save_path}")

    return test_loss.item()
