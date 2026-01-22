# src/train.py
import torch
from torch.optim import Adam
from model_nfm import NFM
from model_ae import AutoEncoder

# Load data
X = torch.load("data/X.pt")

num_features = X.shape[1]

# Models
nfm = NFM(num_features, embed_dim=64)
ae = AutoEncoder(input_dim=64)

optimizer = Adam(
    list(nfm.parameters()) + list(ae.parameters()),
    lr=1e-3
)

loss_fn = torch.nn.MSELoss()
epochs = 500

for epoch in range(epochs):
    optimizer.zero_grad()

    v = nfm(X)
    z, x_hat = ae(v)

    loss = loss_fn(x_hat, v)
    loss.backward()

    # Gradient clipping (stability)
    torch.nn.utils.clip_grad_norm_(
        list(nfm.parameters()) + list(ae.parameters()),
        max_norm=5.0
    )

    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch} | Loss {loss.item():.6f}")

# Save models
torch.save(nfm.state_dict(), "data/nfm.pt")
torch.save(ae.state_dict(), "data/ae.pt")

print("Training completed")
