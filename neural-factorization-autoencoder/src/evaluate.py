
import torch
import numpy as np
from model_nfm import NFM
from model_ae import AutoEncoder
from memory import Memory
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Load data
X = torch.load("data/X.pt")
y = torch.load("data/y.pt")

num_features = X.shape[1]

# Train-test split (80/20, paper style)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Load trained models
nfm = NFM(num_features, embed_dim=64)
ae = AutoEncoder(input_dim=64)

nfm.load_state_dict(torch.load("data/nfm.pt"))
ae.load_state_dict(torch.load("data/ae.pt"))

nfm.eval()
ae.eval()

memory = Memory(size=128)

with torch.no_grad():
    for i in range(len(X_train)):
        if y_train[i] == 0:
            v = nfm(X_train[i].unsqueeze(0))
            z, _ = ae(v)
            memory.update(z.squeeze(0))

-------
scores = []

with torch.no_grad():
    for i in range(len(X_test)):
        v = nfm(X_test[i].unsqueeze(0))
        _, x_hat = ae(v)
        recon_error = torch.mean((x_hat - v) ** 2).item()
        scores.append(recon_error)

scores = np.array(scores)
y_true = y_test.numpy()


auc = roc_auc_score(y_true, scores)
print("AUC (score-based):", auc)


beta = np.percentile(scores, 95)
preds = (scores > beta).astype(int)

f1 = f1_score(y_true, preds)
tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()

tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

print("Threshold beta:", beta)
print("F1:", f1)
print("TPR:", tpr)
print("FPR:", fpr)
