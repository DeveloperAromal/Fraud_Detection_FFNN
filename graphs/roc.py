import pickle
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from config.nn_config import NNCONFIG


def roc():
    
    pkl_path = NNCONFIG["model_path"].replace(".pth", ".pkl")
    
    x_test = torch.tensor(np.random.rand(100, 26), dtype=torch.float32)
    y_test = torch.tensor(np.random.randint(0, 2, size=(100, 1)), dtype=torch.float32)

    
    
    with open(pkl_path, "rb") as f:
        model = pickle.load(f)

        model.eval() 
        
    
    with torch.no_grad():
        y_pred_probs = model(x_test).numpy().flatten()
        y_true = y_test.numpy().flatten()
        
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    print(f"AUC: {roc_auc:.4f}")
    
    optimal_idx = (tpr - fpr).argmax()
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Fraud Detection')
    plt.legend(loc='lower right')
    plt.show()