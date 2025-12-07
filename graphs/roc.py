import pickle
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from config.nn_config import NNCONFIG
from utils.get_data_loader import load_data
from nn.ffnn import FeedForwardNN

def roc():
    
    model_path = NNCONFIG["model_path"]
    
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    model = FeedForwardNN()
    
    model.load_state_dict(torch.load(model_path))
    model.eval() 
    
    with torch.no_grad():
        X_test = X_test.float() 
        y_pred_probs = model(X_test).numpy().flatten()
        y_true = y_test.numpy().flatten()
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    print(f"AUC: {roc_auc:.4f}")
    
    
    optimal_idx = (tpr - fpr).argmax()
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Fraud Detection")
    plt.legend()
    plt.show()