import os
import torch
import numpy as np

from nn.ffnn import FeedForwardNN
from config.nn_config import NNCONFIG


def predict():
    model = FeedForwardNN()
    model_path = NNCONFIG["model_path"]

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    new_user = np.array([
        77.25, 38, 615, 1, 10552.65, 2.65, 102.72,
        10, 16, 4, 31, 10, 24, 1, 31,
        4242, 11.65, 0.99,
        63.93, 337.14, 34.1, 79.3,
        30.45, 26689.37, 0.000074, 3611.52
    ], dtype=float)


    tensor_input = torch.tensor(new_user, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        pred = model(tensor_input)
        prob_fraud = float(pred[0][0])

    label = 1 if prob_fraud >= 0.5 else 0

    print("Fraud Probability:", prob_fraud)
    print("Predicted Label:", label)

