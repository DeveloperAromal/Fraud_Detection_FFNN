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

    new_user = np.array(
                          [
                              127.18806744487678, 37.0, 3256.0, 1.0, 95010.5788704162,
                              5.444171923610376, 308.23786086465145, 17.0, 19.0, 10.0,
                              31.0, 10.0, 21.0, 28.0, 31.0, 1066.0, 6.345238095238095,
                              0.9690431519699813, 65.19, 148.67080000000004, 59.0,
                              94.45, 9.469104009734949, 3000.946493058878,
                              7.43865966796875e-05, 889.6195192337036
                          ], 
                          dtype=float
                       )


    tensor_input = torch.tensor(new_user, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        pred = model(tensor_input)
        prob_fraud = float(pred[0][0])

    label = 1 if prob_fraud >= 0.5 else 0

    print("Fraud Probability:", prob_fraud)
    print("Predicted Label:", label)

