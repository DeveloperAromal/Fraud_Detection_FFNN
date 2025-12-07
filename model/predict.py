import torch
import numpy as np

from nn.ffnn import FeedForwardNN
from config.nn_config import NNCONFIG


def predict(cdr):
    model = FeedForwardNN()
    model_path = NNCONFIG["model_path"]

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    new_cdr_in = np.array(
                          cdr["features"], 
                          dtype=float
                       )


    tensor_input = torch.tensor(new_cdr_in, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        pred = model(tensor_input)
        prob_fraud = float(pred[0][0])

    label = 1 if prob_fraud >= 0.5 else 0

    return {
              "probability of fraud": prob_fraud,
              "label": label
           }

