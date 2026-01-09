import torch
import numpy as np

from nn.ffnn import FeedForwardNN
from config.nn_config import NNCONFIG

model = FeedForwardNN()
model_path = NNCONFIG["model_path"]

model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

EXPECTED_FEATURES = NNCONFIG["input_size"]  # add this to config


def predict(cdr: dict):
    features = cdr.get("features")

    if not isinstance(features, list):
        raise ValueError("features must be a list of floats")

    if len(features) != EXPECTED_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_FEATURES} features, got {len(features)}"
        )


    new_cdr_in = np.array(features, dtype=np.float32)
    tensor_input = torch.tensor(new_cdr_in).unsqueeze(0)

 
    with torch.no_grad():
        output = model(tensor_input)

        prob_fraud = torch.sigmoid(output)[0][0].item()


    label = 1 if prob_fraud >= 0.5 else 0
    status = "Fraud" if label == 1 else  "Not Fraud"

    return {
        "probability_of_fraud": round(prob_fraud, 6),
        "label": label,
        "status": status
    }
