import pandas as pd
import torch
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config.nn_config import NNCONFIG

def load_data():
    dataset_path = os.path.abspath("data/processed/dataset.csv")
    df = pd.read_csv(dataset_path)  

    X = df.drop(columns=["label"])
    y = df["label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X_tensor, y_tensor, test_size=0.2, random_state=42
    )

    NNCONFIG["input_size"] = X_train.shape[1]

    return X_train, X_test, y_train, y_test
