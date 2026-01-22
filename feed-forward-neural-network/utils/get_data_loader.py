import pandas as pd
import torch
import os

from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from config.nn_config import NNCONFIG


def load_data(k=4):
    dataset_path = os.path.abspath("data/processed/dataset.csv")
    df = pd.read_csv(dataset_path)

    X = df.drop(columns=["label"])
    y = df["label"]

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_test = X_test.values
    y_test = y_test.values.reshape(-1, 1)

    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    for fold, (t_idx, v_idx) in enumerate(kf.split(X_trainval), 1):

        X_train = X_trainval.iloc[t_idx]
        X_val   = X_trainval.iloc[v_idx]
        y_train = y_trainval.iloc[t_idx]
        y_val   = y_trainval.iloc[v_idx]

        scaler = StandardScaler()
        scaler.fit(X_train)

        X_train = scaler.transform(X_train)
        X_val   = scaler.transform(X_val)
        X_test_ = scaler.transform(X_test)  

        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_val   = torch.tensor(X_val, dtype=torch.float32)
        X_test_ = torch.tensor(X_test_, dtype=torch.float32)

        y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
        y_val   = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
        y_test_ = torch.tensor(y_test, dtype=torch.float32)

        NNCONFIG["input_size"] = X_train.shape[1]

        yield fold, X_train, X_val, X_test_, y_train, y_val, y_test_
