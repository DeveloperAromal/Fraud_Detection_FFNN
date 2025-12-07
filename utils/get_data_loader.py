import pandas as pd
import torch
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config.nn_config import NNCONFIG

from utils.split_data import splited_dataset

def load_data():
    dataset_path = os.path.abspath("data/processed/dataset.csv")
    df = pd.read_csv(dataset_path)  

    print(df.shape)

    
    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, X_val, X_test, y_train, y_val, y_test = splited_dataset(X, y)

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)
    
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val   = torch.tensor(X_val, dtype=torch.float32)
    X_test  = torch.tensor(X_test, dtype=torch.float32)
    
    y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    y_val   = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
    y_test  = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)


    NNCONFIG["input_size"] = X_train.shape[1]

    return X_train, X_val, X_test, y_train, y_val, y_test
