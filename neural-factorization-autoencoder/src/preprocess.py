
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/dataset.csv"


df = pd.read_csv(DATA_PATH)

df = df.drop(columns=["phone_no_m"])

labels = df["label"].values
df = df.drop(columns=["label"])

categorical_cols = [
    "voccalltype1", "voc_calltype1",
    "calltype_id_unique_x", "calltype_id_unique_y",
    "voc_hour_mode", "voc_day_mode",
    "calltype_2", "hour_mode", "day_mode",
    "month_ids", "flow_month"
]

for col in categorical_cols:
    df[col] = df[col].astype("category").cat.codes


numerical_cols = [c for c in df.columns if c not in categorical_cols]

scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])


X = torch.tensor(df.values, dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.int64)

torch.save(X, "data/X.pt")
torch.save(y, "data/y.pt")

print("Preprocessing done")
print("X shape:", X.shape)
print("y shape:", y.shape)
