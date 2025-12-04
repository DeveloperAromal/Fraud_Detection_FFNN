# utils/dataset_preprocessor.py
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

SELECTED_FEATURES = [
    # call duration stats
    "call_dur_mean", "call_dur_median", "call_dur_max", "call_dur_min",
    "call_dur_var", "call_dur_skew", "call_dur_std",

    # hourly/daily behaviour
    "voc_hour_mode", "voc_hour_nunique",
    "voc_day_mode", "voc_day_nunique",
    "hour_mode", "hour_nunique",
    "day_mode", "day_nunique",

    # sms / call type features
    "sms_count", "sms_rate",
    "calltype_rate",

    # ARPU information
    "arpu_mean", "arpu_var", "arpu_min", "arpu_max",

    # flow / usage data
    "flow_mean", "flow_var", "flow_min", "flow_max",

    # target column
    "label",
]


def clean_dataframe(df: pd.DataFrame, missing_thresh: float = 0.95) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df

    # Drop duplicates & fully-empty columns
    df = df.drop_duplicates()
    df = df.dropna(axis="columns", how="all")

    # Keep only selected features 
    kept = [c for c in SELECTED_FEATURES if c in df.columns]
    df = df[kept]

    # Drop rows with missing values 
    df = df.dropna(axis=0, how="any")

    return df


def preprocess_csv(input_path: str, output_dir: Optional[str] = None, missing_thresh: float = 0.95) -> str:
    input_path = os.path.abspath(input_path)
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    df_clean = clean_dataframe(df, missing_thresh=missing_thresh)

    if output_dir is None:
        input_dir = os.path.dirname(input_path)
        parent = os.path.dirname(input_dir)
        output_dir = os.path.join(parent, "processed")

    os.makedirs(output_dir, exist_ok=True)

    out_name = os.path.basename(input_path)
    out_path = os.path.join(output_dir, out_name)
    df_clean.to_csv(out_path, index=False)

    return out_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m utils.dataset_preprocessor <input-csv> [output-dir]")
        sys.exit(1)

    input_csv = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else None
    out = preprocess_csv(input_csv, out_dir)

    print(f"Saved processed CSV to: {out}")
