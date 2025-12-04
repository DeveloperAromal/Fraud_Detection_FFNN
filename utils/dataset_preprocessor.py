import os
from typing import Optional

import pandas as pd

SELECTED_FEATURES = [
    "call_dur_mean", "call_dur_median", "call_dur_max", "call_dur_min",
    "call_dur_var", "call_dur_skew", "call_dur_std",

    "voc_hour_mode", "voc_hour_nunique",
    "voc_day_mode", "voc_day_nunique",
    "hour_mode", "hour_nunique",
    "day_mode", "day_nunique",

    "sms_count", "sms_rate",
    "calltype_rate",

    "arpu_mean", "arpu_var", "arpu_min", "arpu_max",

    "flow_mean", "flow_var", "flow_min", "flow_max",

    "label",
]


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df

    df = df.drop_duplicates()
    df = df.dropna(axis="columns", how="all")

    kept = [c for c in SELECTED_FEATURES if c in df.columns]
    df = df[kept]

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


