import pandas as pd
import numpy

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    
    cols_to_remove = [
                        "phone_no_m", "opposite_count", "opposite_unique", "voccalltype1", "imeis",
                        "voc_calltype1", "calltype_id_unique_x", "phone2opposite_mean",
                        "phone2opposite_median", "phone2opposite_max", "phone2opposite_min",
                        "phone2opposite_var", "phone2opposite_skew", "phone2opposite_sem",
                        "phone2opposite_std", "phone2opposite_quantile", "phone2oppo_sum_mean",
                        "phone2oppo_sum_median", "phone2oppo_sum_max", "phone2oppo_sum_min",
                        "phone2oppo_sum_var", "phone2oppo_sum_skew", "phone2oppo_sum_sem",
                        "phone2oppo_sum_std", "phone2oppo_sum_quantile", "call_dur_mean",
                        "call_dur_median", "call_dur_max", "call_dur_min", "call_dur_var",
                        "call_dur_skew", "call_dur_sem", "call_dur_std", "call_dur_quantile",
                        "city_name_nunique", "county_name_nunique", "calltype_id_unique_y",
                        "voc_hour_mode", "voc_hour_mode_count", "voc_hour_nunique", "voc_day_mode",
                        "voc_day_mode_count", "voc_day_nunique", "sms_count", "sms_nunique",
                        "sms_rate", "calltype_2", "calltype_rate", "hour_mode", "hour_mode_count",
                        "hour_nunique", "day_mode", "day_mode_count", "day_nunique", "busi_count",
                        "flow_mean", "flow_median", "flow_min", "flow_max", "flow_var", "flow_sum",
                        "month_ids", "flow_month", "arpu_mean", "arpu_var", "arpu_max", "arpu_min",
                        "arpu_median", "arpu_sum", "arpu_skew", "arpu_sem", "arpu_quantile"
                    ]

    df = df.drop_duplicates()
    
    
    df = df.dropna(axis="columns", how="all")
    
    df = df.drop(columns="phone")
    