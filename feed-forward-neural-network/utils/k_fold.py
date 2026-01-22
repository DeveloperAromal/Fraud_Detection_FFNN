import pandas as pd
from sklearn.model_selection import KFold
import os



def split_dataset(k=4):
    
    dataset_path = os.path.abspath("data/processed/dataset.csv")
    
    
    df = pd.read_csv(dataset_path)
    
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    X = df.drop(columns=["label"]) 
    y = df["label"]
    
    
    folds = []
    
    for fold, (t_idx, v_idx) in enumerate(kf.split(X), 1):
        
        folds.append(
                        {
                            "fold": fold,
                            "X_train": X.iloc[t_idx],
                            "y_train": y.iloc[t_idx],
                            "X_val": X.iloc[v_idx],
                            "y_val": y.iloc[v_idx]
                        }
                    )
        
    return folds
        

        
        