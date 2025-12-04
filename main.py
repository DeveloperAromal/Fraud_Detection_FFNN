import pandas as pd

from utils.dataset_preprocessor import preprocess_csv

if __name__ == "__main__":
    input_csv = "data/raw/dataset.csv"
    output_path = preprocess_csv(input_csv)

    print("Processed file saved to:", output_path)
