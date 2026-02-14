import pandas as pd
import numpy as np

def load_and_clean(file_path, date_col):
    df = pd.read_csv(file_path)

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    df = df.ffill().bfill()

    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].clip(lower=0)

    return df

if __name__ == "__main__":
    RAW_PATH = "data/whoop_fitness.csv"
    DATE_COL = "date"

    cleaned = load_and_clean(RAW_PATH, DATE_COL)
    cleaned.to_csv("outputs/cleaned_whoop.csv", index=False)
    print("Saved cleaned_whoop.csv")
