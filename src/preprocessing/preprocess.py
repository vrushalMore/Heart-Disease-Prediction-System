import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def load_data(filepath):
    return pd.read_csv(filepath)

def fix_invalid_values(df):
    df = df[df['ca'] < 4]
    df = df[df['thal'] > 0]
    return df

def save_data(df, filepath):
    df.to_csv(filepath, index=False)

def preprocess_pipeline(input_path, output_path):
    df = load_data(input_path)
    df = fix_invalid_values(df)
    save_data(df, output_path)

preprocess_pipeline("../../data/raw/heart.csv", "../../data/processed/heart_cleaned.csv")
