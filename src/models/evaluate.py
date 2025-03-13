import pandas as pd
import pickle
from pycaret.classification import setup, compare_models, finalize_model, save_model, pull

def train_and_save_model(data_path, model_path, evaluation_path):
    df = pd.read_csv(data_path)
    setup(data=df, target='target', verbose=False, session_id=42)
    
    best_model = compare_models()
    final_model = finalize_model(best_model)
    
    save_model(final_model, model_path)

    metrics = pull().iloc[0]  
    with open(evaluation_path, "w") as f:
        f.write(metrics.to_string())

data_path = "../../data/raw/heart.csv"
model_path = "../../src/models/best_model"
evaluation_path = "../../src/models/evaluation_results.txt"

train_and_save_model(data_path, model_path, evaluation_path)
