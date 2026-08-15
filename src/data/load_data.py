import os
import pandas as pd

def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_data():
    project_root = get_project_root()
    data_path = os.path.join(project_root, "data", "placement_data.csv")
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        # Fallback search in working directory or environment
        alt_path = os.path.join(os.getcwd(), "data", "placement_data.csv")
        if os.path.exists(alt_path):
            df = pd.read_csv(alt_path)
        else:
            raise FileNotFoundError(f"placement_data.csv not found at '{data_path}'. Please ensure the dataset file exists in the data/ directory.")
    return df

def get_summary(df):
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "target": "PlacementStatus"
    }

if __name__ == "__main__":
    df = load_data()
    print(get_summary(df))
    print(df.head())

