import sys
import os

from src.data.load_data import load_data, get_summary
from src.data.eda import basic_eda, univariate, bivariate, multivariate
from src.data.preprocess import preprocess_pipeline

def main():
    print("==================================================")
    print("      PLACEMENT PREDICTION SYSTEM PIPELINE        ")
    print("==================================================\n")
    
    # 1. Load Data
    print("[1/3] Loading dataset...")
    df = load_data()
    summary = get_summary(df)
    print(f"      Rows: {summary['rows']}, Columns: {summary['columns']}, Target: {summary['target']}")
    
    # 2. Run EDA
    print("\n[2/3] Generating Exploratory Data Analysis (EDA) visualizations...")
    basic_eda(df)
    univariate(df)
    bivariate(df)
    multivariate(df)
    print("      EDA charts saved to app/static/charts/")
    
    # 3. Run Preprocessing
    print("\n[3/3] Executing data preprocessing pipeline...")
    X_train, X_test = preprocess_pipeline(df)
    
    print("\n==================================================")
    print("       PIPELINE EXECUTED SUCCESSFULLY!            ")
    print("==================================================")

if __name__ == '__main__':
    main()

