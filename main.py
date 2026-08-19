import sys
import os

from src.data.load_data import load_data, get_summary
from src.data.eda import basic_eda, univariate, bivariate, multivariate
from src.data.preprocess import preprocess_pipeline
from src.MLmodels.logistcs_regression import (
    load_preprocessed_data,
    split_features_target,
    create_and_tune_model,
    evaluate_model,
    save_model
)

def main():
    print("==================================================")
    print("      PLACEMENT PREDICTION SYSTEM PIPELINE        ")
    print("==================================================\n")
    
    # 1. Load Data
    print("[1/4] Loading dataset...")
    df = load_data()
    summary = get_summary(df)
    print(f"      Rows: {summary['rows']}, Columns: {summary['columns']}, Target: {summary['target']}")
    
    # 2. Run EDA
    print("\n[2/4] Generating Exploratory Data Analysis (EDA) visualizations...")
    basic_eda(df)
    univariate(df)
    bivariate(df)
    multivariate(df)
    print("      EDA charts saved to app/static/charts/")
    
    # 3. Run Preprocessing
    print("\n[3/4] Executing data preprocessing pipeline...")
    X_train_df, X_test_df = preprocess_pipeline(df)
    
    # 4. Model Training & Evaluation
    print("\n[4/4] Training and evaluating Logistic Regression model...")
    train_data, test_data = load_preprocessed_data()
    X_train, X_test, Y_train, Y_test = split_features_target(train_data, test_data)
    model = create_and_tune_model(X_train, Y_train)
    evaluate_model(model, X_test, Y_test)
    save_model(model)
    
    print("\n==================================================")
    print("       PIPELINE EXECUTED SUCCESSFULLY!            ")
    print("==================================================")

if __name__ == '__main__':
    main()


