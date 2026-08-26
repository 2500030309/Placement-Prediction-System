import sys
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data.load_data import load_data, get_project_root
from src.data.preprocess import preprocess_pipeline

def load_regression_data(placed_only=True):
    """
    Loads dataset and prepares preprocessed feature matrices (X) 
    and continuous target vector (Y) for Salary Package regression.
    
    Parameters:
    - placed_only (bool): If True, filters dataset for placed students (Salary Package > 0)
                          which provides accurate continuous salary range modeling.
    """
    project_root = get_project_root()
    df = load_data()
    
    if "Salary Package" not in df.columns:
        raise ValueError("Target column 'Salary Package' not found in dataset.")
        
    if placed_only:
        df = df[df["Salary Package"] > 0].reset_index(drop=True)
        
    y = df["Salary Package"]
    
    # Preprocess features
    X_train_df, X_test_df = preprocess_pipeline(df)
    drop_cols = ["PlacementStatus", "Salary Package", "StudentID", "IsAnomaly"]
    feature_cols = [c for c in X_train_df.columns if c not in drop_cols]
    
    X_train = X_train_df[feature_cols]
    X_test = X_test_df[feature_cols]
    
    # Extract target using exact row indices produced by preprocess_pipeline
    Y_train = df.loc[X_train_df.index, "Salary Package"]
    Y_test = df.loc[X_test_df.index, "Salary Package"]
        
    return X_train, X_test, Y_train, Y_test

def create_and_tune_model(X_train, Y_train):
    """
    Tunes Linear / Ridge Regression hyperparameters using 5-Fold Cross-Validation.
    """
    print("\n[INFO] Performing Hyperparameter Tuning for Linear/Ridge Regression (5-Fold CV)...", flush=True)
    
    ridge = Ridge(random_state=42)
    param_grid = {
        'alpha': [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr']
    }
    
    grid_search = GridSearchCV(
        estimator=ridge,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, Y_train)
    
    best_rmse = np.sqrt(-grid_search.best_score_)
    print(f"      Best Hyperparameters: {grid_search.best_params_}", flush=True)
    print(f"      Best 5-Fold CV RMSE: {best_rmse:.4f}", flush=True)
    
    return grid_search.best_estimator_

def train_model(model, X_train, Y_train):
    """Fits Linear Regression model on training dataset."""
    model.fit(X_train, Y_train)
    return model

def evaluate_model(model, X_test, Y_test):
    """
    Evaluates Linear Regression performance metrics: MAE, MSE, RMSE, R2, and Adjusted R2.
    """
    Y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(Y_test, Y_pred)
    mse = mean_squared_error(Y_test, Y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(Y_test, Y_pred)
    
    n = len(Y_test)
    p = X_test.shape[1]
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1)) if (n - p - 1) > 0 else r2
    
    print("\n==================================================")
    print("       LINEAR REGRESSION EVALUATION METRICS       ")
    print("==================================================")
    print(f" Mean Absolute Error (MAE):     {mae:.4f} LPA")
    print(f" Mean Squared Error (MSE):      {mse:.4f}")
    print(f" Root Mean Sq Error (RMSE):     {rmse:.4f} LPA")
    print(f" R-Squared (R2) Score:         {r2:.4f}")
    print(f" Adjusted R2 Score:            {adj_r2:.4f}")
    print("==================================================")
    
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "adjusted_r2": adj_r2
    }

def save_model(model):
    """Saves Linear Regression model artifact to models/ directory."""
    project_root = get_project_root()
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, "linear_regression.pkl")
    joblib.dump(model, model_path)
    print("\n[SUCCESS] Linear Regression Model Saved Successfully")
    print(f"      Model Saved at: {model_path}")

if __name__ == "__main__":
    X_train, X_test, Y_train, Y_test = load_regression_data(placed_only=True)
    print(f"Training Data Features: {X_train.shape}")
    print(f"Testing Data Features:  {X_test.shape}")
    
    model = create_and_tune_model(X_train, Y_train)
    print("\n[INFO] Evaluating Optimal Linear Regression Model...")
    metrics = evaluate_model(model, X_test, Y_test)
    save_model(model)
