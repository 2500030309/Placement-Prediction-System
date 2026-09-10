import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Get project root directory
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from src.data.load_data import load_data
from src.data.preprocess import preprocess_pipeline


def load_regression_data(placed_only=True):
    """
    Loads dataset and prepares features (X) and salary target (Y).
    Filters for placed students (Salary Package > 0) for continuous salary regression.
    """
    df = load_data()
    
    if placed_only:
        df = df[df["Salary Package"] > 0].reset_index(drop=True)
        
    X_train_df, X_test_df = preprocess_pipeline(df)
    
    drop_cols = ["PlacementStatus", "Salary Package", "StudentID", "IsAnomaly"]
    feature_cols = [c for c in X_train_df.columns if c not in drop_cols]
    
    X_train = X_train_df[feature_cols]
    X_test = X_test_df[feature_cols]
    
    Y_train = df.loc[X_train_df.index, "Salary Package"]
    Y_test = df.loc[X_test_df.index, "Salary Package"]
        
    return X_train, X_test, Y_train, Y_test


def create_model(alpha=1.0):
    """Initializes Linear / Ridge Regression model."""
    return Ridge(alpha=alpha, random_state=42)


def train_model(model, X_train, Y_train):
    """Fits Linear Regression model on training dataset."""
    print("[INFO] Training Linear Regression model...", flush=True)
    model.fit(X_train, Y_train)
    return model


def create_and_tune_model(X_train, Y_train):
    """Creates and fits Ridge Regression model (used in pipeline)."""
    model = create_model()
    return train_model(model, X_train, Y_train)


def evaluate_model(model, X_test, Y_test):
    """Evaluates regression metrics: MAE, MSE, RMSE, and R2 Score."""
    Y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(Y_test, Y_pred)
    mse = mean_squared_error(Y_test, Y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(Y_test, Y_pred)

    print("\n" + "=" * 50)
    print("       LINEAR REGRESSION EVALUATION METRICS       ")
    print("=" * 50)
    print(f" Mean Absolute Error (MAE): {mae:.4f} LPA")
    print(f" Mean Squared Error (MSE):  {mse:.4f}")
    print(f" Root Mean Sq Error (RMSE): {rmse:.4f} LPA")
    print(f" R-Squared (R2) Score:      {r2:.4f}")
    print("=" * 50)

    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}


def save_model(model):
    """Saves trained model to disk."""
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "linear_regression.pkl")
    joblib.dump(model, model_path)
    print(f"[SUCCESS] Linear Regression model saved to: {model_path}")


def main():
    # 1. Load Data
    print("[1/4] Loading regression dataset...")
    X_train, X_test, Y_train, Y_test = load_regression_data(placed_only=True)

    # 2. Train Model
    print("[2/4] Training Linear Regression model...")
    model = create_model()
    train_model(model, X_train, Y_train)

    # 3. Evaluate Model
    print("[3/4] Evaluating model performance...")
    evaluate_model(model, X_test, Y_test)

    # 4. Save Model
    print("[4/4] Saving model...")
    save_model(model)
    print("\nAll tasks completed successfully!")


if __name__ == "__main__":
    main()
