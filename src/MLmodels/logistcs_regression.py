import sys
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
    classification_report,
    mean_squared_error
)
from sklearn.model_selection import GridSearchCV

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data.load_data import get_project_root

def load_preprocessed_data():
    """Loads 80/20 preprocessed training and testing datasets using relative paths."""
    project_root = get_project_root()
    train_path = os.path.join(project_root, "data", "preprocessed_train.csv")
    test_path = os.path.join(project_root, "data", "preprocessed_test.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        from src.data.preprocess import preprocess_pipeline
        print("[INFO] Preprocessed data not found. Running preprocessing pipeline...")
        train_data, test_data = preprocess_pipeline()
    else:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        
    return train_data, test_data

def split_features_target(train_data, test_data, target_col="PlacementStatus"):
    """Separates feature matrix (X) and target vector (Y)."""
    feature_cols = [c for c in train_data.columns if c != target_col]
    X_train = train_data[feature_cols]
    X_test = test_data[feature_cols]
    Y_train = train_data[target_col]
    Y_test = test_data[target_col]
    return X_train, X_test, Y_train, Y_test

def create_and_tune_model(X_train, Y_train):
    """Tunes Logistic Regression model hyperparameters using 5-Fold Stratified CV."""
    base_model = LogisticRegression(random_state=42, max_iter=1000)
    
    param_grid = {
        'C': [0.01, 0.1, 1.0, 10.0],
        'solver': ['lbfgs', 'liblinear'],
        'class_weight': [None, 'balanced']
    }
    
    print("\n[INFO] Performing Hyperparameter Tuning for Logistic Regression (5-Fold CV)...", flush=True)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=1,
        verbose=1
    )
    
    grid_search.fit(X_train, Y_train)
    
    print(f"      Best Hyperparameters: {grid_search.best_params_}", flush=True)
    print(f"      Best 5-Fold CV ROC-AUC Score: {grid_search.best_score_:.4f}", flush=True)
    
    return grid_search.best_estimator_

def train_model(model, X_train, Y_train):
    """Fits model on training data."""
    model.fit(X_train, Y_train)
    return model

def evaluate_model(model, X_test, Y_test):
    """Evaluates model performance with Accuracy, Precision, Recall, F1, ROC-AUC, Log Loss, MSE, and RMSE."""
    Y_pred = model.predict(X_test)
    
    if hasattr(model, "predict_proba"):
        Y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        Y_pred_proba = Y_pred
    
    acc = accuracy_score(Y_test, Y_pred)
    prec = precision_score(Y_test, Y_pred, zero_division=0)
    rec = recall_score(Y_test, Y_pred, zero_division=0)
    f1 = f1_score(Y_test, Y_pred, zero_division=0)
    roc_auc = roc_auc_score(Y_test, Y_pred_proba) if len(np.unique(Y_test)) > 1 else 0.0
    loss = log_loss(Y_test, Y_pred_proba) if len(np.unique(Y_test)) > 1 else 0.0
    cm = confusion_matrix(Y_test, Y_pred)
    
    mse = mean_squared_error(Y_test, Y_pred_proba)
    rmse = np.sqrt(mse)
    
    print("\n==================================================")
    print("      LOGISTIC REGRESSION EVALUATION METRICS       ")
    print("==================================================")
    print(f" Accuracy Score:             {acc:.4f} ({acc*100:.2f}%)")
    print(f" Precision Score:            {prec:.4f}")
    print(f" Recall Score:               {rec:.4f}")
    print(f" F1-Score:                   {f1:.4f}")
    print(f" ROC-AUC Score:              {roc_auc:.4f}")
    print(f" Log Loss:                   {loss:.4f}")
    print(f" Mean Squared Error (MSE):   {mse:.4f}")
    print(f" Root Mean Sq Error (RMSE):  {rmse:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(Y_test, Y_pred, zero_division=0))
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "log_loss": loss,
        "mse": mse,
        "rmse": rmse
    }

def save_model(model):
    """Saves Logistic Regression model artifact using relative project paths."""
    project_root = get_project_root()
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    model_path1 = os.path.join(models_dir, "logistcs_regression.pkl")
    model_path2 = os.path.join(models_dir, "logistic_regression.pkl")
    
    joblib.dump(model, model_path1)
    joblib.dump(model, model_path2)
    
    print("\n[SUCCESS] Logistic Regression Model Saved Successfully")
    print(f"      Model Saved at: {model_path2}")

if __name__ == "__main__":
    train_data, test_data = load_preprocessed_data()
    print(f"Training Data Shape: {train_data.shape} (80%)")
    print(f"Testing Data Shape:  {test_data.shape} (20%)")
    
    X_train, X_test, Y_train, Y_test = split_features_target(train_data, test_data)
    print(f"\nX_train Shape: {X_train.shape}")
    print(f"X_test Shape:  {X_test.shape}")

    tuned_model = create_and_tune_model(X_train, Y_train)
    print("\n[INFO] Evaluating Optimal Logistic Regression Model...")
    metrics = evaluate_model(tuned_model, X_test, Y_test)
    save_model(tuned_model)