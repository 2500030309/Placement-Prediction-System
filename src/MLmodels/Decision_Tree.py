import sys
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
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
        print("[INFO] Preprocessed data not found. Running preprocessing pipeline...", flush=True)
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
    """Tunes Decision Tree Classifier hyperparameters using 5-Fold Stratified CV."""
    base_model = DecisionTreeClassifier(random_state=42)
    
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [3, 5, 8],
        'min_samples_split': [2, 10],
        'min_samples_leaf': [1, 5]
    }
    
    print("\n[INFO] Performing Hyperparameter Tuning for Decision Tree (5-Fold CV)...", flush=True)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=1,
        verbose=0
    )
    
    grid_search.fit(X_train, Y_train)
    
    print(f"      Best Hyperparameters: {grid_search.best_params_}", flush=True)
    print(f"      Best 5-Fold CV ROC-AUC Score: {grid_search.best_score_:.4f}", flush=True)
    
    return grid_search.best_estimator_


def train_model(model, X_train, Y_train):
    """Fits Decision Tree model on training data."""
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
    
    print("\n==================================================", flush=True)
    print("        DECISION TREE EVALUATION METRICS          ", flush=True)
    print("==================================================", flush=True)
    print(f" Accuracy Score:             {acc:.4f} ({acc*100:.2f}%)", flush=True)
    print(f" Precision Score:            {prec:.4f}", flush=True)
    print(f" Recall Score:               {rec:.4f}", flush=True)
    print(f" F1-Score:                   {f1:.4f}", flush=True)
    print(f" ROC-AUC Score:              {roc_auc:.4f}", flush=True)
    print(f" Log Loss:                   {loss:.4f}", flush=True)
    print(f" Mean Squared Error (MSE):   {mse:.4f}", flush=True)
    print(f" Root Mean Sq Error (RMSE):  {rmse:.4f}", flush=True)
    print("\nConfusion Matrix:", flush=True)
    print(cm, flush=True)
    print("\nClassification Report:", flush=True)
    print(classification_report(Y_test, Y_pred, zero_division=0), flush=True)
    
    # Feature Importances
    if hasattr(model, "feature_importances_") and hasattr(X_test, "columns"):
        importances = pd.Series(model.feature_importances_, index=X_test.columns).sort_values(ascending=False)
        print("\nTop 5 Important Placement Features:", flush=True)
        for feat, score in importances.head(5).items():
            print(f" - {feat}: {score:.4f}", flush=True)
            
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


def plot_and_save_tree(model, feature_names, class_names=None, max_depth=3, show_plot=False):
    """
    Visualizes, saves, and prints the Decision Tree diagram for the Placement Prediction System.
    """
    if class_names is None:
        class_names = ["Not Placed", "Placed"]
    
    project_root = get_project_root()
    charts_dir = os.path.join(project_root, "app", "static", "charts")
    os.makedirs(charts_dir, exist_ok=True)

    # 1. Print Text-based Tree Structure in Terminal
    print("\n==================================================", flush=True)
    print(f"   PLACEMENT DECISION TREE STRUCTURE (DEPTH {max_depth})   ", flush=True)
    print("==================================================", flush=True)
    tree_rules = export_text(
        model,
        feature_names=list(feature_names),
        max_depth=max_depth
    )
    print(tree_rules, flush=True)
    print("==================================================", flush=True)
    
    # 2. Render and Save Visual Decision Tree Diagram
    plt.figure(figsize=(24, 12), dpi=300)
    plot_tree(
        model,
        feature_names=list(feature_names),
        class_names=class_names,
        filled=True,
        rounded=True,
        max_depth=max_depth,
        fontsize=9,
        precision=2
    )
    plt.title("Placement Prediction System - Decision Tree Diagram", fontsize=18, fontweight='bold', pad=20)
    plt.tight_layout()
    
    chart_path2 = os.path.join(charts_dir, "decision_tree.png")

    plt.savefig(chart_path1, bbox_inches='tight', dpi=300)
    plt.savefig(chart_path2, bbox_inches='tight', dpi=300)
    print(f"\n[INFO] Decision Tree Diagram successfully saved to:")
    print(f"       - {chart_path1}")
    print(f"       - {chart_path2}")

    if show_plot:
        try:
            plt.show()
        except Exception:
            pass
    plt.close()


def save_model(model):
    """Saves Decision Tree model artifact using relative project paths."""
    project_root = get_project_root()
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    model_path1 = os.path.join(models_dir, "Decision_Tree.pkl")
    model_path2 = os.path.join(models_dir, "decision_tree.pkl")
    
    joblib.dump(model, model_path1)
    joblib.dump(model, model_path2)
    
    print("\n[SUCCESS] Decision Tree Model Saved Successfully", flush=True)
    print(f"      Model Saved at: {model_path2}", flush=True)


if __name__ == "__main__":
    train_data, test_data = load_preprocessed_data()
    print(f"Training Data Shape: {train_data.shape} (80%)", flush=True)
    print(f"Testing Data Shape:  {test_data.shape} (20%)", flush=True)
    
    X_train, X_test, Y_train, Y_test = split_features_target(train_data, test_data)
    print(f"\nX_train Shape: {X_train.shape}", flush=True)
    print(f"X_test Shape:  {X_test.shape}", flush=True)

    tuned_model = create_and_tune_model(X_train, Y_train)
    print("\n[INFO] Evaluating Optimal Decision Tree Model...", flush=True)
    metrics = evaluate_model(tuned_model, X_test, Y_test)
    
    # Generate and save project decision tree diagram
    plot_and_save_tree(tuned_model, feature_names=X_train.columns, class_names=["Not Placed", "Placed"], max_depth=3)
    
    save_model(tuned_model)

