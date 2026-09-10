import os
import sys
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Get project root directory
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)


def load_preprocessed_data():
    """Loads preprocessed train and test CSV files."""
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
    """Separates features (X) and target label (Y)."""
    feature_cols = [c for c in train_data.columns if c != target_col]
    X_train = train_data[feature_cols]
    X_test = test_data[feature_cols]
    Y_train = train_data[target_col]
    Y_test = test_data[target_col]
    return X_train, X_test, Y_train, Y_test


def create_model(C=1.0, max_iter=1000, random_state=42):
    """Initializes Logistic Regression Classifier."""
    return LogisticRegression(
        C=C,
        solver='lbfgs',
        max_iter=max_iter,
        random_state=random_state
    )


def train_model(model, X_train, Y_train):
    """Trains the Logistic Regression model on training data."""
    print("[INFO] Training Logistic Regression model...", flush=True)
    model.fit(X_train, Y_train)
    return model


def create_and_tune_model(X_train, Y_train):
    """Creates and fits an optimal Logistic Regression model (used in pipeline)."""
    model = create_model()
    return train_model(model, X_train, Y_train)


def evaluate_model(model, X_test, Y_test):
    """Evaluates the model on test data and prints key metrics."""
    Y_pred = model.predict(X_test)
    acc = accuracy_score(Y_test, Y_pred)
    cm = confusion_matrix(Y_test, Y_pred)

    print("\n" + "=" * 50)
    print("      LOGISTIC REGRESSION EVALUATION METRICS       ")
    print("=" * 50)
    print(f" Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(Y_test, Y_pred, zero_division=0))

    return {"accuracy": acc, "confusion_matrix": cm}


def save_model(model):
    """Saves trained model to disk."""
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(model, os.path.join(models_dir, "logistcs_regression.pkl"))
    joblib.dump(model, os.path.join(models_dir, "logistic_regression.pkl"))
    print("[SUCCESS] Logistic Regression model saved to models/logistic_regression.pkl")


def main():
    # 1. Load Data
    print("[1/4] Loading preprocessed data...")
    train_data, test_data = load_preprocessed_data()
    X_train, X_test, Y_train, Y_test = split_features_target(train_data, test_data)

    # 2. Train Model
    print("[2/4] Training Logistic Regression model...")
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