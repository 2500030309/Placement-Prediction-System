import os
import sys
import joblib
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
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


def create_model(max_depth=5, criterion='gini', random_state=42):
    """Initializes Decision Tree Classifier."""
    return DecisionTreeClassifier(
        max_depth=max_depth,
        criterion=criterion,
        random_state=random_state
    )


def train_model(model, X_train, Y_train):
    """Trains the Decision Tree on training data."""
    print("[INFO] Training Decision Tree model...", flush=True)
    model.fit(X_train, Y_train)
    return model


def create_and_tune_model(X_train, Y_train):
    """Creates and fits an optimal Decision Tree model (used in pipeline)."""
    model = create_model()
    return train_model(model, X_train, Y_train)


def evaluate_model(model, X_test, Y_test):
    """Evaluates the model on test data and prints key metrics."""
    Y_pred = model.predict(X_test)
    acc = accuracy_score(Y_test, Y_pred)
    cm = confusion_matrix(Y_test, Y_pred)

    print("\n" + "=" * 50)
    print("        DECISION TREE EVALUATION METRICS          ")
    print("=" * 50)
    print(f" Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(Y_test, Y_pred, zero_division=0))

    # Top 5 most important placement features
    if hasattr(model, "feature_importances_") and hasattr(X_test, "columns"):
        importances = pd.Series(model.feature_importances_, index=X_test.columns).sort_values(ascending=False)
        print("\nTop 5 Important Placement Features:")
        for feat, score in importances.head(5).items():
            print(f" - {feat}: {score:.4f}")

    return {"accuracy": acc, "confusion_matrix": cm}


def plot_and_save_tree(model, feature_names, class_names=None, max_depth=3):
    """Visualizes and saves the Decision Tree diagram to app/static/charts/."""
    if class_names is None:
        class_names = ["Not Placed", "Placed"]

    charts_dir = os.path.join(project_root, "app", "static", "charts")
    os.makedirs(charts_dir, exist_ok=True)
    chart_path = os.path.join(charts_dir, "decision_tree.png")

    # Print text-based rules in terminal
    print("\nDecision Tree Rules (Depth 3):")
    print(export_text(model, feature_names=list(feature_names), max_depth=max_depth))

    # Plot visual tree
    plt.figure(figsize=(20, 10))
    plot_tree(
        model,
        feature_names=list(feature_names),
        class_names=class_names,
        filled=True,
        rounded=True,
        max_depth=max_depth,
        fontsize=8
    )
    plt.title("Placement Prediction - Decision Tree Diagram", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"[INFO] Decision tree diagram saved to: {chart_path}")


def save_model(model):
    """Saves trained model to disk."""
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(model, os.path.join(models_dir, "Decision_Tree.pkl"))
    joblib.dump(model, os.path.join(models_dir, "decision_tree.pkl"))
    print("[SUCCESS] Decision Tree model saved to models/Decision_Tree.pkl")


def main():
    # 1. Load Data
    print("[1/4] Loading preprocessed data...")
    train_data, test_data = load_preprocessed_data()
    X_train, X_test, Y_train, Y_test = split_features_target(train_data, test_data)

    # 2. Train Model
    print("[2/4] Training Decision Tree model...")
    model = create_model()
    train_model(model, X_train, Y_train)

    # 3. Evaluate Model
    print("[3/4] Evaluating model performance...")
    evaluate_model(model, X_test, Y_test)

    # 4. Plot & Save Model
    print("[4/4] Plotting tree and saving model...")
    plot_and_save_tree(model, X_train.columns)
    save_model(model)
    print("\nAll tasks completed successfully!")


if __name__ == "__main__":
    main()
