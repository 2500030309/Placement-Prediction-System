import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data.load_data import load_data, get_project_root


class GradientDescentLinearRegression:
    """Univariate Linear Regression trained using Batch Gradient Descent."""
    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.theta0 = 0.0
        self.theta1 = 0.0
        self.mse_history = []

    def fit(self, X, y):
        m = len(X)
        self.theta0 = 0.0
        self.theta1 = 0.0
        self.mse_history = []
        for epoch in range(self.epochs):
            y_hat = self.theta0 + self.theta1 * X
            error = y_hat - y
            mse = np.mean(error ** 2)
            self.mse_history.append(mse)
            grad_theta0 = (2 / m) * np.sum(error)
            grad_theta1 = (2 / m) * np.sum(error * X)
            self.theta0 -= self.learning_rate * grad_theta0
            self.theta1 -= self.learning_rate * grad_theta1
        return self

    def predict(self, X):
        return self.theta0 + self.theta1 * X


def create_model(learning_rate=0.01, epochs=100):
    """Initializes and returns a Gradient Descent Linear Regression model."""
    return GradientDescentLinearRegression(learning_rate=learning_rate, epochs=epochs)


def train_model(model, X_train, y_train):
    """Fits the Gradient Descent model on training data."""
    return model.fit(X_train, y_train)


def evaluate_model(model, X_test, y_test):
    """Evaluates regression metrics (MAE, MSE, RMSE) for the Gradient Descent model."""
    y_pred = model.predict(X_test)
    mse = np.mean((y_test - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test - y_pred))
    
    print("\n==================================================")
    print("  GRADIENT DESCENT LINEAR REGRESSION METRICS      ")
    print("==================================================")
    print(f" Theta 0 (Intercept):       {model.theta0:.4f}")
    print(f" Theta 1 (Slope):           {model.theta1:.4f}")
    print(f" Mean Absolute Error (MAE): {mae:.4f}")
    print(f" Mean Squared Error (MSE):  {mse:.4f}")
    print(f" Root Mean Sq Error (RMSE): {rmse:.4f}")
    print("==================================================")
    return {"mae": mae, "mse": mse, "rmse": rmse}


def gradient_descent(X, y, learning_rate=0.01, epochs=100):
    """Backwards-compatible functional interface."""
    model = create_model(learning_rate=learning_rate, epochs=epochs)
    train_model(model, X, y)
    return model.theta0, model.theta1, model.mse_history


def plot_and_save_loss(model, filename="gradient_descent_loss.png"):
    """Saves the MSE cost history plot to app/static/charts/."""
    project_root = get_project_root()
    charts_dir = os.path.join(project_root, "app", "static", "charts")
    os.makedirs(charts_dir, exist_ok=True)
    
    chart_path = os.path.join(charts_dir, filename)
    plt.figure(figsize=(8, 5))
    plt.plot(model.mse_history, color="#3b82f6", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.title("Gradient Descent Optimization Curve")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(chart_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[INFO] Cost function curve saved to: {chart_path}")


def main():
    """Main execution function for Gradient Descent Linear Regression."""
    df = load_data()
    placed_df = df[df["Salary Package"] > 0].reset_index(drop=True)
    
    X = placed_df["CGPA"].values
    y = placed_df["Salary Package"].values
    
    # Feature standardization
    X_mean, X_std = np.mean(X), np.std(X)
    X_scaled = (X - X_mean) / X_std
    
    model = create_model(learning_rate=0.01, epochs=100)
    train_model(model, X_scaled, y)
    evaluate_model(model, X_scaled, y)
    plot_and_save_loss(model)


if __name__ == "__main__":
    main()
