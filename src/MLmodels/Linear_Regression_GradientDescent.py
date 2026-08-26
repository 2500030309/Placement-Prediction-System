import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from src.data.load_data import load_data
except ImportError:
    from src.data.load import load_data


def gradient_descent(X, y, learning_rate=0.01, epochs=100):
    m = len(X)
    theta0 = 0.0
    theta1 = 0.0
    mse_history = []
    for epoch in range(epochs):
        y_hat = theta0 + theta1 * X
        error = y_hat - y
        mse = np.mean(error ** 2)
        mse_history.append(mse)
        grad_theta0 = (2 / m) * np.sum(error)
        grad_theta1 = (2 / m) * np.sum(error * X)
        theta0 = theta0 - learning_rate * grad_theta0
        theta1 = theta1 - learning_rate * grad_theta1
    return theta0, theta1, mse_history


if __name__ == "__main__":
    df = load_data()
    X = df["CGPA"].values
    y = df["Salary Package"].values
    X_scaled = (X - np.mean(X)) / np.std(X)
    theta0, theta1, mse_history = gradient_descent(X_scaled, y, 0.01, 100)
    print("Theta 0:", theta0)
    print("Theta 1:", theta1)
    print("Final MSE:", mse_history[-1])

    plt.plot(mse_history)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Gradient Descent")
    plt.show()
