import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, mean_squared_error, root_mean_squared_error


def load_preprocessed_data():
    train_data = r"C:\Users\LUCKY\PycharmProjects\Placements Prediction System\data\preprocessed_train.csv"
    test_data =  r"C:\Users\LUCKY\PycharmProjects\Placements Prediction System\data\preprocessed_test.csv"
    train_data = pd.read_csv(train_data)
    test_data = pd.read_csv(test_data)
    return train_data, test_data

def split_features_target(train_data, test_data):
    X_train = train_data.drop(columns=["PlacementStatus"])
    X_test = test_data.drop(columns=["PlacementStatus"])
    Y_train = train_data["PlacementStatus"]
    Y_test = test_data["PlacementStatus"]
    return X_train, X_test, Y_train, Y_test

def create_model():
    model = LogisticRegression(
        max_iter = 1000,
        random_state= 42
    )

    return model

def train_model(model, X_train, Y_train):
    model.fit(X_train, Y_train)
    return model

def evaluate_model(model, X_test, Y_test):
    Y_pred = model.predict(X_test)
    print("\nAccuracy: ")
    print(model.score(X_test, Y_test))
    print("\nClassification Report: ")
    print(classification_report(Y_test, Y_pred))

def save_model(model):
    model_path = r"C:\Users\LUCKY\PycharmProjects\Placements Prediction System\models\logistcs_regression.pkl"
    joblib.dump(model, model_path)
    print("\nModel Saved Successfully")
    print(model_path)

if __name__ == "__main__":
    train_data, test_data = load_preprocessed_data()
    print("Training Data Shape: ")
    print(train_data.shape)
    print("Testing Data Shape: ")
    print(test_data.shape)
    X_train, X_test, Y_train, Y_test = split_features_target(train_data, test_data)
    print("\nX_train Shape : ")
    print(X_train.shape)
    print("\nX_test Shape : ")
    print(X_test.shape)

    model = create_model()
    print("\n Logistic Regression Model Created ")
    model = train_model(model, X_train, Y_train)
    print("\n Logistic Regression Model Evaluated ")
    evaluate_model(model, X_test, Y_test)
    save_model(model)