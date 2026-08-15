import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data.load_data import load_data, get_project_root

def split_data(df):
    drop_cols = ["PlacementStatus"]
    for col in ["StudentID", "Salary Package", "IsAnomaly"]:
        if col in df.columns:
            drop_cols.append(col)
    
    X = df.drop(columns=drop_cols)
    y = df["PlacementStatus"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test

def identify_features(X):
    numerical_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()
    return numerical_features, categorical_features

def handle_missing_values(X_train, X_test, numerical_features):
    imputer = SimpleImputer(strategy="median")
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    # Fit only on training data
    X_train[numerical_features] = imputer.fit_transform(X_train[numerical_features])
    # Transform test data using the same imputer
    X_test[numerical_features] = imputer.transform(X_test[numerical_features])
    
    return X_train, X_test, imputer

def standardize_data(X_train, X_test, numerical_features):
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    # Fit only on training data
    X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    # Transform test data using same scaler
    X_test[numerical_features] = scaler.transform(X_test[numerical_features])
    
    return X_train, X_test, scaler

def one_hot_encode_data(X_train, X_test, one_hot_features):
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    # Filter features that are actually present
    valid_features = [f for f in one_hot_features if f in X_train.columns]
    if not valid_features:
        return X_train, X_test, encoder

    # Fit only on training data
    train_encoded = encoder.fit_transform(X_train[valid_features])
    test_encoded = encoder.transform(X_test[valid_features])
    
    # Get encoded column names
    encoded_columns = encoder.get_feature_names_out(valid_features)
    
    # Convert to DataFrames
    train_encoded_df = pd.DataFrame(
        train_encoded,
        columns=encoded_columns,
        index=X_train.index
    )
    test_encoded_df = pd.DataFrame(
        test_encoded,
        columns=encoded_columns,
        index=X_test.index
    )
    
    # Remove original categorical columns and concatenate encoded
    X_train = X_train.drop(columns=valid_features)
    X_test = X_test.drop(columns=valid_features)
    
    X_train = pd.concat([X_train, train_encoded_df], axis=1)
    X_test = pd.concat([X_test, test_encoded_df], axis=1)
    
    return X_train, X_test, encoder

def ordinal_encode_data(X_train, X_test, ordinal_features):
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    valid_features = [f for f in ordinal_features if f in X_train.columns]
    if not valid_features:
        return X_train, X_test, encoder

    # Fit only on training data
    train_encoded = encoder.fit_transform(X_train[valid_features])
    test_encoded = encoder.transform(X_test[valid_features])
    
    # Convert to DataFrames
    train_encoded_df = pd.DataFrame(
        train_encoded,
        columns=valid_features,
        index=X_train.index
    )
    test_encoded_df = pd.DataFrame(
        test_encoded,
        columns=valid_features,
        index=X_test.index
    )
    
    # Remove original ordinal columns and concatenate encoded
    X_train = X_train.drop(columns=valid_features)
    X_test = X_test.drop(columns=valid_features)
    
    X_train = pd.concat([X_train, train_encoded_df], axis=1)
    X_test = pd.concat([X_test, test_encoded_df], axis=1)
    
    return X_train, X_test, encoder

def preprocess_pipeline(df=None):
    if df is None:
        df = load_data()
    
    print("Original Dataset Shape:", df.shape)
    
    X_train, X_test, y_train, y_test = split_data(df)
    print("\nTraining Shape (before encoding):", X_train.shape)
    print("Testing Shape (before encoding):", X_test.shape)
    
    numerical_features, categorical_features = identify_features(X_train)
    print("\nNumerical Features:", numerical_features)
    print("Categorical Features:", categorical_features)
    
    one_hot_features = [
        "Gender",
        "City",
        "Stream",
        "Specialisation",
        "Hostel",
        "HistoryOfBacklogs"
    ]
    ordinal_features = [
        "CollegeTier",
        "CGPA_Tier"
    ]
    
    # Process pipeline
    X_train, X_test, imputer = handle_missing_values(X_train, X_test, numerical_features)
    print("\nMissing Value Handling completed.")
    
    X_train, X_test, scaler = standardize_data(X_train, X_test, numerical_features)
    print("Standardization completed.")
    
    X_train, X_test, one_hot_encoder = one_hot_encode_data(X_train, X_test, one_hot_features)
    print("One-Hot Encoding completed.")
    
    X_train, X_test, ordinal_encoder = ordinal_encode_data(X_train, X_test, ordinal_features)
    print("Ordinal Encoding completed.")
    
    # Attach targets
    X_train["PlacementStatus"] = y_train
    X_test["PlacementStatus"] = y_test
    
    # Save output to data directory
    project_root = get_project_root()
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    train_save_path = os.path.join(data_dir, "preprocessed_train.csv")
    test_save_path = os.path.join(data_dir, "preprocessed_test.csv")
    
    X_train.to_csv(train_save_path, index=False)
    X_test.to_csv(test_save_path, index=False)
    
    print("\n----------------------------------------")
    print("PREPROCESSING COMPLETED")
    print("----------------------------------------")
    print(f"Final Training Shape: {X_train.shape}")
    print(f"Final Testing Shape:  {X_test.shape}")
    print("\nFiles saved successfully:")
    print(f" - Train data: {train_save_path}")
    print(f" - Test data:  {test_save_path}")
    
    return X_train, X_test

if __name__ == "__main__":
    preprocess_pipeline()

