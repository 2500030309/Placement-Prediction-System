import sys
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data.load_data import load_data, get_project_root

def split_data(df):
    """Splits dataset into 80% training and 20% testing sets stratified by target."""
    drop_cols = ["PlacementStatus"]
    for col in ["StudentID", "Salary Package", "IsAnomaly"]:
        if col in df.columns:
            drop_cols.append(col)
    
    X = df.drop(columns=drop_cols)
    y = df["PlacementStatus"]
    
    stratify_target = y if len(np.unique(y)) > 1 else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_target
    )

    return X_train, X_test, y_train, y_test

def identify_features(X):
    """Identifies numerical and categorical features."""
    numerical_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()
    return numerical_features, categorical_features

def handle_missing_values(X_train, X_test, numerical_features):
    """Imputes missing values in numerical columns using median strategy."""
    imputer = SimpleImputer(strategy="median")
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    X_train[numerical_features] = imputer.fit_transform(X_train[numerical_features])
    X_test[numerical_features] = imputer.transform(X_test[numerical_features])
    
    return X_train, X_test, imputer

def ordinal_encode_data(X_train, X_test, ordinal_features):
    """Encodes ordinal features with explicit category hierarchy."""
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    valid_features = [f for f in ordinal_features if f in X_train.columns]
    if not valid_features:
        return X_train, X_test, None

    category_mappings = {
        "CollegeTier": ["Tier3", "Tier2", "Tier1"],
        "CGPA_Tier": ["Low", "Mid", "High"]
    }
    
    categories = [category_mappings[col] for col in valid_features if col in category_mappings]
    
    encoder = OrdinalEncoder(
        categories=categories,
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )
    
    train_encoded = encoder.fit_transform(X_train[valid_features])
    test_encoded = encoder.transform(X_test[valid_features])
    
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
    
    X_train = X_train.drop(columns=valid_features)
    X_test = X_test.drop(columns=valid_features)
    
    X_train = pd.concat([X_train, train_encoded_df], axis=1)
    X_test = pd.concat([X_test, test_encoded_df], axis=1)
    
    return X_train, X_test, encoder

def one_hot_encode_data(X_train, X_test, one_hot_features):
    """Encodes nominal categorical features using OneHotEncoder."""
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    valid_features = [f for f in one_hot_features if f in X_train.columns]
    if not valid_features:
        return X_train, X_test, encoder

    train_encoded = encoder.fit_transform(X_train[valid_features])
    test_encoded = encoder.transform(X_test[valid_features])
    
    encoded_columns = encoder.get_feature_names_out(valid_features)
    
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
    
    X_train = X_train.drop(columns=valid_features)
    X_test = X_test.drop(columns=valid_features)
    
    X_train = pd.concat([X_train, train_encoded_df], axis=1)
    X_test = pd.concat([X_test, test_encoded_df], axis=1)
    
    return X_train, X_test, encoder

def standardize_data(X_train, X_test, feature_columns):
    """Standardizes all predictor features to zero mean and unit variance."""
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    X_train[feature_columns] = scaler.fit_transform(X_train[feature_columns])
    X_test[feature_columns] = scaler.transform(X_test[feature_columns])
    
    return X_train, X_test, scaler

def preprocess_pipeline(df=None, save_files=True):
    """Executes the complete preprocessing pipeline on raw data."""
    if df is None:
        df = load_data()
    
    print("Original Dataset Shape:", df.shape)
    
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"\nData Split (80/20): Training={X_train.shape[0]}, Testing={X_test.shape[0]}")
    
    numerical_features, categorical_features = identify_features(X_train)
    
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
    
    # 1. Handle Missing Values
    X_train, X_test, imputer = handle_missing_values(X_train, X_test, numerical_features)
    print("Missing Value Imputation completed.")
    
    # 2. Ordinal Encoding
    X_train, X_test, ordinal_encoder = ordinal_encode_data(X_train, X_test, ordinal_features)
    print("Ordinal Encoding completed with explicit ordering.")
    
    # 3. One-Hot Encoding
    X_train, X_test, one_hot_encoder = one_hot_encode_data(X_train, X_test, one_hot_features)
    print("One-Hot Encoding completed.")
    
    # 4. Standardize All Predictor Features
    all_feature_cols = X_train.columns.tolist()
    X_train, X_test, scaler = standardize_data(X_train, X_test, all_feature_cols)
    print("Full Feature Standardization completed.")
    
    # Attach target back for saving
    X_train_out = X_train.copy()
    X_test_out = X_test.copy()
    X_train_out["PlacementStatus"] = y_train.values
    X_test_out["PlacementStatus"] = y_test.values
    
    # Save datasets & fitted preprocessors if requested
    if save_files:
        project_root = get_project_root()
        data_dir = os.path.join(project_root, "data")
        models_dir = os.path.join(project_root, "models")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        
        train_save_path = os.path.join(data_dir, "preprocessed_train.csv")
        test_save_path = os.path.join(data_dir, "preprocessed_test.csv")
        preprocessor_save_path = os.path.join(models_dir, "preprocessor.pkl")
        
        try:
            X_train_out.to_csv(train_save_path, index=False)
            X_test_out.to_csv(test_save_path, index=False)
            print("\nFiles saved successfully:")
            print(f" - Train data: {train_save_path}")
            print(f" - Test data:  {test_save_path}")
        except Exception as e:
            print(f"\n[WARNING] Could not write preprocessed CSVs to disk ({e}). Proceeding in memory.")
        
        try:
            preprocessor_artifact = {
                "imputer": imputer,
                "ordinal_encoder": ordinal_encoder,
                "one_hot_encoder": one_hot_encoder,
                "scaler": scaler,
                "numerical_features": numerical_features,
                "ordinal_features": ordinal_features,
                "one_hot_features": one_hot_features,
                "feature_columns": all_feature_cols
            }
            joblib.dump(preprocessor_artifact, preprocessor_save_path)
            print(f" - Preprocessor: {preprocessor_save_path}")
        except Exception as e:
            print(f"\n[WARNING] Could not save preprocessor artifact ({e}).")
    
    print("\n----------------------------------------")
    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    print("----------------------------------------")
    print(f"Final Training Shape: {X_train_out.shape}")
    print(f"Final Testing Shape:  {X_test_out.shape}")
    
    return X_train_out, X_test_out

if __name__ == "__main__":
    preprocess_pipeline()


