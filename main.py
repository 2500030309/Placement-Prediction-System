import sys
import os

from src.data.load_data import load_data, get_summary
from src.data.eda import basic_eda, univariate, bivariate, multivariate
from src.data.preprocess import preprocess_pipeline
from src.MLmodels.logistcs_regression import (
    load_preprocessed_data as load_logistic_data,
    split_features_target as split_logistic_features,
    create_and_tune_model as tune_logistic_model,
    evaluate_model as evaluate_logistic_model,
    save_model as save_logistic_model
)
from src.MLmodels.linear_regression import (
    load_regression_data as load_linear_data,
    create_and_tune_model as tune_linear_model,
    evaluate_model as evaluate_linear_model,
    save_model as save_linear_model
)
from src.MLmodels.Decision_Tree import (
    create_and_tune_model as tune_decision_tree_model,
    evaluate_model as evaluate_decision_tree_model,
    plot_and_save_tree as plot_decision_tree_diagram,
    save_model as save_decision_tree_model
)
from src.MLmodels.Random_Forest import (
    create_and_tune_model as tune_random_forest_model,
    evaluate_model as evaluate_random_forest_model,
    plot_and_save_feature_importance as plot_rf_feature_importance,
    plot_and_save_tree as plot_rf_tree_diagram,
    save_model as save_random_forest_model
)

def main():
    print("==================================================")
    print("      PLACEMENT PREDICTION SYSTEM PIPELINE        ")
    print("==================================================\n")
    
    # 1. Load Data
    print("[1/6] Loading dataset...")
    df = load_data()
    summary = get_summary(df)
    print(f"      Rows: {summary['rows']}, Columns: {summary['columns']}, Target: {summary['target']}")
    
    # 2. Run EDA
    print("\n[2/6] Generating Exploratory Data Analysis (EDA) visualizations...")
    basic_eda(df)
    univariate(df)
    bivariate(df)
    multivariate(df)
    print("      EDA charts saved to app/static/charts/")
    
    # 3. Run Preprocessing
    print("\n[3/6] Executing data preprocessing pipeline...")
    X_train_df, X_test_df = preprocess_pipeline(df)
    
    # 4. Logistic Regression Training & Evaluation (Classification: PlacementStatus)
    print("\n[4/6] Training and evaluating Logistic Regression model (Placement Status)...")
    train_data, test_data = load_logistic_data()
    X_train_cls, X_test_cls, Y_train_cls, Y_test_cls = split_logistic_features(train_data, test_data)
    log_model = tune_logistic_model(X_train_cls, Y_train_cls)
    evaluate_logistic_model(log_model, X_test_cls, Y_test_cls)
    save_logistic_model(log_model)
    
    # 5. Decision Tree Training & Diagram Generation (Classification: PlacementStatus)
    print("\n[5/7] Training and evaluating Decision Tree model & Diagram (Placement Status)...")
    dt_model = tune_decision_tree_model(X_train_cls, Y_train_cls)
    evaluate_decision_tree_model(dt_model, X_test_cls, Y_test_cls)
    plot_decision_tree_diagram(dt_model, feature_names=X_train_cls.columns, class_names=["Not Placed", "Placed"], max_depth=3)
    save_decision_tree_model(dt_model)
    
    # 6. Random Forest Training & Evaluation (Classification: PlacementStatus)
    print("\n[6/7] Training and evaluating Random Forest model (Placement Status)...")
    rf_model = tune_random_forest_model(X_train_cls, Y_train_cls)
    evaluate_random_forest_model(rf_model, X_test_cls, Y_test_cls)
    plot_rf_feature_importance(rf_model, feature_names=X_train_cls.columns, top_n=15)
    plot_rf_tree_diagram(rf_model, feature_names=X_train_cls.columns, class_names=["Not Placed", "Placed"], max_depth=3)
    save_random_forest_model(rf_model)
    
    # 7. Linear Regression Training & Evaluation (Regression: Salary Package)
    print("\n[7/7] Training and evaluating Linear Regression model (Salary Package)...")
    X_train_reg, X_test_reg, Y_train_reg, Y_test_reg = load_linear_data()
    lin_model = tune_linear_model(X_train_reg, Y_train_reg)
    evaluate_linear_model(lin_model, X_test_reg, Y_test_reg)
    save_linear_model(lin_model)
    
    print("\n==================================================")
    print("       PIPELINE EXECUTED SUCCESSFULLY!            ")
    print("==================================================")

if __name__ == '__main__':
    main()
