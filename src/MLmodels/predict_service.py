import os
import sys
import joblib
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

# Global model cache for fast responses
_cache = {}


def get_models():
    """Loads and caches preprocessor and trained ML models."""
    if _cache:
        return _cache

    models_dir = os.path.join(project_root, "models")
    prep_path = os.path.join(models_dir, "preprocessor.pkl")
    log_path = os.path.join(models_dir, "logistic_regression.pkl")
    rf_path = os.path.join(models_dir, "Random_Forest.pkl")
    lin_path = os.path.join(models_dir, "linear_regression.pkl")
    km_path = os.path.join(models_dir, "kmeans_elbow.pkl")

    _cache["preprocessor"] = joblib.load(prep_path) if os.path.exists(prep_path) else None
    _cache["logistic"] = joblib.load(log_path) if os.path.exists(log_path) else None
    _cache["random_forest"] = joblib.load(rf_path) if os.path.exists(rf_path) else None
    _cache["linear"] = joblib.load(lin_path) if os.path.exists(lin_path) else None
    _cache["kmeans"] = joblib.load(km_path) if os.path.exists(km_path) else None

    return _cache


def predict_placement_profile(inputs):
    """
    Accepts student profile parameters and runs inference across top models:
    - Top Classification Model 1: Logistic Regression (92.00% accuracy)
    - Top Classification Model 2: Random Forest (91.16% accuracy)
    - Top Regression Model: Linear Regression (Salary Prediction)
    """
    models = get_models()
    prep = models["preprocessor"]
    log_model = models["logistic"]
    rf_model = models["random_forest"]
    lin_model = models["linear"]

    if not prep or not log_model:
        raise RuntimeError("Models or preprocessor artifacts not loaded.")

    cgpa = float(inputs.get("cgpa", 7.5))
    coding_score = float(inputs.get("coding_score", 60.0))
    aptitude_score = float(inputs.get("aptitude_score", 65.0))
    mock_interview = float(inputs.get("mock_interview", 6.5))
    soft_skills = float(inputs.get("soft_skills", 7.0))
    internships = int(inputs.get("internships", 1))
    projects = int(inputs.get("projects", 2))
    backlogs = inputs.get("backlogs", "No")
    stream = inputs.get("stream", "Computer Science")
    gender = inputs.get("gender", "Male")
    college_tier = inputs.get("college_tier", "Tier2")

    # Tier derivation
    if cgpa >= 8.5:
        cgpa_tier = "High"
    elif cgpa >= 6.5:
        cgpa_tier = "Mid"
    else:
        cgpa_tier = "Low"

    # Build student dataframe with typical semester proportions
    student_dict = {
        "Gender": gender,
        "City": "Tier1",
        "CollegeTier": college_tier,
        "Stream": stream,
        "Specialisation": "Software",
        "Hostel": "No",
        "HistoryOfBacklogs": "Yes" if backlogs in ["Yes", "1", 1, True] else "No",
        "SGPA_Sem1": max(4.0, min(10.0, cgpa - 0.3)),
        "SGPA_Sem2": max(4.0, min(10.0, cgpa - 0.2)),
        "SGPA_Sem3": max(4.0, min(10.0, cgpa - 0.1)),
        "SGPA_Sem4": cgpa,
        "SGPA_Sem5": cgpa,
        "SGPA_Sem6": max(4.0, min(10.0, cgpa + 0.1)),
        "SGPA_Sem7": max(4.0, min(10.0, cgpa + 0.2)),
        "SGPA_Sem8": max(4.0, min(10.0, cgpa + 0.2)),
        "CGPA": cgpa,
        "AttendancePercent": 85.0,
        "Internships": internships,
        "Projects": projects,
        "Workshops": 1,
        "Certifications": 1 if projects >= 2 else 0,
        "Publications": 0,
        "AptitudeTestScore": aptitude_score,
        "SoftSkillsRating": soft_skills,
        "CodingTestScore": coding_score,
        "MockInterviewScore": mock_interview,
        "ExtraCurricular": 1,
        "CGPA_Tier": cgpa_tier
    }

    df_input = pd.DataFrame([student_dict])

    # 1. Impute numericals
    imputer = prep["imputer"]
    num_feats = [f for f in prep["numerical_features"] if f in df_input.columns]
    df_input[num_feats] = imputer.transform(df_input[num_feats])

    # 2. Ordinal encode
    ord_enc = prep["ordinal_encoder"]
    ord_feats = [f for f in prep["ordinal_features"] if f in df_input.columns]
    if ord_enc and ord_feats:
        ord_vals = ord_enc.transform(df_input[ord_feats])
        ord_df = pd.DataFrame(ord_vals, columns=ord_feats, index=df_input.index)
        df_input = df_input.drop(columns=ord_feats)
        df_input = pd.concat([df_input, ord_df], axis=1)

    # 3. One-hot encode
    ohe = prep["one_hot_encoder"]
    ohe_feats = [f for f in prep["one_hot_features"] if f in df_input.columns]
    if ohe and ohe_feats:
        ohe_vals = ohe.transform(df_input[ohe_feats])
        ohe_cols = ohe.get_feature_names_out(ohe_feats)
        ohe_df = pd.DataFrame(ohe_vals, columns=ohe_cols, index=df_input.index)
        df_input = df_input.drop(columns=ohe_feats)
        df_input = pd.concat([df_input, ohe_df], axis=1)

    # 4. Standardize
    scaler = prep["scaler"]
    all_cols = prep["feature_columns"]
    # Reindex columns to match feature order
    for col in all_cols:
        if col not in df_input.columns:
            df_input[col] = 0.0

    X_scaled = scaler.transform(df_input[all_cols])
    X_scaled_df = pd.DataFrame(X_scaled, columns=all_cols)

    # Run Top Models
    # Logistic Regression (Top Accuracy: 92.00%)
    log_prob_placed = float(log_model.predict_proba(X_scaled_df)[0][1])
    log_pred = int(log_model.predict(X_scaled_df)[0])

    # Random Forest (Ensemble Accuracy: 91.16%)
    rf_prob_placed = float(rf_model.predict_proba(X_scaled_df)[0][1]) if rf_model else log_prob_placed
    rf_pred = int(rf_model.predict(X_scaled_df)[0]) if rf_model else log_pred

    # Salary Regression (Continuous Package prediction)
    salary_est = float(lin_model.predict(X_scaled_df)[0]) if lin_model else 4.5
    salary_est = max(2.5, min(35.0, round(salary_est, 2)))

    # Composite Placement Likelihood (Weighted ensemble: 60% Logistic + 40% Random Forest)
    ensemble_prob = (0.6 * log_prob_placed) + (0.4 * rf_prob_placed)
    final_status = "Placed" if ensemble_prob >= 0.50 else "Not Placed"

    return {
        "status": final_status,
        "placement_probability": round(ensemble_prob * 100, 1),
        "logistic_regression": {
            "prediction": "Placed" if log_pred == 1 else "Not Placed",
            "probability": round(log_prob_placed * 100, 1),
            "model_rating": "Top Overall Accuracy (92.00%)"
        },
        "random_forest": {
            "prediction": "Placed" if rf_pred == 1 else "Not Placed",
            "probability": round(rf_prob_placed * 100, 1),
            "model_rating": "Top Ensemble Generalizer (91.16%)"
        },
        "predicted_salary_lpa": salary_est if final_status == "Placed" else 0.0,
        "salary_range": f"{max(2.5, salary_est - 1.5):.1f} - {salary_est + 2.0:.1f} LPA" if final_status == "Placed" else "N/A"
    }
