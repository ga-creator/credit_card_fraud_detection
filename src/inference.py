import json
import joblib
import numpy as np
import pandas as pd

MODEL_DIR = "outputs/models"


def load_artifacts(model_dir: str = MODEL_DIR):
    """Load trained models + preprocessing/explanation artifacts saved by main.py."""
    iso_forest = joblib.load(f"{model_dir}/isolation_forest.pkl")
    rf_model = joblib.load(f"{model_dir}/random_forest.pkl")
    scalers = joblib.load(f"{model_dir}/scalers.pkl")

    with open(f"{model_dir}/feature_columns.json") as f:
        feature_columns = json.load(f)
    with open(f"{model_dir}/class_stats.json") as f:
        class_stats = json.load(f)
    with open(f"{model_dir}/feature_importances.json") as f:
        # saved via pandas Series.to_json -> dict preserving sort order
        feature_importances = json.load(f)

    return {
        "iso_forest": iso_forest,
        "rf_model": rf_model,
        "amount_scaler": scalers["amount_scaler"],
        "time_scaler": scalers["time_scaler"],
        "feature_columns": feature_columns,
        "normal_mean": pd.Series(class_stats["normal_mean"]),
        "normal_std": pd.Series(class_stats["normal_std"]),
        "fraud_mean": pd.Series(class_stats["fraud_mean"]),
        "feature_importances": pd.Series(feature_importances),
    }


def validate_columns(df: pd.DataFrame) -> tuple[bool, str]:
    """Check the uploaded dataframe has the columns we need (Time, V1-V28, Amount)."""
    required = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return False, f"Missing required column(s): {', '.join(missing)}"
    return True, ""


def preprocess_for_scoring(df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    df["scaled_amount"] = artifacts["amount_scaler"].transform(df[["Amount"]])
    df["scaled_time"] = artifacts["time_scaler"].transform(df[["Time"]])
    df = df.drop(columns=["Amount", "Time"])
    df = df[artifacts["feature_columns"]]
    return df


def score_transactions(X: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    rf = artifacts["rf_model"]
    iso = artifacts["iso_forest"]

    rf_proba = rf.predict_proba(X)[:, 1]
    rf_pred = (rf_proba >= 0.5).astype(int)

    iso_raw = iso.predict(X)  # +1 normal / -1 anomaly
    iso_pred = (iso_raw == -1).astype(int)
    # Normalize the raw anomaly score to a rough 0-1 "anomaly confidence"
    # using a logistic squashing around 0 (score_samples is centered near 0).
    raw_score = -iso.score_samples(X)
    iso_confidence = 1 / (1 + np.exp(-15 * (raw_score - np.median(raw_score))))

    results = pd.DataFrame({
        "rf_fraud_probability": rf_proba,
        "rf_prediction": rf_pred,
        "isoforest_prediction": iso_pred,
        "isoforest_anomaly_score": iso_confidence,
    })

    # Combined risk label: prioritizes the supervised model's confidence,
    # but escalates if BOTH models agree something is unusual.
    def risk_label(row):
        both_flag = row["rf_prediction"] == 1 and row["isoforest_prediction"] == 1
        if both_flag or row["rf_fraud_probability"] >= 0.75:
            return "High"
        if row["rf_prediction"] == 1 or row["isoforest_prediction"] == 1 or row["rf_fraud_probability"] >= 0.3:
            return "Medium"
        return "Low"

    results["risk_level"] = results.apply(risk_label, axis=1)
    return results


def explain_transaction(x_row: pd.Series, artifacts: dict, top_n: int = 6) -> pd.DataFrame:
    normal_mean = artifacts["normal_mean"]
    normal_std = artifacts["normal_std"]
    importances = artifacts["feature_importances"]

    z = (x_row - normal_mean) / normal_std
    weighted = (z.abs() * importances.reindex(z.index).fillna(0))

    top_features = weighted.sort_values(ascending=False).head(top_n)

    explanation = pd.DataFrame({
        "feature": top_features.index,
        "deviation_from_normal (z-score)": z[top_features.index].round(2).values,
        "model_importance": importances.reindex(top_features.index).round(4).values,
        "contribution_score": top_features.round(4).values,
    })
    return explanation
