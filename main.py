import os
import json
import joblib
import pandas as pd

from src.preprocess import load_data, preprocess, split_data
from src.train import (
    train_isolation_forest,
    train_random_forest,
    isoforest_predict,
    save_model,
)
from src.evaluate import (
    print_report,
    get_metrics_dict,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_pr_curves,
    plot_class_distribution,
    plot_amount_distribution,
    metrics_table,
)

DATA_PATH = "data/creditcard.csv"
OUT_DIR = "outputs"
MODEL_DIR = "outputs/models"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading data...")
    raw_df = load_data(DATA_PATH)
    print(f"Loaded {len(raw_df):,} transactions "
          f"({raw_df['Class'].sum()} fraud, {raw_df['Class'].mean()*100:.3f}% fraud rate)")

    print("\nGenerating EDA plots...")
    plot_class_distribution(raw_df, f"{OUT_DIR}/class_distribution.png")
    plot_amount_distribution(raw_df, f"{OUT_DIR}/amount_distribution.png")

    print("Preprocessing...")
    df, amount_scaler, time_scaler = preprocess(raw_df)
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Persist preprocessing artifacts so the Streamlit app can score new,
    # unseen transactions with the exact same scaling/column order used here.
    joblib.dump(
        {"amount_scaler": amount_scaler, "time_scaler": time_scaler},
        f"{MODEL_DIR}/scalers.pkl",
    )
    with open(f"{MODEL_DIR}/feature_columns.json", "w") as f:
        json.dump(list(X_train.columns), f)

    # Per-class feature statistics (on training data), used by the app to
    # explain predictions by showing how far a feature is from the "normal"
    # transaction profile.
    class_stats = {
        "normal_mean": X_train[y_train == 0].mean().to_dict(),
        "normal_std": X_train[y_train == 0].std().replace(0, 1e-6).to_dict(),
        "fraud_mean": X_train[y_train == 1].mean().to_dict(),
    }
    with open(f"{MODEL_DIR}/class_stats.json", "w") as f:
        json.dump(class_stats, f, indent=2)

    # ---------------- Isolation Forest ----------------
    print("\nTraining Isolation Forest...")
    iso_model = train_isolation_forest(X_train, y_train)
    save_model(iso_model, f"{MODEL_DIR}/isolation_forest.pkl")

    iso_pred = isoforest_predict(iso_model, X_test)
    # anomaly score: higher = more anomalous. score_samples is higher for
    # normal points, so we flip the sign to use it as a "fraud score".
    iso_score = -iso_model.score_samples(X_test)

    print_report(y_test, iso_pred, "Isolation Forest")

    # ---------------- Random Forest (supervised baseline) ----------------
    print("\nTraining Random Forest (supervised baseline)...")
    rf_model = train_random_forest(X_train, y_train)
    save_model(rf_model, f"{MODEL_DIR}/random_forest.pkl")

    rf_pred = rf_model.predict(X_test)
    rf_score = rf_model.predict_proba(X_test)[:, 1]

    print_report(y_test, rf_pred, "Random Forest")

    importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)
    importances.sort_values(ascending=False).to_json(f"{MODEL_DIR}/feature_importances.json")

    # ---------------- Compare & save results ----------------
    print("\nSaving evaluation plots and metrics...")
    plot_confusion_matrix(y_test, iso_pred, "Isolation Forest", f"{OUT_DIR}/confusion_matrix_isoforest.png")
    plot_confusion_matrix(y_test, rf_pred, "Random Forest", f"{OUT_DIR}/confusion_matrix_randomforest.png")

    curve_data = {
        "Isolation Forest": (y_test, iso_score),
        "Random Forest": (y_test, rf_score),
    }
    plot_roc_curves(curve_data, f"{OUT_DIR}/roc_curve_comparison.png")
    plot_pr_curves(curve_data, f"{OUT_DIR}/pr_curve_comparison.png")

    rows = [
        get_metrics_dict(y_test, iso_pred, iso_score, "Isolation Forest"),
        get_metrics_dict(y_test, rf_pred, rf_score, "Random Forest"),
    ]
    table = metrics_table(rows)
    table.to_csv(f"{OUT_DIR}/metrics_summary.csv")
    print("\nFinal metrics summary:\n")
    print(table.to_string())

    print(f"\nAll outputs saved to '{OUT_DIR}/'")


if __name__ == "__main__":
    main()
