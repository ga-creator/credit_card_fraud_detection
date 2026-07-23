import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve,
)


def print_report(y_true, y_pred, model_name: str):
    print(f"\n{'='*55}\n{model_name} - Classification Report\n{'='*55}")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Fraud"], digits=4))


def get_metrics_dict(y_true, y_pred, y_score, model_name: str) -> dict:
    report = classification_report(y_true, y_pred, output_dict=True)
    return {
        "model": model_name,
        "precision_fraud": report["1"]["precision"],
        "recall_fraud": report["1"]["recall"],
        "f1_fraud": report["1"]["f1-score"],
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
    }


def plot_confusion_matrix(y_true, y_pred, model_name: str, out_path: str):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Fraud"], yticklabels=["Normal", "Fraud"])
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_roc_curves(results: dict, out_path: str):
    """results: {model_name: (y_true, y_score)}"""
    plt.figure(figsize=(6, 5))
    for name, (y_true, y_score) in results.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_pr_curves(results: dict, out_path: str):
    plt.figure(figsize=(6, 5))
    for name, (y_true, y_score) in results.items():
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        plt.plot(rec, prec, label=f"{name} (AP = {ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_class_distribution(df, out_path: str):
    plt.figure(figsize=(5, 4))
    ax = sns.countplot(x="Class", data=df, hue="Class", palette=["#4C72B0", "#C44E52"], legend=False)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Normal", "Fraud"])
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom")
    plt.title("Class Distribution (Normal vs Fraud)")
    plt.yscale("log")
    plt.ylabel("Count (log scale)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_amount_distribution(df, out_path: str, amount_col: str = "Amount"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(df[df["Class"] == 0][amount_col], bins=50, ax=axes[0], color="#4C72B0")
    axes[0].set_title("Transaction Amount - Normal")
    axes[0].set_xlim(0, 2000)
    sns.histplot(df[df["Class"] == 1][amount_col], bins=50, ax=axes[1], color="#C44E52")
    axes[1].set_title("Transaction Amount - Fraud")
    axes[1].set_xlim(0, 2000)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def metrics_table(rows: list) -> pd.DataFrame:
    return pd.DataFrame(rows).set_index("model").round(4)
