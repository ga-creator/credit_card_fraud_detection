import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from src.inference import (
    load_artifacts,
    validate_columns,
    preprocess_for_scoring,
    score_transactions,
    explain_transaction,
)

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")


# ---------------------------------------------------------------------------
# Load models & artifacts 
# ---------------------------------------------------------------------------
@st.cache_resource
def get_artifacts():
    return load_artifacts()


try:
    artifacts = get_artifacts()
except FileNotFoundError:
    st.error(
        "Model artifacts not found in `outputs/models/`. "
        "Run `python main.py` first to train the models before launching this app."
    )
    st.stop()


RISK_COLOR = {"Low": "#4C72B0", "Medium": "#DD8452", "High": "#C44E52"}


def risk_badge(level: str) -> str:
    color = RISK_COLOR[level]
    return f'<span style="background-color:{color};color:white;padding:2px 10px;border-radius:10px;font-weight:600;">{level}</span>'


def show_explanation(x_row: pd.Series, key_prefix: str = ""):
    expl = explain_transaction(x_row, artifacts)
    st.caption(
        "Top features driving this prediction — how far each value sits from a "
        "*typical normal transaction*, weighted by how much the model relies on that feature."
    )
    fig = px.bar(
        expl.sort_values("contribution_score"),
        x="contribution_score",
        y="feature",
        orientation="h",
        labels={"contribution_score": "Contribution to anomaly score", "feature": ""},
        color="contribution_score",
        color_continuous_scale=["#4C72B0", "#C44E52"],
    )
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_explain_chart")
    with st.expander("Raw explanation numbers"):
        st.dataframe(expl, hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("💳 Credit Card Fraud Detection")
st.write(
    "Score transactions using an **Isolation Forest** (unsupervised anomaly detector) "
    "and a **Random Forest** (supervised classifier), trained on the Kaggle Credit Card "
    "Fraud dataset. Upload a batch of transactions or enter one manually."
)

tab_upload, tab_manual, tab_about = st.tabs(["📁 Upload CSV", "✍️ Manual Entry", "ℹ️ About the models"])

# ---------------------------------------------------------------------------
# TAB 1 — Upload CSV
# ---------------------------------------------------------------------------
with tab_upload:
    st.subheader("Upload a CSV of transactions")
    st.caption(
        "Expected columns: `Time`, `V1`...`V28`, `Amount` (the same schema as the Kaggle "
        "dataset). An optional `Class` column, if present, is ignored for scoring."
    )

    uploaded = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded is not None:
        try:
            raw_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read that file as CSV: {e}")
            st.stop()

        ok, msg = validate_columns(raw_df)
        if not ok:
            st.error(msg)
            st.stop()

        st.success(f"Loaded {len(raw_df):,} transactions.")

        if st.button("🔍 Detect Fraud", type="primary", key="batch_detect"):
            with st.spinner("Scoring transactions..."):
                feature_df = raw_df.drop(columns=["Class"], errors="ignore")
                X = preprocess_for_scoring(feature_df, artifacts)
                results = score_transactions(X, artifacts)
                output = pd.concat(
                    [raw_df.reset_index(drop=True), results.reset_index(drop=True)], axis=1
                )

            st.session_state["batch_output"] = output
            st.session_state["batch_X"] = X

        if "batch_output" in st.session_state:
            output = st.session_state["batch_output"]
            X = st.session_state["batch_X"]

            n_total = len(output)
            n_high = (output["risk_level"] == "High").sum()
            n_medium = (output["risk_level"] == "Medium").sum()
            n_low = (output["risk_level"] == "Low").sum()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Transactions scored", f"{n_total:,}")
            c2.metric("🔴 High risk", f"{n_high:,}")
            c3.metric("🟠 Medium risk", f"{n_medium:,}")
            c4.metric("🔵 Low risk", f"{n_low:,}")

            st.divider()
            st.subheader("Results")

            risk_filter = st.multiselect(
                "Filter by risk level", ["High", "Medium", "Low"], default=["High", "Medium"]
            )
            sort_desc = st.checkbox("Sort by fraud probability (highest first)", value=True)

            display_cols = ["Amount", "rf_fraud_probability", "rf_prediction",
                             "isoforest_prediction", "isoforest_anomaly_score", "risk_level"]
            filtered = output[output["risk_level"].isin(risk_filter)] if risk_filter else output
            if sort_desc:
                filtered = filtered.sort_values("rf_fraud_probability", ascending=False)

            st.dataframe(
                filtered[display_cols].rename(columns={
                    "rf_fraud_probability": "RF fraud probability",
                    "rf_prediction": "RF prediction (1=fraud)",
                    "isoforest_prediction": "Isolation Forest prediction (1=anomaly)",
                    "isoforest_anomaly_score": "Isolation Forest anomaly score",
                    "risk_level": "Risk level",
                }).style.format({"RF fraud probability": "{:.2%}", "Isolation Forest anomaly score": "{:.2%}"}),
                use_container_width=True,
                height=400,
            )

            csv_bytes = filtered.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download flagged results as CSV",
                data=csv_bytes,
                file_name="fraud_detection_results.csv",
                mime="text/csv",
            )

            st.divider()
            st.subheader("Explain an individual transaction")
            idx_options = filtered.index.tolist()
            if idx_options:
                chosen_idx = st.selectbox(
                    "Pick a row (by original row number) to see why it was flagged",
                    idx_options,
                )
                colA, colB = st.columns([1, 2])
                with colA:
                    row_out = output.loc[chosen_idx]
                    st.markdown(f"**Amount:** ${row_out['Amount']:.2f}")
                    st.markdown(f"**RF fraud probability:** {row_out['rf_fraud_probability']:.2%}")
                    st.markdown(f"**Risk level:** {risk_badge(row_out['risk_level'])}", unsafe_allow_html=True)
                with colB:
                    show_explanation(X.loc[chosen_idx], key_prefix="batch")
            else:
                st.info("No rows match the current filter.")


# ---------------------------------------------------------------------------
# TAB 2 — Manual entry
# ---------------------------------------------------------------------------
with tab_manual:
    st.subheader("Enter a single transaction")
    st.caption(
        "`Amount` and `Time` are the only human-interpretable fields in this dataset — "
        "the `V1`-`V28` features are anonymized PCA components of the original transaction "
        "data (Kaggle masks these for privacy). Adjust the most influential ones below, "
        "or expand 'Advanced' to fine-tune all 28."
    )

    top_features = artifacts["feature_importances"].sort_values(ascending=False)
    top_v_features = [f for f in top_features.index if f.startswith("V")][:8]
    normal_mean = artifacts["normal_mean"]
    normal_std = artifacts["normal_std"]

    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0, step=10.0)
    with col2:
        time_val = st.number_input(
            "Time (seconds since first transaction in dataset)",
            min_value=0.0, value=50000.0, step=1000.0,
        )

    st.markdown("**Most influential features** (adjust as multiples of typical variation):")
    v_values = {f: normal_mean[f] for f in [f"V{i}" for i in range(1, 29)]}

    slider_cols = st.columns(4)
    for i, feat in enumerate(top_v_features):
        with slider_cols[i % 4]:
            z = st.slider(
                feat, min_value=-8.0, max_value=8.0, value=0.0, step=0.1,
                help="0 = typical normal transaction. Positive/negative moves away from that baseline.",
                key=f"slider_{feat}",
            )
            v_values[feat] = normal_mean[feat] + z * normal_std[feat]

    with st.expander("Advanced: set all V1–V28 directly"):
        adv_cols = st.columns(4)
        for i in range(1, 29):
            feat = f"V{i}"
            with adv_cols[i % 4]:
                v_values[feat] = st.number_input(
                    feat, value=float(round(v_values[feat], 4)), step=0.1, key=f"adv_{feat}"
                )

    if st.button("🔍 Detect Fraud", type="primary", key="manual_detect"):
        row = {"Time": time_val, "Amount": amount, **v_values}
        single_df = pd.DataFrame([row])
        X = preprocess_for_scoring(single_df, artifacts)
        results = score_transactions(X, artifacts)
        r = results.iloc[0]

        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("RF fraud probability", f"{r['rf_fraud_probability']:.2%}")
        m2.metric("Isolation Forest anomaly score", f"{r['isoforest_anomaly_score']:.2%}")
        m3.markdown("**Risk level**")
        m3.markdown(risk_badge(r["risk_level"]), unsafe_allow_html=True)

        if r["risk_level"] == "High":
            st.error("⚠️ This transaction looks highly suspicious.")
        elif r["risk_level"] == "Medium":
            st.warning("This transaction has some unusual characteristics — worth a second look.")
        else:
            st.success("This transaction looks consistent with normal activity.")

        st.subheader("Why?")
        show_explanation(X.iloc[0], key_prefix="manual")


# ---------------------------------------------------------------------------
# TAB 3 — About / model performance
# ---------------------------------------------------------------------------
with tab_about:
    st.subheader("About these models")
    st.markdown(
        """
- **Isolation Forest** — unsupervised. Flags transactions that are "easy to isolate"
  via random partitioning, without ever seeing fraud labels during training.
- **Random Forest** — supervised, trained with `class_weight="balanced"` on labeled
  historical fraud data. Generally the stronger, more precise model when labels exist.
- **Risk level** combines both: flagged as **High** when both models agree or the
  Random Forest is very confident; **Medium** when either model raises a flag;
  otherwise **Low**.
        """
    )

    try:
        metrics_df = pd.read_csv("outputs/metrics_summary.csv").set_index("model")
        st.subheader("Test-set performance")
        st.dataframe(
            metrics_df.style.format("{:.3f}"),
            use_container_width=True,
        )
    except FileNotFoundError:
        st.info("Run `python main.py` to generate `outputs/metrics_summary.csv`.")

    img_col1, img_col2 = st.columns(2)
    try:
        img_col1.image("outputs/roc_curve_comparison.png", caption="ROC Curve Comparison")
        img_col2.image("outputs/pr_curve_comparison.png", caption="Precision-Recall Curve Comparison")
    except Exception:
        pass
