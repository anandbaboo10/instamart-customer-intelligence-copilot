"""Streamlit dashboard for the Instamart Customer Intelligence Copilot.
Run from the repository root:
    streamlit run app/dashboard.py
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
STATE_DIR = PROJECT_ROOT / "state"
DB_PATH = CLEAN_DIR / "instamart.db"

st.set_page_config(page_title="Instamart Intelligence Copilot", page_icon="🛒", layout="wide")


@st.cache_data(show_spinner=False)
def read_csv_if_exists(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def read_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_orders() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as conn:
        orders = pd.read_sql_query("SELECT * FROM clean_orders", conn)
    orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="coerce")
    return orders


def empty_artifact_message(artifact: str) -> None:
    st.warning(f"Missing {artifact}. Run the corresponding pipeline script from the project README.")


def overview_page() -> None:
    st.title("🛒 Instamart Customer Intelligence Copilot")
    st.caption("Synthetic-data analytics, safe ticket intelligence, grounded policy Q&A, and guarded refund workflows.")
    orders = load_orders()
    customers = read_csv_if_exists(CLEAN_DIR / "customers_clean.csv")
    products = read_csv_if_exists(CLEAN_DIR / "products_clean.csv")
    if orders.empty or customers.empty:
        empty_artifact_message("clean data")
        return

    completed = orders[(orders["status"] == "completed") & orders["order_amount"].notna()].copy()
    total_revenue = completed["order_amount"].sum()
    avg_order_value = completed["order_amount"].mean()
    avg_delivery = completed["delivery_minutes"].mean()
    cols = st.columns(4)
    cols[0].metric("Customers", f"{len(customers):,}")
    cols[1].metric("Completed orders", f"{len(completed):,}")
    cols[2].metric("Completed revenue", f"INR {total_revenue:,.0f}")
    cols[3].metric("Average delivery", f"{avg_delivery:.1f} min" if pd.notna(avg_delivery) else "N/A")

    left, right = st.columns(2)
    with left:
        st.subheader("Monthly completed revenue")
        monthly = completed.assign(month=completed["order_ts"].dt.to_period("M").astype(str)).groupby("month")["order_amount"].sum()
        st.bar_chart(monthly)
    with right:
        st.subheader("Revenue by category")
        if not products.empty:
            category = completed.merge(products[["product_id", "category"]], on="product_id", how="left")
            category_summary = category.groupby("category")["order_amount"].sum().sort_values(ascending=False)
            st.bar_chart(category_summary)

    st.subheader("Data-quality status")
    audit = read_json_if_exists(OUTPUT_DIR / "audit_summary.json")
    if audit:
        st.success(f"Validation passed: {audit.get('validation', {}).get('passed', 'unknown')}")
        st.json(audit.get("cleaning_actions", {}), expanded=False)
    else:
        empty_artifact_message("audit summary")


def segmentation_page() -> None:
    st.title("Customer Segmentation & Purchase Propensity")
    features = read_csv_if_exists(CLEAN_DIR / "customer_features.csv")
    metrics = read_json_if_exists(OUTPUT_DIR / "model_metrics.json")
    selection = read_csv_if_exists(OUTPUT_DIR / "segmentation_selection_metrics.csv")
    if features.empty:
        empty_artifact_message("customer features")
        return

    if metrics:
        cols = st.columns(4)
        cols[0].metric("Accuracy", f"{metrics.get('accuracy', 0):.1%}")
        cols[1].metric("Precision", f"{metrics.get('precision', 0):.1%}")
        cols[2].metric("Recall", f"{metrics.get('recall', 0):.1%}")
        cols[3].metric("F1", f"{metrics.get('f1', 0):.1%}")
        st.caption(f"Point-in-time cutoff: {metrics.get('cutoff_date')}; prediction window: {metrics.get('prediction_window')}.")

    profile = (
        features.groupby("segment")
        .agg(
            customers=("customer_id", "count"),
            avg_historical_orders=("historical_orders", "mean"),
            avg_revenue_inr=("historical_revenue_inr", "mean"),
            avg_recency_days=("recency_days", "mean"),
            future_purchase_rate=("future_purchase_60d", "mean"),
        )
        .round(2)
        .reset_index()
    )
    st.subheader("Segment profiles")
    st.dataframe(profile, use_container_width=True, hide_index=True)
    st.bar_chart(profile.set_index("segment")[["customers", "future_purchase_rate"]])

    st.subheader("KMeans selection diagnostics")
    if selection.empty:
        empty_artifact_message("segmentation selection metrics")
    else:
        st.dataframe(selection, use_container_width=True, hide_index=True)
        st.line_chart(selection.set_index("k")[["silhouette_score", "inertia"]])

    with st.expander("Feature sample"):
        st.dataframe(features.head(25), use_container_width=True, hide_index=True)


def ticket_page() -> None:
    st.title("Ticket Intelligence")
    results = read_csv_if_exists(OUTPUT_DIR / "ticket_classification_results.csv")
    metrics = read_json_if_exists(OUTPUT_DIR / "ticket_classification_metrics.json")
    if results.empty:
        empty_artifact_message("ticket classification results")
        return

    zero = metrics.get("zero_shot", {})
    few = metrics.get("few_shot", {})
    cols = st.columns(3)
    cols[0].metric("Zero-shot accuracy", f"{zero.get('accuracy', 0):.1%}")
    cols[1].metric("Few-shot accuracy", f"{few.get('accuracy', 0):.1%}")
    cols[2].metric("Few-shot uplift", f"{metrics.get('accuracy_change_few_shot_minus_zero_shot', 0):+.1%}")
    st.caption("Displayed ticket text has been masked before model inference and output storage.")

    st.subheader("Predicted ticket categories")
    st.bar_chart(results["few_shot_prediction"].value_counts())
    show_columns = [
        column for column in [
            "ticket_id", "true_category", "few_shot_prediction", "few_shot_reason",
            "few_shot_attempts", "few_shot_used_fallback", "masked_text"
        ] if column in results.columns
    ]
    st.dataframe(results[show_columns], use_container_width=True, hide_index=True)


def policy_page() -> None:
    st.title("Grounded Policy Q&A")
    results = read_csv_if_exists(OUTPUT_DIR / "policy_qa_results.csv")
    metrics = read_json_if_exists(OUTPUT_DIR / "policy_evaluation_metrics.json")
    chunks = read_csv_if_exists(OUTPUT_DIR / "policy_retrieval_chunks.csv")
    if results.empty:
        empty_artifact_message("policy Q&A results")
        return

    cols = st.columns(3)
    cols[0].metric("Test cases", metrics.get("test_cases", len(results)))
    cols[1].metric("Source retrieval accuracy", f"{metrics.get('source_retrieval_accuracy', 0):.1%}")
    cols[2].metric("Grounded-answer rate", f"{metrics.get('grounded_answer_rate', 0):.1%}")

    st.subheader("Evaluation trace")
    st.dataframe(results, use_container_width=True, hide_index=True)
    with st.expander("Indexed policy chunks"):
        st.dataframe(chunks, use_container_width=True, hide_index=True)
    st.info("The policy workflow retrieves source text first and is instructed to answer only from that retrieved context.")


def refund_agent_page() -> None:
    st.title("Guarded Refund Agent")
    results = read_csv_if_exists(OUTPUT_DIR / "refund_agent_results.csv")
    if results.empty:
        empty_artifact_message("refund-agent results")
        return

    cols = st.columns(3)
    cols[0].metric("Scenarios", len(results))
    cols[1].metric("Refunds created", int((results["final_status"] == "refund_created").sum()))
    cols[2].metric("Human approvals required", int((results["final_status"] == "needs_human_approval").sum()))
    st.warning("Auto-refunds are limited to INR 1000. Higher eligible amounts require human approval.")
    st.dataframe(results, use_container_width=True, hide_index=True)

    audit_path = OUTPUT_DIR / "refund_agent_audit.jsonl"
    if audit_path.exists():
        audit_rows = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if audit_rows:
            audit = pd.DataFrame(audit_rows)
            st.subheader("Agent audit trail")
            st.dataframe(audit, use_container_width=True, hide_index=True)


PAGES = {
    "Overview": overview_page,
    "Segmentation": segmentation_page,
    "Ticket Intelligence": ticket_page,
    "Policy Q&A": policy_page,
    "Refund Agent": refund_agent_page,
}

page_name = st.sidebar.radio("Navigate", list(PAGES))
st.sidebar.caption("All data is synthetic. No external API or LLM service is used.")
PAGES[page_name]()
