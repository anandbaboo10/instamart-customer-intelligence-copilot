"""Create point-in-time customer features, customer segments, and a future-purchase model.
Run from the repository root after Part A:
    python src/segment_and_model.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CLEAN_DIR = Path("data/clean")
OUTPUT_DIR = Path("outputs")
STATE_DIR = Path("state")
FEATURE_OUTPUT = CLEAN_DIR / "customer_features.csv"
MODEL_OUTPUT = STATE_DIR / "customer_model.joblib"

# Features only use orders observed before this date. The target uses the following 60 days.
CUTOFF_DATE = pd.Timestamp("2025-08-01")
HORIZON_END = pd.Timestamp("2025-10-01")


def load_clean_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = [
        CLEAN_DIR / "customers_clean.csv",
        CLEAN_DIR / "orders_clean.csv",
        CLEAN_DIR / "products_clean.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing clean data. Run src/audit_and_clean.py first: " + ", ".join(missing))

    customers = pd.read_csv(CLEAN_DIR / "customers_clean.csv")
    orders = pd.read_csv(CLEAN_DIR / "orders_clean.csv")
    products = pd.read_csv(CLEAN_DIR / "products_clean.csv")
    orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="coerce")
    return customers, orders, products


def build_features(customers: pd.DataFrame, orders: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Build customer-level features using only completed orders before CUTOFF_DATE."""
    completed = orders.loc[
        (orders["status"] == "completed")
        & (orders["order_ts"] < CUTOFF_DATE)
        & orders["order_amount"].notna()
    ].copy()
    completed = completed.merge(products[["product_id", "category"]], on="product_id", how="left")

    base = customers[["customer_id", "city", "signup_date"]].copy()
    aggregations = completed.groupby("customer_id").agg(
        historical_orders=("order_id", "nunique"),
        historical_revenue_inr=("order_amount", "sum"),
        avg_order_value_inr=("order_amount", "mean"),
        avg_delivery_minutes=("delivery_minutes", "mean"),
        avg_discount_pct=("discount_pct", "mean"),
        last_order_ts=("order_ts", "max"),
    )
    features = base.merge(aggregations, on="customer_id", how="left")

    category_counts = (
        completed.pivot_table(
            index="customer_id", columns="category", values="order_id", aggfunc="nunique", fill_value=0
        )
        .rename(columns=lambda category: f"orders_{str(category).lower().replace(' ', '_')}")
        .reset_index()
    )
    features = features.merge(category_counts, on="customer_id", how="left")

    count_columns = [column for column in features.columns if column.startswith("orders_")]
    for column in ["historical_orders", "historical_revenue_inr", *count_columns]:
        features[column] = features[column].fillna(0)

    for column in ["avg_order_value_inr", "avg_delivery_minutes", "avg_discount_pct"]:
        features[column] = features[column].fillna(0)

    features["recency_days"] = (CUTOFF_DATE - pd.to_datetime(features["last_order_ts"])).dt.days
    # Customers with no historical purchase get a conservative fixed recency value.
    features["recency_days"] = features["recency_days"].fillna(180).clip(lower=0, upper=180)
    features = features.drop(columns=["last_order_ts"])

    future = orders.loc[
        (orders["status"] == "completed")
        & (orders["order_ts"] >= CUTOFF_DATE)
        & (orders["order_ts"] < HORIZON_END)
    ]
    future_customers = set(future["customer_id"].dropna())
    features["future_purchase_60d"] = features["customer_id"].isin(future_customers).astype(int)
    return features


def add_segments(features: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Select k from 3--6 by the highest silhouette score and add KMeans labels."""
    matrix = features[feature_columns].copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)

    selection_rows = []
    fitted: dict[int, KMeans] = {}
    for k in range(3, 7):
        model = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = model.fit_predict(scaled)
        selection_rows.append(
            {"k": k, "inertia": float(model.inertia_), "silhouette_score": float(silhouette_score(scaled, labels))}
        )
        fitted[k] = model

    selection = pd.DataFrame(selection_rows)
    best_k = int(selection.sort_values(["silhouette_score", "k"], ascending=[False, True]).iloc[0]["k"])
    features = features.copy()
    features["segment"] = fitted[best_k].predict(scaled) + 1
    return features, selection, best_k


def train_purchase_model(features: pd.DataFrame, feature_columns: list[str]) -> dict[str, object]:
    """Evaluate a future-purchase classifier using a stratified held-out test set."""
    X = features[feature_columns]
    y = features["future_purchase_60d"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    importance = (
        pd.DataFrame({"feature": feature_columns, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .to_dict(orient="records")
    )
    metrics: dict[str, object] = {
        "cutoff_date": CUTOFF_DATE.strftime("%Y-%m-%d"),
        "prediction_window": f"{CUTOFF_DATE.strftime('%Y-%m-%d')} to {(HORIZON_END - pd.Timedelta(days=1)).strftime('%Y-%m-%d')}",
        "target_definition": "1 when a customer placed at least one completed order during the 60-day prediction window; otherwise 0.",
        "class_distribution_full_data": {str(key): int(value) for key, value in y.value_counts().sort_index().items()},
        "class_distribution_test_data": {str(key): int(value) for key, value in y_test.value_counts().sort_index().items()},
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "confusion_matrix_labels": [0, 1],
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        "top_feature_importances": importance[:10],
    }
    artifact = {
        "model": model,
        "feature_columns": feature_columns,
        "cutoff_date": CUTOFF_DATE.strftime("%Y-%m-%d"),
        "horizon_end": HORIZON_END.strftime("%Y-%m-%d"),
        "target_name": "future_purchase_60d",
    }
    joblib.dump(artifact, MODEL_OUTPUT)
    return metrics


def markdown_table(frame: pd.DataFrame) -> str:
    """Create a Markdown table without requiring the optional tabulate package."""
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(str(column) for column in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = [str(value).replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_segmentation_report(features: pd.DataFrame, selection: pd.DataFrame, best_k: int) -> None:
    profile = (
        features.groupby("segment")
        .agg(
            customers=("customer_id", "count"),
            avg_historical_orders=("historical_orders", "mean"),
            avg_historical_revenue_inr=("historical_revenue_inr", "mean"),
            avg_recency_days=("recency_days", "mean"),
            future_purchase_rate=("future_purchase_60d", "mean"),
        )
        .round(2)
        .reset_index()
    )
    lines = [
        "# Customer Segmentation",
        "",
        "## Method",
        "- Algorithm: KMeans after StandardScaler normalization.",
        "- Candidate segment counts: 3, 4, 5, and 6.",
        "- Selection rule: highest silhouette score; inertia is retained for comparison.",
        f"- Selected segment count: {best_k}.",
        "",
        "## Candidate metrics",
        markdown_table(selection),
        "",
        "## Segment profile",
        markdown_table(profile),
        "",
        "## Point-in-time design",
        f"- Features use completed orders strictly before {CUTOFF_DATE.strftime('%Y-%m-%d')}.",
        "- The outcome is measured only in the following 60 days, preventing future-order leakage into features.",
    ]
    (OUTPUT_DIR / "segmentation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    customers, orders, products = load_clean_data()
    features = build_features(customers, orders, products)
    feature_columns = [
        column
        for column in features.columns
        if column not in {"customer_id", "city", "signup_date", "future_purchase_60d"}
    ]
    features, selection, best_k = add_segments(features, feature_columns)
    metrics = train_purchase_model(features, feature_columns)

    features.to_csv(FEATURE_OUTPUT, index=False)
    selection.to_csv(OUTPUT_DIR / "segmentation_selection_metrics.csv", index=False)
    write_segmentation_report(features, selection, best_k)
    with (OUTPUT_DIR / "model_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(f"Customer features written: {len(features)} rows")
    print(f"Selected segment count: {best_k}")
    print(f"Future-purchase class distribution: {metrics['class_distribution_full_data']}")
    print(f"Model accuracy: {metrics['accuracy']:.3f}")
    print(f"Model F1: {metrics['f1']:.3f}")
    print("Saved model artifact: state/customer_model.joblib")


if __name__ == "__main__":
    main()
