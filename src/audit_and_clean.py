"""Audit, clean, validate, and load Instamart synthetic data into SQLite.
Run from the repository root:
    python src/audit_and_clean.py
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/clean")
OUTPUT_DIR = Path("outputs")
DB_PATH = CLEAN_DIR / "instamart.db"


def as_int(value: object) -> int:
    """Convert pandas/numpy counts to normal Python integers for JSON output."""
    return int(value)


def load_raw_data() -> dict[str, pd.DataFrame]:
    files = {
        "customers": "customers.csv",
        "products": "products.csv",
        "orders": "orders.csv",
        "tickets": "tickets.csv",
    }
    missing = [filename for filename in files.values() if not (RAW_DIR / filename).exists()]
    if missing:
        raise FileNotFoundError(f"Missing raw data files: {', '.join(missing)}. Run generate_data.py first.")
    return {name: pd.read_csv(RAW_DIR / filename) for name, filename in files.items()}


def raw_audit(data: dict[str, pd.DataFrame]) -> dict[str, object]:
    """Profile raw data before applying any transformations."""
    customers = data["customers"].copy()
    products = data["products"].copy()
    orders = data["orders"].copy()
    tickets = data["tickets"].copy()

    age_num = pd.to_numeric(customers["age"], errors="coerce")
    amount_num = pd.to_numeric(orders["order_amount"], errors="coerce")
    delivery_num = pd.to_numeric(orders["delivery_minutes"], errors="coerce")

    report: dict[str, object] = {
        "row_counts": {name: as_int(len(frame)) for name, frame in data.items()},
        "null_counts": {
            name: {column: as_int(count) for column, count in frame.isna().sum().items()}
            for name, frame in data.items()
        },
        "duplicate_rows": {name: as_int(frame.duplicated().sum()) for name, frame in data.items()},
        "key_quality": {
            "customers_duplicate_customer_ids": as_int(customers["customer_id"].duplicated().sum()),
            "products_duplicate_product_ids": as_int(products["product_id"].duplicated().sum()),
            "orders_duplicate_order_ids": as_int(orders["order_id"].duplicated().sum()),
            "tickets_duplicate_ticket_ids": as_int(tickets["ticket_id"].duplicated().sum()),
        },
        "invalid_values": {
            "customer_age_outside_13_to_100": as_int((age_num.notna() & ~age_num.between(13, 100)).sum()),
            "order_amount_unparseable": as_int((orders["order_amount"].notna() & amount_num.isna()).sum()),
            "order_amount_not_positive": as_int((amount_num.notna() & (amount_num <= 0)).sum()),
            "delivery_minutes_negative": as_int((delivery_num.notna() & (delivery_num < 0)).sum()),
            "completed_orders_with_delivery_outside_8_to_75": as_int(
                ((orders["status"].astype(str).str.lower() == "completed")
                 & delivery_num.notna() & ~delivery_num.between(8, 75)).sum()
            ),
        },
        "referential_integrity_raw": {
            "orders_customer_ids_not_in_customers": as_int((~orders["customer_id"].isin(customers["customer_id"])).sum()),
            "orders_product_ids_not_in_products": as_int((~orders["product_id"].isin(products["product_id"])).sum()),
            "tickets_customer_ids_not_in_customers": as_int((~tickets["customer_id"].isin(customers["customer_id"])).sum()),
            "tickets_nonblank_order_ids_not_in_orders": as_int(
                (~tickets.loc[tickets["order_id"].notna(), "order_id"].isin(orders["order_id"])).sum()
            ),
        },
    }
    return report


def clean_customers(customers: pd.DataFrame) -> pd.DataFrame:
    clean = customers.copy()
    clean["city"] = clean["city"].astype("string").str.strip().str.title()
    clean["signup_date"] = pd.to_datetime(clean["signup_date"], errors="coerce")

    gender = clean["gender"].astype("string").str.strip().str.lower()
    clean["gender"] = gender.map({"m": "male", "male": "male", "f": "female", "female": "female"}).astype("string")

    age = pd.to_numeric(clean["age"], errors="coerce")
    clean["age"] = age.where(age.between(13, 100)).round().astype("Int64")
    return clean


def clean_products(products: pd.DataFrame) -> pd.DataFrame:
    clean = products.copy()
    clean["product_name"] = clean["product_name"].astype("string").str.strip()
    clean["category"] = clean["category"].astype("string").str.strip().str.title()
    price = pd.to_numeric(clean["price_inr"], errors="coerce")
    clean["price_inr"] = price.where(price > 0).round(2)
    return clean


def clean_orders(orders: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    clean = orders.copy()
    exact_duplicates_removed = as_int(clean.duplicated().sum())
    clean = clean.drop_duplicates().copy()

    clean["order_ts"] = pd.to_datetime(clean["order_ts"], errors="coerce")
    clean["status"] = clean["status"].astype("string").str.strip().str.lower()
    clean["quantity"] = pd.to_numeric(clean["quantity"], errors="coerce").astype("Int64")
    clean["discount_pct"] = pd.to_numeric(clean["discount_pct"], errors="coerce")
    clean["discount_pct"] = clean["discount_pct"].where(clean["discount_pct"].between(0, 100))

    amount = pd.to_numeric(clean["order_amount"], errors="coerce")
    clean["order_amount"] = amount.where(amount > 0).round(2)

    delivery = pd.to_numeric(clean["delivery_minutes"], errors="coerce")
    completed_valid_delivery = (clean["status"] == "completed") & delivery.between(8, 75)
    clean["delivery_minutes"] = delivery.where(completed_valid_delivery).round().astype("Int64")

    # The source intentionally includes an earlier pending version of some orders.
    # Keep the latest timestamped version of each order after removing exact duplicates.
    before_order_id_dedup = len(clean)
    clean = clean.sort_values(["order_id", "order_ts"], kind="mergesort").drop_duplicates("order_id", keep="last")
    order_id_versions_removed = as_int(before_order_id_dedup - len(clean))
    clean = clean.sort_values("order_id").reset_index(drop=True)

    return clean, {
        "exact_duplicate_rows_removed": exact_duplicates_removed,
        "earlier_order_versions_removed": order_id_versions_removed,
    }


def clean_tickets(tickets: pd.DataFrame) -> pd.DataFrame:
    clean = tickets.copy()
    for column in ["customer_id", "order_id", "text", "true_category"]:
        clean[column] = clean[column].astype("string").str.strip()
    clean["order_id"] = clean["order_id"].replace("", pd.NA)
    clean["text"] = clean["text"].replace("", pd.NA)
    clean["true_category"] = clean["true_category"].replace("", pd.NA)
    return clean.drop_duplicates().sort_values("ticket_id").reset_index(drop=True)


def validate_clean_data(data: dict[str, pd.DataFrame]) -> dict[str, object]:
    customers = data["customers"]
    products = data["products"]
    orders = data["orders"]
    tickets = data["tickets"]

    invalid_delivery = orders["delivery_minutes"].notna() & ~orders["delivery_minutes"].between(8, 75)
    invalid_age = customers["age"].notna() & ~customers["age"].between(13, 100)
    duplicate_primary_keys = {
        "customers": as_int(customers["customer_id"].duplicated().sum()),
        "products": as_int(products["product_id"].duplicated().sum()),
        "orders": as_int(orders["order_id"].duplicated().sum()),
        "tickets": as_int(tickets["ticket_id"].duplicated().sum()),
    }

    checks: dict[str, object] = {
        "row_counts": {name: as_int(len(frame)) for name, frame in data.items()},
        "duplicate_primary_keys": duplicate_primary_keys,
        "invalid_customer_ages": as_int(invalid_age.sum()),
        "invalid_nonnull_delivery_minutes": as_int(invalid_delivery.sum()),
        "invalid_nonpositive_order_amounts": as_int((orders["order_amount"].notna() & (orders["order_amount"] <= 0)).sum()),
        "orders_customer_ids_not_in_customers": as_int((~orders["customer_id"].isin(customers["customer_id"])).sum()),
        "orders_product_ids_not_in_products": as_int((~orders["product_id"].isin(products["product_id"])).sum()),
        "tickets_customer_ids_not_in_customers": as_int((~tickets["customer_id"].isin(customers["customer_id"])).sum()),
        "tickets_nonblank_order_ids_not_in_orders": as_int(
            (~tickets.loc[tickets["order_id"].notna(), "order_id"].isin(orders["order_id"])).sum()
        ),
    }
    checks["passed"] = all(value == 0 for value in duplicate_primary_keys.values()) and all(
        checks[key] == 0
        for key in [
            "invalid_customer_ages",
            "invalid_nonnull_delivery_minutes",
            "invalid_nonpositive_order_amounts",
            "orders_customer_ids_not_in_customers",
            "orders_product_ids_not_in_products",
            "tickets_customer_ids_not_in_customers",
            "tickets_nonblank_order_ids_not_in_orders",
        ]
    )
    return checks


def write_sqlite(data: dict[str, pd.DataFrame]) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        for name, frame in data.items():
            db_frame = frame.copy()
            for column in db_frame.select_dtypes(include=["datetime64[ns]"]).columns:
                db_frame[column] = db_frame[column].dt.strftime("%Y-%m-%d %H:%M:%S")
            db_frame.to_sql(f"clean_{name}", conn, if_exists="replace", index=False)

        conn.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_clean_orders_customer_id ON clean_orders(customer_id);
            CREATE INDEX IF NOT EXISTS idx_clean_orders_product_id ON clean_orders(product_id);
            CREATE INDEX IF NOT EXISTS idx_clean_orders_order_ts ON clean_orders(order_ts);
            CREATE INDEX IF NOT EXISTS idx_clean_tickets_customer_id ON clean_tickets(customer_id);
            """
        )


def write_report(raw: dict[str, object], actions: dict[str, int], validation: dict[str, object]) -> None:
    with (OUTPUT_DIR / "audit_summary.json").open("w", encoding="utf-8") as file:
        json.dump({"raw_audit": raw, "cleaning_actions": actions, "validation": validation}, file, indent=2)

    lines = [
        "# Instamart Data Quality Report",
        "",
        "## Raw-data audit",
        f"- Raw rows: {raw['row_counts']}",
        f"- Exact duplicate rows: {raw['duplicate_rows']}",
        f"- Invalid values: {raw['invalid_values']}",
        "",
        "## Cleaning decisions",
        "- City values were trimmed and standardized to title case.",
        "- Gender variants such as M/m/MALE and F/f/FEMALE were standardized to male/female.",
        "- Ages outside 13–100 were set to missing rather than imputed.",
        "- Unparseable or non-positive order amounts were set to missing rather than invented.",
        "- Delivery minutes outside 8–75 for completed orders, including negative values, were set to missing.",
        "- Exact duplicate order rows were removed; for repeated order IDs, the latest timestamped version was retained.",
        f"- Exact duplicate order rows removed: {actions['exact_duplicate_rows_removed']}",
        f"- Earlier order versions removed: {actions['earlier_order_versions_removed']}",
        "",
        "## Validation",
        f"- Passed: {validation['passed']}",
        f"- Clean row counts: {validation['row_counts']}",
        f"- Duplicate primary keys: {validation['duplicate_primary_keys']}",
        f"- Invalid non-null delivery values: {validation['invalid_nonnull_delivery_minutes']}",
        f"- Foreign-key violations: orders→customers={validation['orders_customer_ids_not_in_customers']}, "
        f"orders→products={validation['orders_product_ids_not_in_products']}, "
        f"tickets→customers={validation['tickets_customer_ids_not_in_customers']}, "
        f"tickets→orders={validation['tickets_nonblank_order_ids_not_in_orders']}",
        "",
        "## Outputs",
        "- Clean CSVs: data/clean/",
        "- SQLite database: data/clean/instamart.db",
        "- Machine-readable audit: outputs/audit_summary.json",
    ]
    (OUTPUT_DIR / "data_quality_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_data = load_raw_data()
    audit = raw_audit(raw_data)

    customers = clean_customers(raw_data["customers"])
    products = clean_products(raw_data["products"])
    orders, cleaning_actions = clean_orders(raw_data["orders"])
    tickets = clean_tickets(raw_data["tickets"])
    clean_data = {"customers": customers, "products": products, "orders": orders, "tickets": tickets}

    for name, frame in clean_data.items():
        output = CLEAN_DIR / f"{name}_clean.csv"
        frame.to_csv(output, index=False, na_rep="")

    write_sqlite(clean_data)
    validation = validate_clean_data(clean_data)
    write_report(audit, cleaning_actions, validation)

    print("Raw data audit completed.")
    print(f"Clean CSVs written to: {CLEAN_DIR}")
    print(f"SQLite database written to: {DB_PATH}")
    print(f"Validation passed: {validation['passed']}")
    print(json.dumps({"cleaning_actions": cleaning_actions, "validation": validation}, indent=2))


if __name__ == "__main__":
    main()
