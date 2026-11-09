"""Simulate a safe multi-tool refund agent over clean Instamart orders.
Run from the repository root:
    python src/refund_agent.py
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

DB_PATH = Path("data/clean/instamart.db")
SCENARIO_PATH = Path("data/scenarios/refund_scenarios.json")
OUTPUT_DIR = Path("outputs")
STATE_DIR = Path("state")
LEDGER_PATH = STATE_DIR / "refund_ledger.json"
AUDIT_PATH = OUTPUT_DIR / "refund_agent_audit.jsonl"
MAX_ITERATIONS = 6
MAX_SAME_ACTION_SIGNATURE = 1
AUTO_REFUND_LIMIT_INR = 1000.0


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_orders() -> pd.DataFrame:
    if not DB_PATH.exists():
        raise FileNotFoundError("Missing clean SQLite data. Run src/audit_and_clean.py first.")
    with sqlite3.connect(DB_PATH) as conn:
        orders = pd.read_sql_query("SELECT * FROM clean_orders", conn)
    required = {"order_id", "customer_id", "status", "order_amount", "delivery_minutes"}
    if not required.issubset(orders.columns):
        raise ValueError(f"clean_orders is missing required columns: {sorted(required - set(orders.columns))}")
    return orders


def build_scenarios(orders: pd.DataFrame, already_refunded: set[str]) -> list[dict[str, str]]:
    """Create four reproducible test scenarios selected from the clean order data."""
    completed = orders[(orders["status"] == "completed") & orders["order_amount"].notna()].copy()
    late = completed[completed["delivery_minutes"] > 45].sort_values("order_amount")
    cancelled = orders[orders["status"] == "cancelled"]
    on_time = completed[completed["delivery_minutes"].between(8, 45)].sort_values("order_amount")
    if late.empty or cancelled.empty or on_time.empty:
        raise ValueError("Cannot construct refund scenarios from the available clean order data.")

    # Prefer a qualifying order that has not already received a local refund on an earlier run.
    available_late = late[~late["order_id"].astype(str).isin(already_refunded)]
    if available_late.empty:
        available_late = late
    low_late = available_late[available_late["order_amount"] <= (AUTO_REFUND_LIMIT_INR * 2)]
    low_late_order = low_late.iloc[0] if not low_late.empty else available_late.iloc[0]
    high_late = late[(late["order_amount"] > (AUTO_REFUND_LIMIT_INR * 2)) & (late["order_id"] != low_late_order["order_id"])]
    high_late_order = high_late.iloc[-1] if not high_late.empty else late.iloc[-1]

    scenarios = [
        {
            "scenario_id": "late_delivery_auto_refund",
            "order_id": str(low_late_order["order_id"]),
            "customer_request": "My completed order arrived very late. Please process the applicable refund.",
        },
        {
            "scenario_id": "late_delivery_above_limit",
            "order_id": str(high_late_order["order_id"]),
            "customer_request": "My completed order arrived late. I want the policy refund.",
        },
        {
            "scenario_id": "cancelled_order_refund_request",
            "order_id": str(cancelled.iloc[0]["order_id"]),
            "customer_request": "Please refund my cancelled order.",
        },
        {
            "scenario_id": "on_time_order_refund_request",
            "order_id": str(on_time.iloc[0]["order_id"]),
            "customer_request": "I would like a refund for this completed order.",
        },
    ]
    return scenarios


def write_scenarios(scenarios: list[dict[str, str]]) -> None:
    SCENARIO_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCENARIO_PATH.write_text(json.dumps(scenarios, indent=2) + "\n", encoding="utf-8")


def load_ledger() -> list[dict[str, Any]]:
    if not LEDGER_PATH.exists():
        return []
    return json.loads(LEDGER_PATH.read_text(encoding="utf-8"))


def save_ledger(ledger: list[dict[str, Any]]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    LEDGER_PATH.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")


def lookup_order(orders: pd.DataFrame, order_id: str) -> dict[str, Any]:
    found = orders.loc[orders["order_id"] == order_id]
    if found.empty:
        return {"found": False, "order_id": order_id}
    row = found.iloc[0]
    return {
        "found": True,
        "order_id": str(row["order_id"]),
        "customer_id": str(row["customer_id"]),
        "status": str(row["status"]),
        "order_amount": None if pd.isna(row["order_amount"]) else round(float(row["order_amount"]), 2),
        "delivery_minutes": None if pd.isna(row["delivery_minutes"]) else int(row["delivery_minutes"]),
    }


def check_refund_eligibility(order: dict[str, Any]) -> dict[str, Any]:
    """Apply the documented refund policy. This function never creates a refund."""
    if not order.get("found"):
        return {"eligible": False, "decision": "deny", "reason": "order_not_found", "policy_ref": "refund_policy.txt"}
    if order["status"] != "completed":
        return {"eligible": False, "decision": "deny", "reason": "only_completed_orders_are_eligible", "policy_ref": "refund_policy.txt"}
    if order["delivery_minutes"] is None or order["delivery_minutes"] <= 45:
        return {"eligible": False, "decision": "deny", "reason": "delivery_delay_not_greater_than_45_minutes", "policy_ref": "refund_policy.txt"}
    if order["order_amount"] is None:
        return {"eligible": False, "decision": "deny", "reason": "missing_order_amount", "policy_ref": "refund_policy.txt"}

    proposed_refund = round(order["order_amount"] * 0.50, 2)
    if proposed_refund > AUTO_REFUND_LIMIT_INR:
        return {
            "eligible": True,
            "decision": "needs_human_approval",
            "reason": "proposed_refund_exceeds_inr_1000_auto_approval_limit",
            "proposed_refund_inr": proposed_refund,
            "policy_ref": "refund_policy.txt; escalation.txt",
        }
    return {
        "eligible": True,
        "decision": "approve_automatic_refund",
        "reason": "completed_order_delayed_more_than_45_minutes",
        "proposed_refund_inr": proposed_refund,
        "policy_ref": "refund_policy.txt",
    }


def create_refund(order_id: str, amount_inr: float, ledger: list[dict[str, Any]]) -> dict[str, Any]:
    """Create a ledger entry only when all financial guardrails are satisfied."""
    if amount_inr <= 0:
        return {"created": False, "reason": "refund_amount_must_be_positive"}
    if amount_inr > AUTO_REFUND_LIMIT_INR:
        return {"created": False, "reason": "refund_exceeds_auto_approval_limit"}
    if any(entry["order_id"] == order_id for entry in ledger):
        return {"created": False, "reason": "refund_already_recorded_for_order"}

    refund = {
        "refund_id": f"R-{uuid.uuid4().hex[:10].upper()}",
        "order_id": order_id,
        "amount_inr": round(amount_inr, 2),
        "created_at_utc": now_utc(),
        "status": "created",
    }
    ledger.append(refund)
    save_ledger(ledger)
    return {"created": True, **refund}


def log_event(run_id: str, scenario_id: str, iteration: int, event: str, details: dict[str, Any]) -> None:
    record = {
        "timestamp_utc": now_utc(),
        "run_id": run_id,
        "scenario_id": scenario_id,
        "iteration": iteration,
        "event": event,
        "details": details,
    }
    with AUDIT_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")


def execute_scenario(scenario: dict[str, str], orders: pd.DataFrame, ledger: list[dict[str, Any]]) -> dict[str, Any]:
    """Execute a bounded agent loop with explicit tool selection and duplicate-call protection."""
    run_id = uuid.uuid4().hex
    action_counts: Counter[str] = Counter()
    order: dict[str, Any] | None = None
    eligibility: dict[str, Any] | None = None
    final_status = "stopped"
    final_reason = "iteration_limit_reached"
    refund_id = ""

    for iteration in range(1, MAX_ITERATIONS + 1):
        if order is None:
            action, arguments = "lookup_order", {"order_id": scenario["order_id"]}
        elif eligibility is None:
            action, arguments = "check_refund_eligibility", {"order_id": scenario["order_id"]}
        elif eligibility["decision"] == "approve_automatic_refund":
            action, arguments = "create_refund", {
                "order_id": scenario["order_id"],
                "amount_inr": eligibility["proposed_refund_inr"],
            }
        else:
            final_status = eligibility["decision"]
            final_reason = eligibility["reason"]
            log_event(run_id, scenario["scenario_id"], iteration, "final_decision", {"status": final_status, "reason": final_reason})
            break

        signature = json.dumps({"action": action, "arguments": arguments}, sort_keys=True)
        action_counts[signature] += 1
        if action_counts[signature] > MAX_SAME_ACTION_SIGNATURE:
            final_status = "stopped"
            final_reason = "duplicate_tool_call_blocked"
            log_event(run_id, scenario["scenario_id"], iteration, "guardrail_stop", {"reason": final_reason, "action": action})
            break

        log_event(run_id, scenario["scenario_id"], iteration, "tool_call", {"action": action, "arguments": arguments})
        if action == "lookup_order":
            order = lookup_order(orders, scenario["order_id"])
            log_event(run_id, scenario["scenario_id"], iteration, "tool_result", order)
        elif action == "check_refund_eligibility":
            eligibility = check_refund_eligibility(order or {})
            log_event(run_id, scenario["scenario_id"], iteration, "tool_result", eligibility)
        elif action == "create_refund":
            refund_result = create_refund(arguments["order_id"], float(arguments["amount_inr"]), ledger)
            log_event(run_id, scenario["scenario_id"], iteration, "tool_result", refund_result)
            if refund_result["created"]:
                refund_id = refund_result["refund_id"]
                final_status = "refund_created"
                final_reason = "automatic_refund_created_within_policy_limit"
            else:
                final_status = "refund_not_created"
                final_reason = refund_result["reason"]
            log_event(run_id, scenario["scenario_id"], iteration, "final_decision", {"status": final_status, "reason": final_reason})
            break

    return {
        "scenario_id": scenario["scenario_id"],
        "order_id": scenario["order_id"],
        "customer_request": scenario["customer_request"],
        "final_status": final_status,
        "final_reason": final_reason,
        "refund_id": refund_id,
        "proposed_refund_inr": None if eligibility is None else eligibility.get("proposed_refund_inr"),
        "iterations": iteration,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    orders = load_orders()
    ledger = load_ledger()
    already_refunded = {str(entry["order_id"]) for entry in ledger if entry.get("status") == "created"}
    scenarios = build_scenarios(orders, already_refunded)
    write_scenarios(scenarios)
    AUDIT_PATH.write_text("", encoding="utf-8")

    results = [execute_scenario(scenario, orders, ledger) for scenario in scenarios]
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "refund_agent_results.csv", index=False)

    lines = [
        "# Refund Agent Simulation",
        "",
        "## Tools",
        "- `lookup_order`: obtains order status, amount, and delivery time from the clean SQLite data.",
        "- `check_refund_eligibility`: applies the refund and escalation policy without taking financial action.",
        "- `create_refund`: creates a local ledger record only for an approved positive refund at or below INR 1000.",
        "",
        "## Guardrails",
        f"- Maximum loop iterations per scenario: {MAX_ITERATIONS}.",
        "- Duplicate tool calls with identical arguments are blocked.",
        "- Only completed orders delayed more than 45 minutes can qualify.",
        "- The refund is 50% of order value; values above INR 1000 require human approval and are not auto-created.",
        "- The ledger blocks a second refund for the same order.",
        "",
        "## Scenario results",
        "| Scenario | Order | Final status | Reason | Proposed refund (INR) | Refund ID | Iterations |",
        "| --- | --- | --- | --- | ---: | --- | ---: |",
    ]
    for row in results_df.itertuples(index=False):
        amount = "" if pd.isna(row.proposed_refund_inr) else f"{row.proposed_refund_inr:.2f}"
        lines.append(
            f"| {row.scenario_id} | {row.order_id} | {row.final_status} | {row.final_reason} | {amount} | {row.refund_id} | {row.iterations} |"
        )
    lines.extend([
        "",
        "## Auditability",
        "Every tool call, tool result, guardrail stop, and final decision is recorded as a JSON line in `outputs/refund_agent_audit.jsonl`.",
        "The local refund ledger is stored in `state/refund_ledger.json` for duplicate-refund protection across runs.",
    ])
    (OUTPUT_DIR / "agent_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Scenarios executed: {len(results_df)}")
    print(results_df[["scenario_id", "final_status", "final_reason"]].to_string(index=False))
    print("Wrote outputs/refund_agent_results.csv, outputs/refund_agent_audit.jsonl, and outputs/agent_report.md")


if __name__ == "__main__":
    main()
