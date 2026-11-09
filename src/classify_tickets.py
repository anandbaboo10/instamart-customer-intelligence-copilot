"""Classify support tickets using the offline mock LLM with robust safety and evaluation controls.
Run from the repository root:
    python src/classify_tickets.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

# Allow this file to run directly with: python src/classify_tickets.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mock_llm

RAW_TICKETS = Path("data/raw/tickets.csv")
OUTPUT_DIR = Path("outputs")
ALLOWED_LABELS = ["billing", "delivery", "technical", "fraud"]
MAX_WORDS = 180
MAX_ATTEMPTS = 3

EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_PATTERN = re.compile(r"(?<!\d)[6-9]\d{9}(?!\d)")

SYSTEM_ZERO_SHOT = """CLASSIFY
You are a support ticket classifier. Return JSON only, with exactly these fields:
{"category": one allowed label, "reason": short explanation}
Allowed categories: billing, delivery, technical, fraud."""

SYSTEM_FEW_SHOT = """CLASSIFY
You are a support ticket classifier. Return JSON only, with exactly these fields:
{"category": one allowed label, "reason": short explanation}
Allowed categories: billing, delivery, technical, fraud.
EXAMPLES:
Ticket: I was charged twice for my order. -> {"category":"billing","reason":"duplicate charge"}
Ticket: The rider marked my package delivered but I did not receive it. -> {"category":"delivery","reason":"delivery dispute"}
Ticket: The app crashes when I open the cart. -> {"category":"technical","reason":"application failure"}
Ticket: I did not make this transaction and my account may be hacked. -> {"category":"fraud","reason":"unauthorized activity"}"""

FALLBACK_KEYWORDS = {
    "fraud": {"unauthorized", "fraud", "hacked", "stolen"},
    "billing": {"charged", "invoice", "bill", "refund", "deducted", "payment"},
    "technical": {"app", "crash", "login", "otp", "error", "freezes"},
    "delivery": {"arrived", "rider", "delivery", "delivered", "late"},
}


def mask_pii(text: str) -> tuple[str, int]:
    """Mask email addresses and 10-digit Indian mobile numbers before model input."""
    masked, email_count = EMAIL_PATTERN.subn("[EMAIL]", text)
    masked, phone_count = PHONE_PATTERN.subn("[PHONE]", masked)
    return masked, email_count + phone_count


def truncate_text(text: str, max_words: int = MAX_WORDS) -> tuple[str, bool]:
    words = text.split()
    if len(words) <= max_words:
        return text, False
    return " ".join(words[:max_words]) + " [TRUNCATED]", True


def fallback_label(text: str) -> str:
    words = set(re.findall(r"[a-z]+", text.lower()))
    for label in ["fraud", "technical", "billing", "delivery"]:
        if FALLBACK_KEYWORDS[label] & words:
            return label
    return "technical"


def parse_response(content: str | None) -> tuple[str, str]:
    """Parse and validate the required JSON schema and category allow-list."""
    if not content:
        raise ValueError("empty model response")
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError("response is not a JSON object")
    category = parsed.get("category")
    reason = parsed.get("reason")
    if category not in ALLOWED_LABELS:
        raise ValueError(f"invalid category: {category!r}")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("missing or invalid reason")
    return category, reason.strip()


def classify_one(prepared_text: str, system_prompt: str) -> dict[str, object]:
    """Call mock_llm with transient-error retries and deterministic fallback on bad output."""
    last_error = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = mock_llm.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prepared_text},
                ]
            )
            category, reason = parse_response(response.choices[0].message.content)
            return {
                "prediction": category,
                "reason": reason,
                "attempts": attempt,
                "used_fallback": False,
                "fallback_reason": "",
            }
        except mock_llm.TransientLLMError as error:
            last_error = f"transient_error: {error}"
            if attempt < MAX_ATTEMPTS:
                time.sleep(0.05 * attempt)
        except (json.JSONDecodeError, ValueError, TypeError) as error:
            last_error = f"invalid_structured_output: {error}"
            break

    return {
        "prediction": fallback_label(prepared_text),
        "reason": "deterministic keyword fallback",
        "attempts": attempt,
        "used_fallback": True,
        "fallback_reason": last_error,
    }


def run_strategy(tickets: pd.DataFrame, name: str, system_prompt: str) -> pd.DataFrame:
    """Run one prompt strategy from a reset mock state for a fair comparison."""
    mock_llm.reset()
    rows = []
    for row in tickets.itertuples(index=False):
        source_text = "" if pd.isna(row.text) else str(row.text)
        masked_text, pii_items_masked = mask_pii(source_text)
        prepared_text, was_truncated = truncate_text(masked_text)
        result = classify_one(prepared_text, system_prompt)
        rows.append(
            {
                "ticket_id": row.ticket_id,
                f"{name}_prediction": result["prediction"],
                f"{name}_reason": result["reason"],
                f"{name}_attempts": result["attempts"],
                f"{name}_used_fallback": result["used_fallback"],
                f"{name}_fallback_reason": result["fallback_reason"],
                "masked_text": prepared_text,
                "pii_items_masked": pii_items_masked,
                "was_truncated": was_truncated,
            }
        )
    return pd.DataFrame(rows)


def metrics_for_strategy(results: pd.DataFrame, prediction_column: str, fallback_column: str) -> dict[str, object]:
    evaluable = results[results["true_category"].isin(ALLOWED_LABELS)].copy()
    accuracy = (evaluable[prediction_column] == evaluable["true_category"]).mean()
    return {
        "evaluated_tickets": int(len(evaluable)),
        "excluded_nonstandard_labels": int(len(results) - len(evaluable)),
        "accuracy": round(float(accuracy), 4),
        "fallback_count": int(results[fallback_column].sum()),
        "fallback_rate": round(float(results[fallback_column].mean()), 4),
    }


def main() -> None:
    if not RAW_TICKETS.exists():
        raise FileNotFoundError("Missing raw tickets file. Run generate_data.py first.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tickets = pd.read_csv(RAW_TICKETS)
    required_columns = {"ticket_id", "text", "true_category"}
    if not required_columns.issubset(tickets.columns):
        raise ValueError(f"tickets.csv must contain: {sorted(required_columns)}")

    zero_shot = run_strategy(tickets, "zero_shot", SYSTEM_ZERO_SHOT)
    few_shot = run_strategy(tickets, "few_shot", SYSTEM_FEW_SHOT)
    results = tickets[["ticket_id", "customer_id", "order_id", "true_category"]].merge(
        zero_shot, on="ticket_id", how="left"
    ).merge(
        few_shot.drop(columns=["masked_text", "pii_items_masked", "was_truncated"]), on="ticket_id", how="left")

    zero_metrics = metrics_for_strategy(results, "zero_shot_prediction", "zero_shot_used_fallback")
    few_metrics = metrics_for_strategy(results, "few_shot_prediction", "few_shot_used_fallback")
    comparison = {
        "zero_shot": zero_metrics,
        "few_shot": few_metrics,
        "accuracy_change_few_shot_minus_zero_shot": round(few_metrics["accuracy"] - zero_metrics["accuracy"], 4),
        "safety_controls": {
            "pii_masking": "Emails and 10-digit mobile numbers are replaced before model calls.",
            "input_limit": f"Inputs are truncated to {MAX_WORDS} words after masking.",
            "structured_output": "JSON is parsed and category labels are validated against an allow-list.",
            "retry_policy": f"Transient model failures receive up to {MAX_ATTEMPTS} attempts.",
            "fallback": "Malformed output, invalid labels, and exhausted retries use deterministic keyword fallback.",
        },
    }

    results.to_csv(OUTPUT_DIR / "ticket_classification_results.csv", index=False)
    with (OUTPUT_DIR / "ticket_classification_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(comparison, file, indent=2)

    report = [
        "# Ticket Classification Evaluation",
        "",
        "## Evaluation design",
        "- Ground truth is the synthetic `true_category` field.",
        "- Tickets whose synthetic label is `none` are excluded from accuracy because `none` is not an allowed production class.",
        "- Both strategies are run from a reset mock state so the simulated transient errors are comparable.",
        "",
        "## Results",
        "| Strategy | Evaluated tickets | Accuracy | Fallback count | Fallback rate |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| Zero-shot | {zero_metrics['evaluated_tickets']} | {zero_metrics['accuracy']:.2%} | {zero_metrics['fallback_count']} | {zero_metrics['fallback_rate']:.2%} |",
        f"| Few-shot | {few_metrics['evaluated_tickets']} | {few_metrics['accuracy']:.2%} | {few_metrics['fallback_count']} | {few_metrics['fallback_rate']:.2%} |",
        "",
        f"Few-shot accuracy change versus zero-shot: {comparison['accuracy_change_few_shot_minus_zero_shot']:+.2%}.",
        "",
        "## Reliability and safety controls",
        "- Email addresses and phone numbers are masked before inference.",
        f"- Inputs are limited to {MAX_WORDS} words after masking; long tickets are explicitly marked as truncated.",
        "- Each response must parse as JSON and contain an allowed category and a non-empty reason.",
        f"- Transient errors receive up to {MAX_ATTEMPTS} attempts before fallback.",
        "- Invalid JSON, invalid category values, and exhausted retries use a deterministic keyword fallback and are logged in the result file.",
    ]
    (OUTPUT_DIR / "classification_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"Tickets processed: {len(results)}")
    print(f"Zero-shot accuracy: {zero_metrics['accuracy']:.3f}; fallbacks: {zero_metrics['fallback_count']}")
    print(f"Few-shot accuracy: {few_metrics['accuracy']:.3f}; fallbacks: {few_metrics['fallback_count']}")
    print("Wrote outputs/ticket_classification_results.csv and outputs/classification_report.md")


if __name__ == "__main__":
    main()