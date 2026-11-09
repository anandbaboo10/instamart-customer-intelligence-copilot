"""Create the synthetic policy documents used by the grounded policy QA workflow.
Run from the repository root:
    python src/create_policy_docs.py
"""
from pathlib import Path

POLICY_DIR = Path("data/policy")

POLICIES = {
    "refund_policy.txt": (
        "Refund eligibility policy: only completed orders are eligible for a refund. "
        "For delivery delays greater than 45 minutes, customers may receive a 50% refund. "
        "Automatic refunds are capped at INR 1000; any higher amount requires human approval. "
        "Cancelled orders are never eligible for a refund."
    ),
    "delivery_sla.txt": (
        "Delivery service-level agreement: the target delivery time is 20 to 35 minutes. "
        "An order is considered delayed when delivery exceeds 45 minutes. "
        "Agents may apologize for delays but must not promise compensation beyond the refund policy."
    ),
    "privacy.txt": (
        "Privacy policy: agents must mask email addresses and phone numbers in customer-facing records. "
        "Agents must never disclose full payment instrument numbers, passwords, OTPs, or authentication secrets. "
        "Only the minimum customer information needed to resolve the request may be used."
    ),
    "escalation.txt": (
        "Escalation policy: refund requests above INR 1000 require approval from a human lead. "
        "Fraud, account takeover, legal threat, and data privacy incident claims must be escalated immediately. "
        "Support agents should document the reason for escalation and avoid making unapproved commitments."
    ),
}


def main() -> None:
    POLICY_DIR.mkdir(parents=True, exist_ok=True)
    for filename, content in POLICIES.items():
        (POLICY_DIR / filename).write_text(content + "\n", encoding="utf-8")
        print(f"wrote: {POLICY_DIR / filename}")
    print(f"Policy documents created: {len(POLICIES)}")


if __name__ == "__main__":
    main()
