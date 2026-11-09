# Refund Agent Simulation

## Tools
- `lookup_order`: obtains order status, amount, and delivery time from the clean SQLite data.
- `check_refund_eligibility`: applies the refund and escalation policy without taking financial action.
- `create_refund`: creates a local ledger record only for an approved positive refund at or below INR 1000.

## Guardrails
- Maximum loop iterations per scenario: 6.
- Duplicate tool calls with identical arguments are blocked.
- Only completed orders delayed more than 45 minutes can qualify.
- The refund is 50% of order value; values above INR 1000 require human approval and are not auto-created.
- The ledger blocks a second refund for the same order.

## Scenario results
| Scenario | Order | Final status | Reason | Proposed refund (INR) | Refund ID | Iterations |
| --- | --- | --- | --- | ---: | --- | ---: |
| late_delivery_auto_refund | ORD-107504 | refund_created | automatic_refund_created_within_policy_limit | 7.50 | R-F657ACC4EF | 3 |
| late_delivery_above_limit | ORD-102879 | needs_human_approval | proposed_refund_exceeds_inr_1000_auto_approval_limit | 1999.00 |  | 3 |
| cancelled_order_refund_request | ORD-100020 | deny | only_completed_orders_are_eligible |  |  | 3 |
| on_time_order_refund_request | ORD-108395 | deny | delivery_delay_not_greater_than_45_minutes |  |  | 3 |

## Auditability
Every tool call, tool result, guardrail stop, and final decision is recorded as a JSON line in `outputs/refund_agent_audit.jsonl`.
The local refund ledger is stored in `state/refund_ledger.json` for duplicate-refund protection across runs.
