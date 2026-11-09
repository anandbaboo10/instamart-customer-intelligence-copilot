# Instamart Customer Intelligence Copilot

A reproducible, synthetic-data customer-intelligence project for Instamart. The project combines data-quality controls, customer segmentation, future-purchase prediction, safe support-ticket classification, grounded policy Q&A, and a guarded refund-agent simulation.

> **Data notice:** All customers, products, orders, tickets, policies, and scenarios are synthetic. The repository is designed for demonstration and evaluation only.

## Capabilities

| Component | What it does | Key outputs |
| --- | --- | --- |
| Data pipeline | Audits raw data, cleans invalid values, deduplicates orders, validates relationships, and loads SQLite | Clean CSVs, SQLite database, audit report |
| Segmentation and prediction | Creates point-in-time customer features, selects a KMeans segment count, and predicts future purchase | Segment report, metrics, serialized model |
| Ticket intelligence | Classifies support tickets with PII masking, structured-output validation, retries, fallback, and prompt comparison | Classification results and evaluation report |
| Grounded policy Q&A | Retrieves policy text before answering and evaluates source retrieval and groundedness | Retrieval chunks, question-answer results, evaluation report |
| Refund agent | Simulates order lookup, policy eligibility, restricted refund creation, loop controls, and audit logging | Scenario results, JSONL audit trail, local ledger |
| Dashboard | Presents outputs in a five-page Streamlit application | Overview, segmentation, ticket, policy, and agent pages |

## Project structure

```text
app/                         Streamlit dashboard
src/                         Reproducible pipeline scripts
data/raw/                    Synthetic source CSV files
data/clean/                  Clean CSV files, features, and SQLite database
data/policy/                 Synthetic policy source documents
data/scenarios/              Generated refund-agent scenarios
outputs/                     Reports, metrics, result files, and audit logs
state/                       Serialized model and local refund ledger
sql/                          Reusable SQLite analysis queries
```

## Setup

Use Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce the project

Run commands from the repository root in this order:

```bash
python generate_data.py
python src/create_policy_docs.py
python src/audit_and_clean.py
python src/segment_and_model.py
python src/classify_tickets.py
python src/policy_qa.py
python src/refund_agent.py
```

The data generator uses a fixed random seed. The policy generator creates the local policy documents needed by the grounded Q&A workflow.

## Run the dashboard

```bash
streamlit run app/dashboard.py
```

The dashboard has these pages:

1. **Overview** — clean-data KPIs, monthly revenue, category revenue, and data-quality status.
2. **Segmentation** — customer segment profiles, KMeans diagnostics, and purchase-model metrics.
3. **Ticket Intelligence** — zero-shot versus few-shot results, fallback visibility, and masked ticket text.
4. **Policy Q&A** — retrieved policy-source traceability and grounded-answer evaluation.
5. **Refund Agent** — scenario outcomes, approval-limit controls, and detailed audit events.

## Data-quality methodology

The audit and cleaning pipeline performs the following controls:

- Profiles raw row counts, nulls, duplicate rows, invalid values, and referential integrity.
- Standardizes city and gender values.
- Sets impossible ages, invalid delivery values, and unparseable/non-positive order amounts to missing rather than fabricating values.
- Removes exact duplicate order rows.
- Resolves repeated order IDs by retaining the latest timestamped version.
- Validates primary-key uniqueness, foreign-key relationships, positive non-null amounts, valid non-null delivery times, and valid non-null ages.
- Loads the clean tables into SQLite: `clean_customers`, `clean_products`, `clean_orders`, and `clean_tickets`.

Reusable SQL examples are in `sql/analysis_queries.sql`.

## Modeling methodology

### Customer segmentation

KMeans is evaluated for segment counts 3 through 6 after feature scaling. The chosen value of `k` has the best silhouette score. Segment profiling includes customer count, historical order activity, historical revenue, recency, and observed future-purchase rate.

### Future-purchase prediction

Features use completed orders strictly before **August 1, 2025**. The label is whether a customer placed at least one completed order from **August 1 through September 30, 2025**. This point-in-time design prevents future-label information from leaking into the features.

The workflow uses a stratified held-out test split and reports class distribution, accuracy, precision, recall, F1, a confusion matrix, and feature importance. The model artifact is saved with its feature list and date-window metadata.

## Ticket-classification safety controls

The ticket workflow uses only the offline `mock_llm.py` module. No external LLM API key or network access is required.

- Emails and 10-digit mobile numbers are masked before inference.
- Inputs are truncated after masking.
- The expected response is JSON containing an allowed category and a short reason.
- JSON parsing and category allow-list validation are enforced.
- Simulated transient failures receive bounded retries.
- Invalid output or exhausted retries use a deterministic fallback classifier.
- Zero-shot and few-shot prompt strategies are evaluated against synthetic ground truth. Non-production `none` labels are excluded from the four-class accuracy calculation.

## Grounded policy Q&A controls

The policy workflow indexes the local synthetic policy documents as sentence chunks with source filenames. It retrieves top matching chunks with TF-IDF before calling the policy assistant. The assistant is instructed to use only supplied context and to refuse when no relevant context is found. The evaluation report measures both source retrieval accuracy and whether each answer is supported by the retrieved chunk.

## Refund-agent controls

The refund agent has three tools:

1. `lookup_order`
2. `check_refund_eligibility`
3. `create_refund`

The agent only auto-creates a refund if all controls pass:

- Order status is completed.
- Delivery took more than 45 minutes.
- Refund amount equals 50% of the order amount.
- Proposed refund is positive and at or below INR 1000.
- No previous refund is already recorded for that order.

Refunds above INR 1000 require human approval. The agent has a maximum iteration count, blocks duplicate tool calls with the same arguments, and writes all tool calls, results, final decisions, and guardrail stops to a JSONL audit log.

## Limitations and responsible-use notes

- The dataset and policies are synthetic and should not be treated as real operational, legal, financial, or customer-support policy.
- The Random Forest metrics are illustrative, and the model should not be deployed without real-data validation, drift monitoring, calibration checks, and fairness review.
- The ticket classifier is a controlled mock-LLM demonstration, not a production safety system.
- The refund agent creates only local demonstration ledger records; it does not process payments or contact customers.
- Human approval remains required for high-value refunds and sensitive escalation scenarios.

## Key generated artifacts

| Artifact | Description |
| --- | --- |
| `outputs/data_quality_report.md` | Raw audit, cleaning choices, and validation results |
| `data/clean/instamart.db` | SQLite database containing clean tables |
| `outputs/segmentation.md` | Segment-selection and profile report |
| `outputs/model_metrics.json` | Purchase-model evaluation metrics |
| `state/customer_model.joblib` | Serialized model artifact with metadata |
| `outputs/classification_report.md` | Zero-shot versus few-shot ticket evaluation |
| `outputs/policy_evaluation.md` | Grounded policy-Q&A evaluation |
| `outputs/agent_report.md` | Refund-agent controls and scenario outcomes |
| `outputs/refund_agent_audit.jsonl` | Detailed agent execution audit trail |
