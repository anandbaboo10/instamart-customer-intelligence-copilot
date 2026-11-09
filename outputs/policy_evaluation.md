# Grounded Policy QA Evaluation

## Retrieval design
- Source documents: generated `.txt` files in `data/policy`.
- Chunking: each policy sentence is a traceable chunk with its source filename.
- Retrieval: TF-IDF cosine similarity; only positive-score chunks are passed to the answerer.
- Context limit: top 3 chunks.
- Generation guardrail: the assistant is instructed to use only retrieved context and to refuse when no context is retrieved.

## Evaluation results
- Test cases: 11
- Source retrieval accuracy: 100.00%
- Grounded-answer rate: 100.00%
- Unsupported answers: 0

## Test-case traceability
| Question | Expected source | Retrieved source(s) | Source match | Grounded answer |
| --- | --- | --- | ---: | ---: |
| Which orders are eligible for a refund? | refund_policy.txt | refund_policy.txt; refund_policy.txt; delivery_sla.txt | True | True |
| What refund is available when delivery takes more than 45 minutes? | refund_policy.txt | refund_policy.txt; delivery_sla.txt; delivery_sla.txt | True | True |
| What is the maximum automatic refund amount? | refund_policy.txt | refund_policy.txt; refund_policy.txt; refund_policy.txt | True | True |
| Can a cancelled order receive a refund? | refund_policy.txt | refund_policy.txt; refund_policy.txt; delivery_sla.txt | True | True |
| What delivery time does the delivery SLA target? | delivery_sla.txt | delivery_sla.txt; delivery_sla.txt; refund_policy.txt | True | True |
| When is a delivery considered delayed? | delivery_sla.txt | delivery_sla.txt; delivery_sla.txt; refund_policy.txt | True | True |
| What customer information must support agents mask? | privacy.txt | privacy.txt; privacy.txt; escalation.txt | True | True |
| Can an agent repeat a customer's full card number in a reply? | privacy.txt | privacy.txt; privacy.txt | True | True |
| Which refund cases require escalation to a human lead? | escalation.txt | escalation.txt; refund_policy.txt; refund_policy.txt | True | True |
| What should happen when a customer reports a fraud or account takeover claim? | escalation.txt | escalation.txt; privacy.txt; privacy.txt | True | True |
| How can an employee change a parking pass? | none | none | True | True |
