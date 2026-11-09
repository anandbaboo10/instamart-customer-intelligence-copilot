# Ticket Classification Evaluation

## Evaluation design
- Ground truth is the synthetic `true_category` field.
- Tickets whose synthetic label is `none` are excluded from accuracy because `none` is not an allowed production class.
- Both strategies are run from a reset mock state so the simulated transient errors are comparable.

## Results
| Strategy | Evaluated tickets | Accuracy | Fallback count | Fallback rate |
| --- | ---: | ---: | ---: | ---: |
| Zero-shot | 160 | 86.88% | 21 | 12.80% |
| Few-shot | 160 | 100.00% | 21 | 12.80% |

Few-shot accuracy change versus zero-shot: +13.12%.

## Reliability and safety controls
- Email addresses and phone numbers are masked before inference.
- Inputs are limited to 180 words after masking; long tickets are explicitly marked as truncated.
- Each response must parse as JSON and contain an allowed category and a non-empty reason.
- Transient errors receive up to 3 attempts before fallback.
- Invalid JSON, invalid category values, and exhausted retries use a deterministic keyword fallback and are logged in the result file.
