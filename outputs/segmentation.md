# Customer Segmentation

## Method
- Algorithm: KMeans after StandardScaler normalization.
- Candidate segment counts: 3, 4, 5, and 6.
- Selection rule: highest silhouette score; inertia is retained for comparison.
- Selected segment count: 4.

## Candidate metrics
| k | inertia | silhouette_score |
| --- | --- | --- |
| 3 | 6787.490146920961 | 0.3544675894782058 |
| 4 | 5495.822000306771 | 0.372153242422699 |
| 5 | 4890.057700415416 | 0.331190796627138 |
| 6 | 4436.2197682446085 | 0.32623692605323656 |

## Segment profile
| segment | customers | avg_historical_orders | avg_historical_revenue_inr | avg_recency_days | future_purchase_rate |
| --- | --- | --- | --- | --- | --- |
| 1 | 523 | 2.4 | 531.43 | 39.11 | 0.64 |
| 2 | 173 | 4.31 | 4667.65 | 26.23 | 0.83 |
| 3 | 413 | 12.8 | 2283.05 | 9.0 | 1.0 |
| 4 | 91 | 0.18 | 47.98 | 164.73 | 0.64 |

## Point-in-time design
- Features use completed orders strictly before 2025-08-01.
- The outcome is measured only in the following 60 days, preventing future-order leakage into features.
