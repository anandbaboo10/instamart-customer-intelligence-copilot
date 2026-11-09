# Instamart Data Quality Report

## Raw-data audit
- Raw rows: {'customers': 1200, 'products': 40, 'orders': 13084, 'tickets': 164}
- Exact duplicate rows: {'customers': 0, 'products': 0, 'orders': 119, 'tickets': 0}
- Invalid values: {'customer_age_outside_13_to_100': 6, 'order_amount_unparseable': 67, 'order_amount_not_positive': 0, 'delivery_minutes_negative': 106, 'completed_orders_with_delivery_outside_8_to_75': 106}

## Cleaning decisions
- City values were trimmed and standardized to title case.
- Gender variants such as M/m/MALE and F/f/FEMALE were standardized to male/female.
- Ages outside 13–100 were set to missing rather than imputed.
- Unparseable or non-positive order amounts were set to missing rather than invented.
- Delivery minutes outside 8–75 for completed orders, including negative values, were set to missing.
- Exact duplicate order rows were removed; for repeated order IDs, the latest timestamped version was retained.
- Exact duplicate order rows removed: 119
- Earlier order versions removed: 656

## Validation
- Passed: True
- Clean row counts: {'customers': 1200, 'products': 40, 'orders': 12309, 'tickets': 164}
- Duplicate primary keys: {'customers': 0, 'products': 0, 'orders': 0, 'tickets': 0}
- Invalid non-null delivery values: 0
- Foreign-key violations: orders→customers=0, orders→products=0, tickets→customers=0, tickets→orders=0

## Outputs
- Clean CSVs: data/clean/
- SQLite database: data/clean/instamart.db
- Machine-readable audit: outputs/audit_summary.json
