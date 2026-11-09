-- Instamart Part A: reusable SQLite analysis queries.
-- Run after src/audit_and_clean.py creates data/clean/instamart.db.

-- 1. Monthly completed-order revenue, order count, and average order value.
SELECT
    substr(order_ts, 1, 7) AS order_month,
    COUNT(*) AS completed_orders,
    ROUND(SUM(order_amount), 2) AS revenue_inr,
    ROUND(AVG(order_amount), 2) AS avg_order_value_inr
FROM clean_orders
WHERE status = 'completed'
  AND order_amount IS NOT NULL
GROUP BY substr(order_ts, 1, 7)
ORDER BY order_month;

-- 2. Customer-level completed-order metrics.
SELECT
    c.customer_id,
    c.city,
    COUNT(o.order_id) AS completed_orders,
    ROUND(COALESCE(SUM(o.order_amount), 0), 2) AS revenue_inr,
    ROUND(AVG(o.order_amount), 2) AS avg_order_value_inr,
    ROUND(AVG(o.delivery_minutes), 2) AS avg_delivery_minutes
FROM clean_customers AS c
LEFT JOIN clean_orders AS o
    ON c.customer_id = o.customer_id
   AND o.status = 'completed'
GROUP BY c.customer_id, c.city
ORDER BY revenue_inr DESC, completed_orders DESC;

-- 3. Top ten products by completed-order revenue.
SELECT
    p.product_id,
    p.product_name,
    p.category,
    COUNT(o.order_id) AS completed_orders,
    ROUND(SUM(o.order_amount), 2) AS revenue_inr
FROM clean_orders AS o
JOIN clean_products AS p ON o.product_id = p.product_id
WHERE o.status = 'completed'
  AND o.order_amount IS NOT NULL
GROUP BY p.product_id, p.product_name, p.category
ORDER BY revenue_inr DESC
LIMIT 10;

-- 4. Completed-order performance by product category.
SELECT
    p.category,
    COUNT(o.order_id) AS completed_orders,
    ROUND(SUM(o.order_amount), 2) AS revenue_inr,
    ROUND(AVG(o.order_amount), 2) AS avg_order_value_inr,
    ROUND(AVG(o.delivery_minutes), 2) AS avg_delivery_minutes
FROM clean_orders AS o
JOIN clean_products AS p ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY p.category
ORDER BY revenue_inr DESC;

-- 5. Delivery performance by city, including late deliveries over 45 minutes.
SELECT
    c.city,
    COUNT(o.order_id) AS completed_orders_with_delivery_time,
    ROUND(AVG(o.delivery_minutes), 2) AS avg_delivery_minutes,
    SUM(CASE WHEN o.delivery_minutes > 45 THEN 1 ELSE 0 END) AS late_orders,
    ROUND(100.0 * SUM(CASE WHEN o.delivery_minutes > 45 THEN 1 ELSE 0 END) / COUNT(o.order_id), 2) AS late_order_pct
FROM clean_orders AS o
JOIN clean_customers AS c ON o.customer_id = c.customer_id
WHERE o.status = 'completed'
  AND o.delivery_minutes IS NOT NULL
GROUP BY c.city
ORDER BY late_order_pct DESC, avg_delivery_minutes DESC;

-- 6. Discount-band effectiveness on completed orders.
SELECT
    CASE
        WHEN discount_pct = 0 THEN '0%'
        WHEN discount_pct <= 10 THEN '1-10%'
        WHEN discount_pct <= 25 THEN '11-25%'
        ELSE '26%+'
    END AS discount_band,
    COUNT(*) AS completed_orders,
    ROUND(SUM(order_amount), 2) AS revenue_inr,
    ROUND(AVG(order_amount), 2) AS avg_order_value_inr
FROM clean_orders
WHERE status = 'completed'
  AND order_amount IS NOT NULL
GROUP BY discount_band
ORDER BY completed_orders DESC;

-- 7. Support-ticket counts and empty-ticket rate by true category.
SELECT
    COALESCE(true_category, 'missing') AS true_category,
    COUNT(*) AS ticket_count,
    SUM(CASE WHEN text IS NULL THEN 1 ELSE 0 END) AS empty_text_tickets,
    ROUND(100.0 * SUM(CASE WHEN text IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) AS empty_text_pct
FROM clean_tickets
GROUP BY COALESCE(true_category, 'missing')
ORDER BY ticket_count DESC;

-- 8. Customers with completed orders but no support tickets.
SELECT
    c.customer_id,
    c.city,
    COUNT(o.order_id) AS completed_orders,
    ROUND(SUM(o.order_amount), 2) AS revenue_inr
FROM clean_customers AS c
JOIN clean_orders AS o
    ON c.customer_id = o.customer_id
   AND o.status = 'completed'
LEFT JOIN clean_tickets AS t ON c.customer_id = t.customer_id
WHERE t.ticket_id IS NULL
GROUP BY c.customer_id, c.city
ORDER BY revenue_inr DESC;
