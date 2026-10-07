from functools import lru_cache
from pathlib import Path
import json
import duckdb

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MQL_PATH = DATA_DIR / "olist_marketing_qualified_leads_dataset.csv"
DEALS_PATH = DATA_DIR / "olist_closed_deals_dataset.csv"
ORDERS_PATH = DATA_DIR / "olist_orders_dataset.csv"
ORDER_ITEMS_PATH = DATA_DIR / "olist_order_items_dataset.csv"
SELLERS_PATH = DATA_DIR / "olist_sellers_dataset.csv"


def _csv_view(con, name: str, path: Path):
    if not path.exists():
        return False
    p = path.as_posix().replace("'", "''")
    con.execute(
        f"CREATE VIEW {name} AS SELECT * FROM read_csv_auto("
        f"'{p}', quote='\"', escape='\"', strict_mode=false)"
    )
    return True


@lru_cache(maxsize=1)
def _get_conn():
    con = duckdb.connect(database=":memory:")
    _csv_view(con, "mql", MQL_PATH)
    _csv_view(con, "closed_deals", DEALS_PATH)
    _csv_view(con, "orders_raw", ORDERS_PATH)
    _csv_view(con, "order_items_raw", ORDER_ITEMS_PATH)
    _csv_view(con, "sellers_raw", SELLERS_PATH)
    return con


def _records(df):
    return json.loads(df.to_json(orient="records", date_format="iso"))


def get_marketing_funnel_summary(origin: str | None = None) -> dict:
    con = _get_conn()
    row = con.execute(
        """
        WITH converted AS (SELECT DISTINCT mql_id FROM closed_deals)
        SELECT
            COUNT(*) AS mql_count,
            COUNT(c.mql_id) AS closed_deals,
            ROUND(100.0 * COUNT(c.mql_id) / NULLIF(COUNT(*), 0), 2) AS conversion_rate_pct
        FROM mql m
        LEFT JOIN converted c ON m.mql_id = c.mql_id
        WHERE (? IS NULL OR COALESCE(NULLIF(TRIM(m.origin), ''), 'unknown') = ?)
        """,
        [origin, origin],
    ).fetchone()
    return {
        "origin": origin,
        "mql_count": int(row[0]),
        "closed_deals": int(row[1]),
        "conversion_rate_pct": float(row[2] or 0),
    }


def get_marketing_channel_performance(limit: int = 20) -> dict:
    con = _get_conn()
    limit = max(1, min(int(limit), 50))
    df = con.execute(
        """
        WITH deals AS (
            SELECT DISTINCT mql_id, CAST(won_date AS TIMESTAMP) AS won_date
            FROM closed_deals
        )
        SELECT
            COALESCE(NULLIF(TRIM(m.origin), ''), 'unknown') AS origin,
            COUNT(*) AS mql_count,
            COUNT(d.mql_id) AS closed_deals,
            ROUND(100.0 * COUNT(d.mql_id) / NULLIF(COUNT(*), 0), 2) AS conversion_rate_pct,
            ROUND(AVG(
                CASE WHEN d.mql_id IS NOT NULL
                     THEN date_diff('day', CAST(m.first_contact_date AS DATE), d.won_date)
                END
            ), 2) AS avg_time_to_close_days
        FROM mql m
        LEFT JOIN deals d ON m.mql_id = d.mql_id
        GROUP BY 1
        ORDER BY conversion_rate_pct DESC, closed_deals DESC
        LIMIT ?
        """,
        [limit],
    ).fetchdf()
    return {"results": _records(df)}


def get_seller_360(seller_id: str) -> dict:
    con = _get_conn()
    seller_id = str(seller_id).strip()

    acquisition = con.execute(
        """
        SELECT
            d.seller_id,
            COALESCE(NULLIF(TRIM(m.origin), ''), 'unknown') AS acquisition_origin,
            CAST(m.first_contact_date AS DATE) AS first_contact_date,
            CAST(d.won_date AS TIMESTAMP) AS won_date,
            date_diff('day', CAST(m.first_contact_date AS DATE), CAST(d.won_date AS TIMESTAMP)) AS time_to_close_days,
            d.business_segment, d.lead_type, d.lead_behaviour_profile, d.business_type,
            d.has_company, d.has_gtin, d.average_stock,
            d.declared_product_catalog_size, d.declared_monthly_revenue
        FROM closed_deals d
        LEFT JOIN mql m ON d.mql_id = m.mql_id
        WHERE d.seller_id = ?
        LIMIT 1
        """,
        [seller_id],
    ).fetchdf()

    if acquisition.empty:
        return {"found": False, "seller_id": seller_id, "message": "Không tìm thấy seller trong closed_deals."}

    logistics = con.execute(
        """
        WITH seller_order AS (
            SELECT seller_id, order_id, SUM(price) AS seller_gmv, SUM(freight_value) AS seller_freight
            FROM order_items_raw
            WHERE seller_id = ?
            GROUP BY seller_id, order_id
        ),
        joined AS (
            SELECT
                so.order_id, so.seller_gmv, so.seller_freight,
                CAST(o.order_purchase_timestamp AS TIMESTAMP) AS purchase_ts,
                o.order_status,
                CASE WHEN o.order_status = 'delivered'
                     AND o.order_delivered_customer_date IS NOT NULL
                     AND o.order_estimated_delivery_date IS NOT NULL
                     AND CAST(o.order_delivered_customer_date AS TIMESTAMP) > CAST(o.order_estimated_delivery_date AS TIMESTAMP)
                     THEN 1 ELSE 0 END AS is_late,
                CASE WHEN o.order_status = 'delivered' AND o.order_delivered_customer_date IS NOT NULL
                     THEN date_diff('hour', CAST(o.order_purchase_timestamp AS TIMESTAMP), CAST(o.order_delivered_customer_date AS TIMESTAMP)) / 24.0
                END AS delivery_days
            FROM seller_order so
            JOIN orders_raw o ON so.order_id = o.order_id
        )
        SELECT
            COUNT(DISTINCT order_id) AS n_orders_all,
            COUNT(DISTINCT order_id) FILTER (WHERE order_status = 'delivered') AS n_orders_delivered,
            ROUND(100.0 * AVG(is_late) FILTER (WHERE order_status = 'delivered'), 2) AS late_rate_pct,
            ROUND(AVG(delivery_days) FILTER (WHERE order_status = 'delivered'), 2) AS avg_delivery_days,
            ROUND(SUM(seller_gmv), 2) AS observed_gmv,
            ROUND(SUM(seller_freight), 2) AS observed_freight,
            MIN(purchase_ts) AS first_order_date
        FROM joined
        """,
        [seller_id],
    ).fetchdf()

    activation = con.execute(
        """
        WITH deal AS (
            SELECT CAST(won_date AS TIMESTAMP) AS won_date
            FROM closed_deals
            WHERE seller_id = ?
            LIMIT 1
        ),
        seller_order AS (
            SELECT order_id, SUM(price) AS seller_gmv
            FROM order_items_raw
            WHERE seller_id = ?
            GROUP BY order_id
        ),
        joined AS (
            SELECT CAST(o.order_purchase_timestamp AS TIMESTAMP) AS purchase_ts, so.seller_gmv, d.won_date
            FROM seller_order so
            JOIN orders_raw o ON so.order_id = o.order_id
            CROSS JOIN deal d
        )
        SELECT
            MIN(purchase_ts) FILTER (WHERE purchase_ts >= won_date) AS first_order_after_won_date,
            MIN(date_diff('day', won_date, purchase_ts)) FILTER (WHERE purchase_ts >= won_date) AS days_to_first_order,
            ROUND(SUM(seller_gmv) FILTER (WHERE purchase_ts >= won_date AND purchase_ts < won_date + INTERVAL '30 day'), 2) AS gmv_30d,
            ROUND(SUM(seller_gmv) FILTER (WHERE purchase_ts >= won_date AND purchase_ts < won_date + INTERVAL '90 day'), 2) AS gmv_90d
        FROM joined
        """,
        [seller_id, seller_id],
    ).fetchdf()

    seller_profile = con.execute(
        "SELECT seller_id, seller_city, seller_state FROM sellers_raw WHERE seller_id = ? LIMIT 1",
        [seller_id],
    ).fetchdf()

    return {
        "found": True,
        "seller_id": seller_id,
        "seller_profile": seller_profile.iloc[0].to_dict() if not seller_profile.empty else None,
        "acquisition": acquisition.iloc[0].to_dict(),
        "logistics": logistics.iloc[0].to_dict(),
        "activation": activation.iloc[0].to_dict(),
        "note": "GMV là tổng giá trị item quan sát được trong dataset, không phải kế toán doanh thu thực tế hay contractual LTV.",
    }


def get_acquisition_logistics_performance(limit: int = 20) -> dict:
    con = _get_conn()
    limit = max(1, min(int(limit), 50))
    df = con.execute(
        """
        WITH acquired AS (
            SELECT DISTINCT
                d.seller_id,
                COALESCE(NULLIF(TRIM(m.origin), ''), 'unknown') AS origin,
                CAST(d.won_date AS TIMESTAMP) AS won_date
            FROM closed_deals d
            JOIN mql m ON d.mql_id = m.mql_id
        ),
        seller_order AS (
            SELECT seller_id, order_id, SUM(price) AS seller_gmv
            FROM order_items_raw
            GROUP BY seller_id, order_id
        ),
        perf AS (
            SELECT
                a.origin, a.seller_id, so.order_id, so.seller_gmv,
                CASE WHEN o.order_status = 'delivered'
                     AND o.order_delivered_customer_date IS NOT NULL
                     AND o.order_estimated_delivery_date IS NOT NULL
                     AND CAST(o.order_delivered_customer_date AS TIMESTAMP) > CAST(o.order_estimated_delivery_date AS TIMESTAMP)
                     THEN 1 ELSE 0 END AS is_late
            FROM acquired a
            LEFT JOIN seller_order so ON a.seller_id = so.seller_id
            LEFT JOIN orders_raw o
              ON so.order_id = o.order_id
             AND CAST(o.order_purchase_timestamp AS TIMESTAMP) >= a.won_date
        ),
        grouped AS (
            SELECT
                origin,
                COUNT(DISTINCT seller_id) AS acquired_sellers,
                COUNT(DISTINCT seller_id) FILTER (WHERE order_id IS NOT NULL) AS sellers_with_orders,
                COUNT(DISTINCT order_id) AS post_acquisition_orders,
                ROUND(100.0 * AVG(is_late) FILTER (WHERE order_id IS NOT NULL), 2) AS late_rate_pct,
                ROUND(SUM(seller_gmv), 2) AS post_acquisition_gmv
            FROM perf
            GROUP BY origin
        )
        SELECT
            *,
            ROUND(100.0 * sellers_with_orders / NULLIF(acquired_sellers, 0), 2) AS activation_match_pct
        FROM grouped
        ORDER BY post_acquisition_gmv DESC NULLS LAST
        LIMIT ?
        """,
        [limit],
    ).fetchdf()
    return {"results": _records(df)}
