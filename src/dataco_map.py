"""Country flow analytics for the DataCo Supply Chain dataset."""
from __future__ import annotations

from typing import Any

from src import dataco_tools
from src.dataco_country_coords import COUNTRY_COORDS


_DATE_EXPR = (
    "COALESCE(TRY_CAST(order_date_dateorders AS TIMESTAMP), "
    "TRY_STRPTIME(CAST(order_date_dateorders AS VARCHAR), '%m/%d/%Y %H:%M'), "
    "TRY_STRPTIME(CAST(order_date_dateorders AS VARCHAR), '%m/%d/%Y %H:%M:%S'))"
)
_ALLOWED_METRICS = {"orders", "late_risk_pct", "avg_shipping_days", "sales"}


def _filters(start_date=None, end_date=None, market=None, shipping_mode=None):
    parts, params = [], []
    if start_date:
        parts.append(f"CAST({_DATE_EXPR} AS DATE) >= TRY_CAST(? AS DATE)")
        params.append(str(start_date))
    if end_date:
        parts.append(f"CAST({_DATE_EXPR} AS DATE) <= TRY_CAST(? AS DATE)")
        params.append(str(end_date))
    if market:
        parts.append("market = ?")
        params.append(str(market))
    if shipping_mode:
        parts.append("shipping_mode = ?")
        params.append(str(shipping_mode))
    return (" WHERE " + " AND ".join(parts) if parts else ""), params


def _clean_coord(name: str):
    value = (name or "").strip()
    if value in COUNTRY_COORDS:
        lat, lng = COUNTRY_COORDS[value]
        return {"country": value, "lat": lat, "lng": lng}
    return None


def get_map_filters() -> dict[str, Any]:
    con = dataco_tools._get_conn()
    date_row = con.execute(f"SELECT MIN({_DATE_EXPR}), MAX({_DATE_EXPR}) FROM dataco_orders").fetchone()
    markets = [r[0] for r in con.execute(
        "SELECT DISTINCT market FROM dataco_orders WHERE market IS NOT NULL ORDER BY market"
    ).fetchall()]
    modes = [r[0] for r in con.execute(
        "SELECT DISTINCT shipping_mode FROM dataco_orders WHERE shipping_mode IS NOT NULL ORDER BY shipping_mode"
    ).fetchall()]
    origins = [r[0] for r in con.execute(
        "SELECT DISTINCT customer_country FROM dataco_orders WHERE customer_country IS NOT NULL ORDER BY customer_country"
    ).fetchall()]
    destinations = [r[0] for r in con.execute(
        "SELECT DISTINCT order_country FROM dataco_orders WHERE order_country IS NOT NULL ORDER BY order_country"
    ).fetchall()]
    return {
        "start_date": date_row[0].strftime("%Y-%m-%d") if date_row and date_row[0] else None,
        "end_date": date_row[1].strftime("%Y-%m-%d") if date_row and date_row[1] else None,
        "markets": markets,
        "shipping_modes": modes,
        "origin_countries": origins,
        "destination_countries": destinations,
        "mapped_origin_countries": sum(1 for x in origins if _clean_coord(x)),
        "mapped_destination_countries": sum(1 for x in destinations if _clean_coord(x)),
        "map_note": "Customer Country → Order Country is a proxy from the available dataset fields, not a verified warehouse/seller route.",
    }


def get_map_summary(start_date=None, end_date=None, market=None, shipping_mode=None) -> dict[str, Any]:
    con = dataco_tools._get_conn()
    where, params = _filters(start_date, end_date, market, shipping_mode)
    row = con.execute(
        f"""SELECT COUNT(*) AS line_items,
                   COUNT(DISTINCT order_id) AS orders,
                   ROUND(AVG(TRY_CAST(days_for_shipping_real AS DOUBLE)), 2) AS avg_actual_days,
                   ROUND(AVG(TRY_CAST(days_for_shipment_scheduled AS DOUBLE)), 2) AS avg_scheduled_days,
                   ROUND(100.0 * AVG(TRY_CAST(late_delivery_risk AS DOUBLE)), 2) AS late_risk_pct,
                   ROUND(SUM(TRY_CAST(sales AS DOUBLE)), 2) AS sales
            FROM dataco_orders{where}""",
        params,
    ).fetchone()
    return {
        "line_items": int(row[0] or 0), "orders": int(row[1] or 0),
        "avg_actual_days": row[2], "avg_scheduled_days": row[3],
        "late_risk_pct": row[4], "sales": row[5],
        "metric_note": "Late-risk is the average of the dataset's existing label, not an ML prediction.",
    }


def get_country_flows(
    start_date=None, end_date=None, market=None, shipping_mode=None,
    origin_country=None, destination_country=None, metric="orders", limit=40, min_orders=20,
) -> dict[str, Any]:
    metric = metric if metric in _ALLOWED_METRICS else "orders"
    try:
        limit = max(1, min(int(limit), 250))
        min_orders = max(1, min(int(min_orders), 1000000))
        con = dataco_tools._get_conn()
        where, params = _filters(start_date, end_date, market, shipping_mode)
        extra = []
        if origin_country:
            extra.append("customer_country = ?")
            params.append(str(origin_country))
        if destination_country:
            extra.append("order_country = ?")
            params.append(str(destination_country))
        if extra:
            where += (" AND " if where else " WHERE ") + " AND ".join(extra)
        metric_expr = {
            "orders": "orders",
            "late_risk_pct": "late_risk_pct",
            "avg_shipping_days": "avg_actual_days",
            "sales": "sales",
        }[metric]
        rows = con.execute(
            f"""SELECT TRIM(CAST(customer_country AS VARCHAR)) AS origin_country,
                       TRIM(CAST(order_country AS VARCHAR)) AS destination_country,
                       COUNT(DISTINCT order_id) AS orders,
                       COUNT(*) AS line_items,
                       ROUND(AVG(TRY_CAST(days_for_shipping_real AS DOUBLE)), 2) AS avg_actual_days,
                       ROUND(AVG(TRY_CAST(days_for_shipment_scheduled AS DOUBLE)), 2) AS avg_scheduled_days,
                       ROUND(100.0 * AVG(TRY_CAST(late_delivery_risk AS DOUBLE)), 2) AS late_risk_pct,
                       ROUND(SUM(TRY_CAST(sales AS DOUBLE)), 2) AS sales
                FROM dataco_orders{where}
                GROUP BY 1, 2
                HAVING COUNT(DISTINCT order_id) >= ?
                ORDER BY {metric_expr} DESC NULLS LAST
                LIMIT ?""",
            [*params, min_orders, limit],
        ).fetchall()
        flows = []
        for row in rows:
            origin, destination = row[0], row[1]
            a, b = _clean_coord(origin), _clean_coord(destination)
            if not a or not b or not origin or not destination or origin == destination:
                continue
            flows.append({
                "id": f"{origin}::{destination}",
                "origin_country": origin, "destination_country": destination,
                "origin": a, "destination": b,
                "orders": int(row[2]), "line_items": int(row[3]),
                "avg_actual_days": row[4], "avg_scheduled_days": row[5],
                "late_risk_pct": row[6], "sales": row[7],
            })
        return {"metric": metric, "flows": flows, "n_flows": len(flows),
                "proxy_note": "Customer Country → Order Country is a proxy, not a verified physical route.",
                "metric_note": "late_risk_pct uses the pre-existing dataset label; no DataCo-specific ML model is applied."}
    except Exception as exc:
        return {"error": f"Không thể tổng hợp country flows: {exc}"}


def get_country_flow_orders(
    origin_country: str, destination_country: str, start_date=None, end_date=None,
    market=None, shipping_mode=None, limit=25,
) -> dict[str, Any]:
    try:
        limit = max(1, min(int(limit), 100))
        con = dataco_tools._get_conn()
        where, params = _filters(start_date, end_date, market, shipping_mode)
        extra = ["customer_country = ?", "order_country = ?"]
        params.extend([origin_country, destination_country])
        where += (" AND " if where else " WHERE ") + " AND ".join(extra)
        rows = con.execute(
            f"""SELECT order_id, order_status, delivery_status, market, shipping_mode,
                       order_region, days_for_shipping_real, days_for_shipment_scheduled,
                       late_delivery_risk, ROUND(TRY_CAST(sales AS DOUBLE), 2) AS sales,
                       order_item_quantity, category_name
                FROM dataco_orders{where}
                ORDER BY {_DATE_EXPR} DESC NULLS LAST LIMIT ?""",
            [*params, limit],
        ).fetchdf()
        return {
            "origin_country": origin_country, "destination_country": destination_country,
            "rows": rows.where(rows.notna(), None).to_dict(orient="records"),
            "n_rows": len(rows),
            "note": "Each row is an order item; an order may appear more than once.",
        }
    except Exception as exc:
        return {"error": f"Không thể lấy mẫu đơn: {exc}"}
