"""Safe, read-only agent tools for DataCo Supply Chain and optional access logs.

The public agent table deliberately omits direct personal fields. Access-log IP and URL
columns are never materialized into the queryable table or returned to the model.
"""
from __future__ import annotations

import json
import re
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import duckdb

BASE_DIR = Path(__file__).resolve().parent.parent
DATACO_CSV = BASE_DIR / "data_2" / "DataCoSupplyChainDataset.csv"
ACCESS_LOG_CSV = BASE_DIR / "data_2" / "tokenized_access_logs.csv"
MAX_ROWS = 200

_EXCLUDED_COLUMNS = {
    "customer_email", "customer_fname", "customer_lname", "customer_password",
    "customer_street", "customer_zipcode", "product_image",
}
_FORBIDDEN_SQL = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|copy|export|import|"
    r"install|load|pragma|call|execute|read_csv|read_parquet|read_json|read_text|"
    r"sqlite_scan|postgres_scan|httpfs|glob|pg_read_file|read_blob)\b",
    re.IGNORECASE,
)
_COLUMN_INFO: list[tuple[str, str, str]] = []


def _quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _safe_name(value: str) -> str:
    name = re.sub(r"[^0-9a-zA-Z]+", "_", str(value).strip()).strip("_").lower()
    if not name:
        name = "column"
    if name[0].isdigit():
        name = "col_" + name
    return name


def _prepare_dataco_csv(path: Path) -> Path:
    """Transcode the mixed ISO-8859-1/Windows-1252 source bytes into cached UTF-8.

    The source has a few Windows-1252 punctuation bytes in the C1 range, which DuckDB's
    latin-1 reader rejects. The original CSV is not modified.
    """
    stat = path.stat()
    cache_dir = Path(tempfile.gettempdir()) / "logistic_agent_dataco_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{path.stem}-{stat.st_size}-{stat.st_mtime_ns}.utf8.csv"
    if cached.is_file() and cached.stat().st_size > 0:
        return cached
    with path.open("r", encoding="cp1252", errors="replace", newline="") as source, \
         cached.open("w", encoding="utf-8", newline="") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)
    return cached


@lru_cache(maxsize=1)
def _get_conn():
    """Load only safe DataCo columns into a private DuckDB connection."""
    if not DATACO_CSV.is_file():
        raise FileNotFoundError(f"Không tìm thấy {DATACO_CSV}. Hãy đặt CSV chính vào data_2/.")
    con = duckdb.connect(database=":memory:")
    csv_path = _prepare_dataco_csv(DATACO_CSV)
    path = str(csv_path).replace("'", "''")
    source = f"read_csv_auto('{path}', header=true, sample_size=20000)"
    schema = con.execute(f"DESCRIBE SELECT * FROM {source}").fetchall()
    aliases, used = [], set()
    for row in schema:
        original = str(row[0])
        alias = _safe_name(original)
        if alias in used:
            suffix = 2
            while f"{alias}_{suffix}" in used:
                suffix += 1
            alias = f"{alias}_{suffix}"
        used.add(alias)
        if alias in _EXCLUDED_COLUMNS:
            continue
        aliases.append((original, alias, str(row[1])))
    select_sql = ", ".join(
        f"{_quote_ident(original)} AS {_quote_ident(alias)}" for original, alias, _ in aliases
    )
    con.execute(f"CREATE TABLE dataco_orders AS SELECT {select_sql} FROM {source}")
    global _COLUMN_INFO
    _COLUMN_INFO = aliases
    return con


def describe_dataset() -> dict[str, Any]:
    con = _get_conn()
    count = int(con.execute("SELECT COUNT(*) FROM dataco_orders").fetchone()[0])
    return {
        "dataset": "DataCo Supply Chain",
        "table": "dataco_orders",
        "n_rows": count,
        "n_orders": int(con.execute("SELECT COUNT(DISTINCT order_id) FROM dataco_orders").fetchone()[0]),
        "columns": [{"name": alias, "source_name": source, "type": dtype} for source, alias, dtype in _COLUMN_INFO],
        "notes": [
            "Mỗi dòng là một order item; đếm đơn bằng COUNT(DISTINCT order_id).",
            "late_delivery_risk là nhãn có sẵn trong dữ liệu, không phải xác suất dự đoán ML.",
            "Customer Country là quốc gia khách hàng mua hàng; Order Country là quốc gia đích theo mô tả dataset.",
            "Các cột tên, email, mật khẩu, đường phố, mã bưu chính và URL ảnh không được đưa vào bảng agent.",
        ],
    }


def query_dataset_sql(sql: str) -> dict[str, Any]:
    """Run one read-only query on the sanitized dataco_orders table."""
    if not isinstance(sql, str) or not sql.strip():
        return {"error": "Cần cung cấp một câu SQL SELECT."}
    cleaned = sql.strip()
    if ";" in cleaned or "--" in cleaned or "/*" in cleaned or "*/" in cleaned:
        return {"error": "SQL chỉ được có một câu lệnh và không được chứa comment."}
    if not re.match(r"^(select|with)\b", cleaned, re.IGNORECASE):
        return {"error": "Chỉ cho phép SELECT hoặc WITH...SELECT."}
    if _FORBIDDEN_SQL.search(cleaned):
        return {"error": "SQL chứa từ khóa/hàm không được phép."}
    if not re.search(r"\bdataco_orders\b", cleaned, re.IGNORECASE):
        return {"error": "Truy vấn phải sử dụng bảng dataco_orders."}
    try:
        con = _get_conn()
        df = con.execute(f"SELECT * FROM ({cleaned}) AS agent_query LIMIT {MAX_ROWS}").fetchdf()
        return {
            "n_rows_returned": len(df),
            "columns": list(df.columns),
            "rows": json.loads(df.to_json(orient="records", date_format="iso")),
            "limit": MAX_ROWS,
        }
    except Exception as exc:
        return {"error": f"Lỗi SQL DataCo: {exc}"}


def get_order_info(order_id: str) -> dict[str, Any]:
    """Look up a DataCo order without personal contact/address fields."""
    try:
        con = _get_conn()
        df = con.execute(
            """SELECT order_id, order_status, delivery_status, customer_country, order_country,
                      order_region, market, shipping_mode, days_for_shipping_real,
                      days_for_shipment_scheduled, late_delivery_risk, sales, order_item_total,
                      order_item_quantity, category_name, product_name
               FROM dataco_orders WHERE CAST(order_id AS VARCHAR) = ? LIMIT 20""",
            [str(order_id).strip()],
        ).fetchdf()
        if df.empty:
            return {"found": False, "order_id": str(order_id)}
        return {"found": True, "order_id": str(order_id), "n_line_items_returned": len(df),
                "items": json.loads(df.to_json(orient="records", date_format="iso"))}
    except Exception as exc:
        return {"found": False, "error": str(exc)}


def top_risky_regions(n: int = 5) -> dict[str, Any]:
    """Rank regions by their pre-existing dataset late-risk label."""
    try:
        limit = max(1, min(int(n), 20))
        con = _get_conn()
        df = con.execute(
            """SELECT order_region, COUNT(DISTINCT order_id) AS n_orders,
                      ROUND(100.0 * AVG(TRY_CAST(late_delivery_risk AS DOUBLE)), 2) AS late_risk_pct,
                      ROUND(AVG(TRY_CAST(days_for_shipping_real AS DOUBLE)), 2) AS avg_actual_shipping_days,
                      ROUND(AVG(TRY_CAST(days_for_shipment_scheduled AS DOUBLE)), 2) AS avg_scheduled_days
               FROM dataco_orders WHERE order_region IS NOT NULL AND TRIM(CAST(order_region AS VARCHAR)) <> ''
               GROUP BY order_region ORDER BY late_risk_pct DESC NULLS LAST LIMIT ?""",
            [limit],
        ).fetchdf()
        return {"rows": json.loads(df.to_json(orient="records")), "metric": "existing Late_delivery_risk label"}
    except Exception as exc:
        return {"error": str(exc)}


def get_country_delivery_stats(country: str = "", n: int = 10) -> dict[str, Any]:
    try:
        limit = max(1, min(int(n), 30))
        con = _get_conn()
        where = ""
        params: list[Any] = []
        if country and country.strip():
            where = "WHERE order_country ILIKE ?"
            params.append("%" + country.strip() + "%")
        params.append(limit)
        df = con.execute(
            f"""SELECT order_country, COUNT(DISTINCT order_id) AS n_orders,
                       ROUND(AVG(TRY_CAST(days_for_shipping_real AS DOUBLE)), 2) AS avg_actual_days,
                       ROUND(AVG(TRY_CAST(days_for_shipment_scheduled AS DOUBLE)), 2) AS avg_scheduled_days,
                       ROUND(100.0 * AVG(TRY_CAST(late_delivery_risk AS DOUBLE)), 2) AS late_risk_pct,
                       ROUND(SUM(TRY_CAST(sales AS DOUBLE)), 2) AS total_sales
                FROM dataco_orders {where}
                GROUP BY order_country ORDER BY n_orders DESC LIMIT ?""",
            params,
        ).fetchdf()
        return {"rows": json.loads(df.to_json(orient="records")),
                "metric_note": "late_risk_pct is the existing dataset label average, not a model prediction"}
    except Exception as exc:
        return {"error": str(exc)}


def compare_shipping_modes(market: str = "") -> dict[str, Any]:
    try:
        con = _get_conn()
        where = "WHERE market = ?" if market.strip() else ""
        params = [market.strip()] if where else []
        df = con.execute(
            f"""SELECT shipping_mode, COUNT(DISTINCT order_id) AS n_orders,
                       ROUND(AVG(TRY_CAST(days_for_shipping_real AS DOUBLE)), 2) AS avg_actual_days,
                       ROUND(AVG(TRY_CAST(days_for_shipment_scheduled AS DOUBLE)), 2) AS avg_scheduled_days,
                       ROUND(100.0 * AVG(TRY_CAST(late_delivery_risk AS DOUBLE)), 2) AS late_risk_pct
                FROM dataco_orders {where}
                GROUP BY shipping_mode ORDER BY n_orders DESC""",
            params,
        ).fetchdf()
        return {"rows": json.loads(df.to_json(orient="records")), "market": market or "All markets",
                "metric_note": "late_risk_pct is the existing dataset label average"}
    except Exception as exc:
        return {"error": str(exc)}


def _get_access_conn():
    """Build an aggregate-only access log table, omitting IP and URL at ingestion."""
    if not ACCESS_LOG_CSV.is_file():
        raise FileNotFoundError("Không có data_2/tokenized_access_logs.csv; công cụ access logs là tùy chọn.")
    con = _get_conn()
    exists = con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name='dataco_access_logs'"
    ).fetchone()[0]
    if not exists:
        path = str(ACCESS_LOG_CSV).replace("'", "''")
        # The access log is UTF-8. Only the safe six columns are selected below.
        source = f"read_csv_auto('{path}', header=true, sample_size=20000)"
        schema = [row[0] for row in con.execute(f"DESCRIBE SELECT * FROM {source}").fetchall()]
        by_norm = {_safe_name(c): c for c in schema}
        required = ["product", "category", "date", "month", "hour", "department"]
        missing = [c for c in required if c not in by_norm]
        if missing:
            raise ValueError("Access log thiếu cột cần thiết: " + ", ".join(missing))
        selected = [f"{_quote_ident(by_norm[safe])} AS {_quote_ident(safe)}" for safe in required]
        con.execute(f"CREATE TABLE dataco_access_logs AS SELECT {', '.join(selected)} FROM {source}")
    return con


def top_accessed_products(n: int = 10) -> dict[str, Any]:
    try:
        limit = max(1, min(int(n), 30))
        con = _get_access_conn()
        df = con.execute(
            """SELECT product, category, COUNT(*) AS visits
               FROM dataco_access_logs GROUP BY product, category ORDER BY visits DESC LIMIT ?""",
            [limit],
        ).fetchdf()
        return {"rows": json.loads(df.to_json(orient="records")),
                "source": "aggregate access events; IP and URL omitted"}
    except Exception as exc:
        return {"error": str(exc)}


def access_traffic_by_hour() -> dict[str, Any]:
    try:
        con = _get_access_conn()
        df = con.execute(
            """SELECT TRY_CAST(hour AS INTEGER) AS hour, COUNT(*) AS visits
               FROM dataco_access_logs GROUP BY TRY_CAST(hour AS INTEGER) ORDER BY hour"""
        ).fetchdf()
        return {"rows": json.loads(df.to_json(orient="records")),
                "source": "aggregate access events; IP and URL omitted"}
    except Exception as exc:
        return {"error": str(exc)}


def show_country_flow_map(origin_country: str = "", destination_country: str = "", metric: str = "orders") -> dict[str, Any]:
    """UI action: open the DataCo country-flow map, optionally focused on a country pair."""
    metric = metric if metric in {"orders", "late_risk_pct", "avg_shipping_days", "sales"} else "orders"
    return {
        "open_map": True, "dataset": "dataco",
        "origin_country": origin_country.strip(), "destination_country": destination_country.strip(),
        "metric": metric,
        "message": "Mở bản đồ luồng DataCo. Customer Country → Order Country là proxy theo trường dữ liệu, không xác nhận kho hoặc seller xuất phát.",
    }


SYSTEM_PROMPT = """Bạn là AI Logistics Agent chuyên phân tích bộ DataCo Supply Chain (thị trường quốc tế).
Chỉ dùng công cụ DataCo hoặc bảng dataco_orders; không trộn dữ liệu Olist.
Hãy gọi describe_dataset nếu chưa rõ schema. Đếm đơn bằng COUNT(DISTINCT order_id) vì mỗi đơn có nhiều dòng hàng.
Công cụ gồm: describe_dataset, query_dataset_sql, get_order_info, top_risky_regions,
get_country_delivery_stats, compare_shipping_modes, top_accessed_products, access_traffic_by_hour,
show_country_flow_map.
Access-log tools chỉ trả thống kê tổng hợp; không yêu cầu hoặc tiết lộ IP, URL, email, tên, mật khẩu, địa chỉ đường phố hoặc mã bưu chính.
Late_delivery_risk là nhãn đã có trong dataset, không phải dự đoán ML. Không dùng model Olist cho DataCo.
Country flow map vẽ Customer Country → Order Country như proxy theo metadata hiện có; không khẳng định đó là tuyến kho/seller đã xác minh.
Trả lời bằng tiếng Việt, nêu phép tính ngắn gọn, không bịa dữ liệu. Nếu access log chưa có file cục bộ thì báo công cụ đó không khả dụng.
"""

TOOLS = [
    {"name": "describe_dataset", "description": "Xem schema an toàn, số dòng và số đơn duy nhất của DataCo.", "input_schema": {"type": "object", "properties": {}}},
    {"name": "query_dataset_sql", "description": "Truy vấn SQL SELECT/WITH chỉ đọc trên bảng dataco_orders; tối đa 200 dòng.", "input_schema": {"type": "object", "properties": {"sql": {"type": "string", "description": "Một câu SELECT/WITH trên dataco_orders"}}, "required": ["sql"]}},
    {"name": "get_order_info", "description": "Tra cứu các dòng hàng của một Order Id DataCo; không trả thông tin liên hệ/địa chỉ riêng tư.", "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}},
    {"name": "top_risky_regions", "description": "Xếp hạng Order Region theo trung bình nhãn Late_delivery_risk có sẵn.", "input_schema": {"type": "object", "properties": {"n": {"type": "integer", "default": 5}}}},
    {"name": "get_country_delivery_stats", "description": "Thống kê số đơn, số ngày vận chuyển và nhãn rủi ro theo quốc gia đích; có thể truyền country và n.", "input_schema": {"type": "object", "properties": {"country": {"type": "string"}, "n": {"type": "integer", "default": 10}}}},
    {"name": "compare_shipping_modes", "description": "So sánh số ngày vận chuyển và nhãn rủi ro theo Shipping Mode; market là tùy chọn.", "input_schema": {"type": "object", "properties": {"market": {"type": "string"}}}},
    {"name": "top_accessed_products", "description": "Nếu tokenized_access_logs.csv có sẵn cục bộ, trả top sản phẩm/danh mục theo lượt truy cập; không trả IP hoặc URL.", "input_schema": {"type": "object", "properties": {"n": {"type": "integer", "default": 10}}}},
    {"name": "access_traffic_by_hour", "description": "Nếu access log có sẵn cục bộ, thống kê lượt truy cập theo giờ; không trả IP hoặc URL.", "input_schema": {"type": "object", "properties": {}}},
    {"name": "show_country_flow_map", "description": "Mở bản đồ luồng quốc gia DataCo, có thể focus vào origin_country/destination_country. metric: orders, late_risk_pct, avg_shipping_days, sales.", "input_schema": {"type": "object", "properties": {"origin_country": {"type": "string"}, "destination_country": {"type": "string"}, "metric": {"type": "string", "enum": ["orders", "late_risk_pct", "avg_shipping_days", "sales"]}}}},
]
