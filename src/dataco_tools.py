"""Read-only analytics adapter for the DataCo Supply Chain dataset in ./data_2.

The CSV is supplied by the repository owner; this module never downloads or copies data.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import duckdb

BASE_DIR = Path(__file__).resolve().parent.parent
DATACO_CSV = BASE_DIR / "data_2" / "DataCoSupplyChainDataset.csv"
MAX_ROWS = 200

def _quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'

def _safe_name(value: str) -> str:
    name = re.sub(r"[^0-9a-zA-Z]+", "_", str(value).strip()).strip("_").lower()
    if not name:
        name = "column"
    if name[0].isdigit():
        name = "col_" + name
    return name

@lru_cache(maxsize=1)
def _get_conn():
    if not DATACO_CSV.is_file():
        raise FileNotFoundError(
            f"Chưa tìm thấy {DATACO_CSV}. Hãy push DataCoSupplyChainDataset.csv vào thư mục data_2/."
        )
    con = duckdb.connect(database=":memory:")
    path = str(DATACO_CSV).replace("'", "''")
    source = f"read_csv_auto('{path}', header=true, sample_size=10000, encoding='latin-1')"
    # Read the detected header from DuckDB, then expose stable snake_case aliases.
    schema = con.execute(f"DESCRIBE SELECT * FROM {source}").fetchall()
    aliases = []
    used = set()
    for row in schema:
        original = row[0]
        alias = _safe_name(original)
        if alias in used:
            suffix = 2
            while f"{alias}_{suffix}" in used:
                suffix += 1
            alias = f"{alias}_{suffix}"
        used.add(alias)
        aliases.append((original, alias, row[1]))
    select_sql = ", ".join(
        f"{_quote_ident(original)} AS {_quote_ident(alias)}" for original, alias, _ in aliases
    )
    con.execute(f"CREATE TABLE dataco_orders AS SELECT {select_sql} FROM {source}")
    con.execute("CREATE INDEX IF NOT EXISTS idx_dataco_order_id ON dataco_orders(order_id)")
    con._dataco_columns = aliases
    return con

def describe_dataset() -> dict:
    """Describe the DataCo table and list canonical column names and types."""
    con = _get_conn()
    count = con.execute("SELECT COUNT(*) FROM dataco_orders").fetchone()[0]
    columns = [
        {"name": row[1], "source_name": row[0], "type": row[2]}
        for row in con._dataco_columns
    ]
    return {
        "dataset": "DataCo Supply Chain",
        "table": "dataco_orders",
        "n_rows": int(count),
        "columns": columns,
        "notes": [
            "Each row is an order-item/line-level record; use COUNT(DISTINCT order_id) for order counts.",
            "late_delivery_risk is a dataset-provided label (1=late risk, 0=not late), not a model prediction.",
            "days_for_shipping_real and days_for_shipment_scheduled can be compared to study delivery lateness.",
            "The dataset spans multiple markets and countries; it is not limited to Argentina.",
        ],
    }

_FORBIDDEN_SQL = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|copy|export|import|"
    r"install|load|pragma|call|execute|read_csv|read_parquet|read_json|sqlite_scan|postgres_scan)\b",
    re.IGNORECASE,
)

def query_dataset_sql(sql: str) -> dict:
    """Run a read-only SELECT/WITH query on dataco_orders; returns at most 200 rows."""
    if not isinstance(sql, str) or not sql.strip():
        return {"error": "Cần cung cấp câu SQL SELECT."}
    cleaned = sql.strip().rstrip(";")
    if ";" in cleaned:
        return {"error": "Chỉ cho phép một câu SQL mỗi lần."}
    if not (cleaned.lower().startswith("select ") or cleaned.lower().startswith("with ")):
        return {"error": "Chỉ cho phép câu lệnh SELECT hoặc WITH...SELECT."}
    if _FORBIDDEN_SQL.search(cleaned):
        return {"error": "SQL chứa từ khóa/hàm không được phép."}
    if not re.search(r"\bdataco_orders\b", cleaned, re.IGNORECASE):
        return {"error": "Truy vấn phải sử dụng bảng dataco_orders."}
    try:
        con = _get_conn()
        df = con.execute(f"SELECT * FROM ({cleaned}) AS user_query LIMIT {MAX_ROWS}").fetchdf()
        return {
            "n_rows_returned": len(df),
            "columns": list(df.columns),
            "rows": json.loads(df.to_json(orient="records", date_format="iso")),
        }
    except Exception as exc:
        return {"error": f"Lỗi SQL DataCo: {exc}"}

def get_order_info(order_id: str) -> dict:
    """Look up line-level records for a DataCo order ID."""
    try:
        con = _get_conn()
        rows = con.execute(
            "SELECT order_id, order_status, delivery_status, order_country, order_region, "
            "order_city, category_name, shipping_mode, days_for_shipping_real, "
            "days_for_shipment_scheduled, late_delivery_risk, sales, order_item_total "
            "FROM dataco_orders WHERE CAST(order_id AS VARCHAR) = ? LIMIT 20",
            [str(order_id)],
        ).fetchdf()
        if rows.empty:
            return {"found": False, "order_id": str(order_id)}
        return {
            "found": True,
            "order_id": str(order_id),
            "n_line_items_returned": len(rows),
            "items": json.loads(rows.to_json(orient="records", date_format="iso")),
        }
    except Exception as exc:
        return {"found": False, "error": str(exc)}

def top_risky_regions(n: int = 5) -> dict:
    """Rank destination regions by the dataset's observed late-delivery risk label."""
    try:
        limit = max(1, min(int(n), 20))
        con = _get_conn()
        df = con.execute(
            "SELECT order_region, COUNT(DISTINCT order_id) AS n_orders, "
            "ROUND(100.0 * AVG(TRY_CAST(late_delivery_risk AS DOUBLE)), 2) AS late_risk_pct, "
            "ROUND(AVG(TRY_CAST(days_for_shipping_real AS DOUBLE)), 2) AS avg_actual_shipping_days, "
            "ROUND(AVG(TRY_CAST(days_for_shipment_scheduled AS DOUBLE)), 2) AS avg_scheduled_days "
            "FROM dataco_orders WHERE order_region IS NOT NULL AND TRIM(CAST(order_region AS VARCHAR)) <> '' "
            "GROUP BY order_region ORDER BY late_risk_pct DESC NULLS LAST LIMIT ?",
            [limit],
        ).fetchdf()
        return {"rows": json.loads(df.to_json(orient="records")), "metric": "dataset label Late_delivery_risk"}
    except Exception as exc:
        return {"error": str(exc)}

SYSTEM_PROMPT = """Bạn là AI Logistics Agent đang phục vụ bộ dữ liệu DataCo Supply Chain (quốc tế), KHÔNG phải Olist.
Chỉ sử dụng dữ liệu trong bảng DuckDB dataco_orders qua các tool được cung cấp.
Các công cụ:
- describe_dataset: xem schema chính xác trước khi viết SQL.
- query_dataset_sql: chạy truy vấn SELECT/WITH chỉ đọc trên dataco_orders.
- get_order_info: tra cứu Order Id của DataCo; một đơn có thể có nhiều dòng sản phẩm.
- top_risky_regions: xếp hạng khu vực theo nhãn rủi ro giao trễ có sẵn.
Các cột thường dùng (đã chuẩn hóa thành snake_case): order_id, order_status, delivery_status,
days_for_shipping_real, days_for_shipment_scheduled, late_delivery_risk, order_country,
order_region, order_state, order_city, market, shipping_mode, category_name, sales,
order_item_total, order_item_quantity, order_date_dateorders, shipping_date_dateorders.
Luôn gọi describe_dataset nếu chưa chắc tên cột. Đếm đơn bằng COUNT(DISTINCT order_id) vì mỗi đơn có thể có nhiều dòng.
late_delivery_risk là nhãn đã có trong dataset, không được gọi là dự đoán của model ML.
Không khẳng định DataCo có seller-to-customer coordinates hoặc bản đồ tuyến đường Olist.
Không dùng các model ML Olist để dự đoán cho DataCo. Nếu cần dự đoán đơn mới, nói rõ model DataCo chưa được huấn luyện.
Chỉ trả lời bằng tiếng Việt, nêu số liệu và cách tính ngắn gọn; không bịa dữ liệu.
"""

TOOLS = [
    {
        "name": "describe_dataset",
        "description": "Xem schema và số dòng của DataCo Supply Chain table dataco_orders. Nên gọi trước khi viết SQL.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "query_dataset_sql",
        "description": "Chạy SQL SELECT/WITH chỉ đọc trên bảng dataco_orders; bắt buộc truy vấn bảng này, tối đa 200 dòng trả về.",
        "input_schema": {
            "type": "object",
            "properties": {"sql": {"type": "string", "description": "Một câu SELECT/WITH trên dataco_orders"}},
            "required": ["sql"],
        },
    },
    {
        "name": "get_order_info",
        "description": "Tra cứu thông tin DataCo theo Order Id; một đơn có thể có nhiều dòng sản phẩm.",
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "top_risky_regions",
        "description": "Xếp hạng các khu vực order_region theo tỷ lệ nhãn late_delivery_risk có sẵn.",
        "input_schema": {
            "type": "object",
            "properties": {"n": {"type": "integer", "default": 5}},
        },
    },
]
