"""
Các "tool" (hàm nghiệp vụ) mà AI Logistics Agent có thể gọi:
- Tra cứu đơn hàng theo order_id
- Dự đoán rủi ro giao trễ + số ngày giao hàng cho một đơn mới
- Thống kê hiệu suất giao hàng theo bang, theo seller, theo ngành hàng
- Tìm các khu vực / ngành hàng rủi ro cao nhất

File này KHÔNG phụ thuộc vào Anthropic SDK -> có thể test độc lập,
và cũng được agent.py import để expose thành "tools" cho LLM.
"""
import json
import os
from pathlib import Path
from functools import lru_cache

import duckdb
import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed_dataset.csv"
# DuckDB đọc từ bản .parquet (đã có sẵn trong data/) thay vì .csv: nhanh hơn và tránh lỗi
# parse CSV do vài dòng city name chứa dấu phẩy trong ngoặc kép bị auto-detect sai delimiter/quote.
DATA_PATH_PARQUET = BASE_DIR / "data" / "processed_dataset.parquet"

# Các file CSV GỐC (chưa gộp) — dùng cho các câu hỏi cần chi tiết mức DÒNG (từng sản phẩm,
# từng lượt thanh toán...) mà bảng 'orders' (đã gộp mỗi đơn 1 dòng) không còn giữ được.
# Ví dụ: 1 đơn có 3 sản phẩm thuộc 3 category khác nhau -> bảng 'orders' chỉ lưu category của
# sản phẩm ĐẦU TIÊN, nên câu hỏi "doanh thu theo category" PHẢI dùng order_items, không dùng orders.
_RAW_TABLES = {
    "order_items": "olist_order_items_dataset.csv",
    "order_payments": "olist_order_payments_dataset.csv",
    "order_reviews": "olist_order_reviews_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}
_MARKETING_TABLES = {
    "marketing_leads": "olist_marketing_qualified_leads_dataset.csv",
    "closed_deals": "olist_closed_deals_dataset.csv",
}

# Các cột ngày giờ cần parse lại khi đọc từ CSV (CSV không giữ dtype datetime)
_DATE_COLUMNS = [
    "order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
    "order_delivered_customer_date", "order_estimated_delivery_date",
]
MODEL_DIR = BASE_DIR / "models"
HISTORY_LOOKUP_PATH = MODEL_DIR / "history_lookup.json"

FEATURE_DEFAULTS = {
    "n_items": 1,
    "total_price": 100.0,
    "total_freight": 20.0,
    "payment_value": 120.0,
    "payment_installments": 1,
    "product_weight_g": 1000.0,
    "product_volume_cm3": 6000.0,
    "distance_km": 500.0,
    "estimated_days": 20,
    "purchase_dow": 2,
    "purchase_month": 6,
    "purchase_hour": 12,
    "same_state": 0,
    "customer_state": "SP",
    "seller_state": "SP",
    "product_category_name_english": "unknown",
    # --- giá trị mặc định cho đặc trưng mới (khớp với train_models.py) ---
    "n_distinct_sellers": 1,
    "n_distinct_categories": 1,
    "total_weight_g": 1000.0,
    "max_weight_g": 1000.0,
    "total_volume_cm3": 6000.0,
    "is_weekend": 0,
    "is_peak_season": 0,
    "price_per_item": 100.0,
    "freight_to_price_ratio": 0.2,
    "log_total_price": 4.615,   # log1p(100)
    "log_total_freight": 3.045,  # log1p(20)
    "log_distance_km": 6.217,   # log1p(500)
    "log_total_weight_g": 6.909,  # log1p(1000)
    "payment_type": "credit_card",
}


@lru_cache(maxsize=1)
def _load_model_metadata() -> dict:
    meta_path = MODEL_DIR / "model_metadata.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@lru_cache(maxsize=1)
def _load_history_lookup() -> dict:
    """Bảng tra cứu đặc trưng lịch sử (seller/ngành hàng/cặp bang/bang khách), do
    data_pipeline.py tạo ra từ toàn bộ dữ liệu đã có. Seller/ngành hàng/bang chưa từng
    xuất hiện sẽ tự rơi về mức trung bình toàn cục (priors) — không lỗi, không NaN."""
    if HISTORY_LOOKUP_PATH.exists():
        with open(HISTORY_LOOKUP_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"priors": {"late": 0.081, "days": 12.5, "carrier_h": 67.0}, "seller": {}, "cat": {}, "pair": {}, "cstate": {}}


def _hist_defaults_from_priors() -> dict:
    lk = _load_history_lookup()
    p = lk["priors"]
    return {
        "seller_hist_late": p["late"], "seller_hist_days": p["days"],
        "seller_hist_carrier_h": p["carrier_h"], "seller_hist_n": 0,
        "cat_hist_late": p["late"], "cat_hist_days": p["days"], "cat_hist_n": 0,
        "pair_hist_late": p["late"], "pair_hist_days": p["days"], "pair_hist_n": 0,
        "cstate_hist_late": p["late"], "cstate_hist_days": p["days"], "cstate_hist_n": 0,
    }


def _apply_history_lookup(f: dict, seller_id: str | None) -> dict:
    """Tra cứu và điền đặc trưng lịch sử dựa trên seller_id (nếu có), ngành hàng, và
    cặp/bang đã có trong f. Chỉ ghi đè các trường KHÔNG được người dùng cung cấp trực tiếp."""
    lk = _load_history_lookup()

    def fill(prefix, key):
        rec = lk.get(prefix, {}).get(key)
        if rec is None:
            return
        f[f"{prefix}_hist_late"] = rec["late"]
        f[f"{prefix}_hist_days"] = rec["days"]
        f[f"{prefix}_hist_n"] = rec["n"]
        if prefix == "seller" and "carrier_h" in rec:
            f["seller_hist_carrier_h"] = rec["carrier_h"]

    if seller_id:
        fill("seller", seller_id)
    cat = f.get("product_category_name_english")
    if cat and cat != "unknown":
        fill("cat", cat)
    pair_key = f"{f.get('seller_state', '')}|{f.get('customer_state', '')}"
    fill("pair", pair_key)
    if f.get("customer_state"):
        fill("cstate", f["customer_state"])
    return f


@lru_cache(maxsize=1)
def _load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, parse_dates=_DATE_COLUMNS)


@lru_cache(maxsize=1)
def _load_models():
    clf = joblib.load(MODEL_DIR / "late_delivery_classifier.joblib")
    reg = joblib.load(MODEL_DIR / "delivery_days_regressor.joblib")
    return clf, reg


@lru_cache(maxsize=1)
def _get_duckdb_conn():
    """Đăng ký bảng 'orders' (processed_dataset) + các bảng GỐC (order_items, order_payments,
    order_reviews, products, sellers, customers, category_translation) trong cùng 1 kết nối
    DuckDB, đọc trực tiếp từ CSV/parquet — không load vào RAM qua pandas."""
    con = duckdb.connect(database=":memory:")
    if DATA_PATH_PARQUET.exists():
        con.execute(f"CREATE VIEW orders AS SELECT * FROM read_parquet('{DATA_PATH_PARQUET.as_posix()}')")
    else:
        con.execute(
            f"CREATE VIEW orders AS SELECT * FROM read_csv_auto("
            f"'{DATA_PATH.as_posix()}', quote='\"', escape='\"', strict_mode=false)"
        )

    for table_name, filename in {**_RAW_TABLES, **_MARKETING_TABLES}.items():
        path = BASE_DIR / "data" / filename
        if path.exists():
            con.execute(
                f"CREATE VIEW {table_name} AS SELECT * FROM read_csv_auto("
                f"'{path.as_posix()}', quote='\"', escape='\"', strict_mode=false)"
            )
    return con


def _raw_tables_available() -> bool:
    """True nếu đủ file CSV gốc để trả lời các câu hỏi cần chi tiết mức dòng (không chỉ dùng
    bảng 'orders' đã gộp). Dùng để tools/providers báo lỗi rõ ràng thay vì lỗi SQL khó hiểu."""
    return all((BASE_DIR / "data" / f).exists() for f in _RAW_TABLES.values())


# Các từ khoá SQL không cho phép (chỉ cho phép truy vấn đọc dữ liệu, không cho sửa/xoá)
_SQL_FORBIDDEN_KEYWORDS = [
    "insert", "update", "delete", "drop", "alter", "attach", "detach",
    "copy", "pragma", "create", "call", "install", "load", "export", "import",
]


def _df_to_json_safe_records(df) -> list:
    """Chuyển 1 DataFrame thành list[dict] AN TOÀN để json.dumps() — tránh lỗi
    'Object of type Timestamp/Decimal/int64 is not JSON serializable' (Gemini SDK tự lo
    việc này nên trước đây không lộ lỗi, nhưng OpenAI/Groq/xAI cần tool result là JSON
    string thuần, nên phải tự đảm bảo). Đi qua JSON string trung gian (to_json) rồi
    parse lại — cách rẻ, chắc chắn, không cần liệt kê từng kiểu dữ liệu đặc biệt."""
    return json.loads(df.to_json(orient="records", date_format="iso"))


def describe_dataset() -> dict:
    """
    Liệt kê schema của bảng 'orders' (đã gộp mỗi đơn 1 dòng) VÀ các bảng gốc chi tiết mức
    dòng (order_items, order_payments, order_reviews, products, sellers, customers,
    category_translation) để agent biết dataset có những trường/bảng nào trước khi viết SQL
    bằng tool `query_dataset_sql`. LUÔN gọi tool này trước nếu chưa chắc tên bảng/cột.
    """
    con = _get_duckdb_conn()
    all_tables = {**_RAW_TABLES, **_MARKETING_TABLES}

    tables = ["orders"] + [
        t for t in all_tables.keys() if (BASE_DIR / "data" / all_tables[t]).exists()
    ]

    result = {"tables": {}}
    for table in tables:
        schema = con.execute(f"DESCRIBE {table}").fetchdf()
        result["tables"][table] = {
            "n_rows": con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0],
            "columns": [{"name": r["column_name"], "type": r["column_type"]} for _, r in schema.iterrows()],
        }

    # Giữ tương thích ngược: các key cũ (n_rows, columns, sample_rows) vẫn trỏ về bảng 'orders'.
    result["n_rows"] = result["tables"]["orders"]["n_rows"]
    result["columns"] = result["tables"]["orders"]["columns"]
    result["sample_rows"] = _df_to_json_safe_records(con.execute("SELECT * FROM orders LIMIT 3").fetchdf())
    result["notes"] = (
        "Bảng 'orders' chứa cả đơn chưa giao (order_status != 'delivered'); is_late và "
        "actual_delivery_days chỉ có giá trị (không NULL) với đơn đã 'delivered'. "
        "Dùng WHERE order_status = 'delivered' khi tính tỉ lệ trễ / thời gian giao thực tế. "
        "QUAN TRỌNG: bảng 'orders' đã GỘP mỗi đơn thành 1 dòng — category/seller trong đó chỉ "
        "là của SẢN PHẨM ĐẦU TIÊN trong đơn, KHÔNG chính xác cho đơn có nhiều sản phẩm/nhiều "
        "seller/nhiều category khác nhau. Với câu hỏi về doanh thu/số lượng THEO category, "
        "THEO seller, THEO sản phẩm, hoặc cần đối chiếu payment với price+freight ở mức dòng, "
        "PHẢI JOIN từ bảng order_items (mỗi dòng = 1 sản phẩm trong đơn) thay vì dùng 'orders'."
    )
    return result


def query_dataset_sql(sql: str) -> dict:
    """
    Chạy một câu SQL (SELECT hoặc WITH...SELECT) TUỲ Ý trên các bảng có sẵn: 'orders'
    (đã gộp mỗi đơn 1 dòng) và các bảng gốc mức dòng — 'order_items', 'order_payments',
    'order_reviews', 'products', 'sellers', 'customers', 'category_translation' — để trả
    lời bất kỳ câu hỏi thống kê/lọc/nhóm/JOIN nào không có sẵn tool riêng.
    Luôn gọi `describe_dataset` trước nếu chưa biết chính xác tên bảng/cột.
    Kết quả giới hạn 200 dòng để tránh trả về quá nhiều dữ liệu.
    """
    cleaned = sql.strip().rstrip(";")
    lowered = cleaned.lower()
    if not (lowered.startswith("select") or lowered.startswith("with")):
        return {"error": "Chỉ cho phép câu lệnh SELECT hoặc WITH...SELECT (không được sửa/xoá dữ liệu)."}
    if any(f" {kw} " in f" {lowered} " or lowered.startswith(kw + " ") for kw in _SQL_FORBIDDEN_KEYWORDS):
        return {"error": "Câu lệnh chứa từ khoá không được phép (chỉ đọc dữ liệu, không sửa/xoá/cấu hình)."}
    try:
        con = _get_duckdb_conn()
        result_df = con.execute(f"SELECT * FROM ({cleaned}) AS q LIMIT 200").fetchdf()
        return {
            "n_rows_returned": len(result_df),
            "columns": list(result_df.columns),
            "rows": json.loads(result_df.to_json(orient="records", date_format="iso")),
        }
    except Exception as e:  # noqa: BLE001
        return {"error": f"Lỗi SQL: {e}"}


def train_custom_model(
    target: str,
    feature_columns: list = None,
    filter_sql: str = None,
    task: str = "classification",
) -> dict:
    """
    Huấn luyện NHANH một model tạm thời (RandomForest) khi câu hỏi cần một dự đoán/phân tích
    KHÔNG nằm trong 2 model chính đã deploy (is_late, actual_delivery_days) — ví dụ dự đoán
    review_score, hoặc train riêng cho một subset (vd chỉ đơn ở bang RJ). Model này KHÔNG
    ghi đè lên 2 model production trong models/.

    - target: tên cột nhãn cần dự đoán (vd 'review_score', 'is_late').
    - feature_columns: danh sách cột đặc trưng dùng để train; mặc định dùng bộ feature chuẩn
      (numeric: n_items, total_price, total_freight, payment_value, product_weight_g,
      product_volume_cm3, distance_km, estimated_days; categorical: customer_state,
      seller_state, product_category_name_english).
    - filter_sql: điều kiện WHERE tuỳ chọn để lọc subset trước khi train (vd "customer_state = 'RJ'").
    - task: 'classification' hoặc 'regression'.

    Trả về: số dòng dùng để train, metric trên test set (20%), và top feature importance.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import mean_absolute_error, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    default_numeric = [
        "n_items", "total_price", "total_freight", "payment_value",
        "product_weight_g", "product_volume_cm3", "distance_km", "estimated_days",
    ]
    default_categorical = ["customer_state", "seller_state", "product_category_name_english"]

    where_clause = f"WHERE order_status = 'delivered' AND {filter_sql}" if filter_sql else "WHERE order_status = 'delivered'"
    con = _get_duckdb_conn()
    try:
        df = con.execute(f"SELECT * FROM orders {where_clause}").fetchdf()
    except Exception as e:  # noqa: BLE001
        return {"error": f"Lỗi filter_sql: {e}"}

    if target not in df.columns:
        return {"error": f"Không tìm thấy cột target '{target}' trong dataset."}

    df = df.dropna(subset=[target])
    if len(df) < 100:
        return {"error": f"Chỉ có {len(df)} dòng sau khi lọc — quá ít để train (cần >= 100)."}

    numeric_features = [c for c in (feature_columns or default_numeric) if c in df.columns and c != target]
    categorical_features = [c for c in default_categorical if c in df.columns and c != target and (not feature_columns or c in feature_columns)]
    if not numeric_features and not categorical_features:
        numeric_features = [c for c in default_numeric if c in df.columns and c != target]

    X = df[numeric_features + categorical_features]
    y = df[target]
    if task == "classification":
        y = y.astype(int)

    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_features),
    ])

    stratify = y if task == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    model = (
        RandomForestClassifier(n_estimators=200, max_depth=8, n_jobs=-1, random_state=42)
        if task == "classification"
        else RandomForestRegressor(n_estimators=200, max_depth=8, n_jobs=-1, random_state=42)
    )
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)

    metrics = {}
    if task == "classification":
        proba = pipe.predict_proba(X_test)
        if proba.shape[1] == 2:
            metrics["roc_auc"] = round(roc_auc_score(y_test, proba[:, 1]), 4)
        metrics["accuracy"] = round(pipe.score(X_test, y_test), 4)
    else:
        pred = pipe.predict(X_test)
        metrics["mae"] = round(mean_absolute_error(y_test, pred), 4)
        metrics["r2"] = round(pipe.score(X_test, y_test), 4)

    feature_names = list(pipe.named_steps["prep"].get_feature_names_out())
    importances = pipe.named_steps["model"].feature_importances_
    top_features = sorted(zip(feature_names, importances), key=lambda x: -x[1])[:10]

    return {
        "target": target,
        "task": task,
        "n_rows_used": len(df),
        "features_used": numeric_features + categorical_features,
        "metrics": metrics,
        "top_feature_importance": [{"feature": f, "importance": round(float(i), 4)} for f, i in top_features],
        "note": "Model tạm thời — chỉ dùng để phân tích/trả lời câu hỏi, không được lưu vào models/.",
    }


def get_order_info(order_id: str) -> dict:
    """Tra cứu thông tin & tình trạng một đơn hàng có sẵn trong dataset."""
    df = _load_dataset()
    row = df[df["order_id"] == order_id]
    if row.empty:
        return {"found": False, "message": f"Không tìm thấy order_id={order_id}"}
    r = row.iloc[0]
    return {
        "found": True,
        "order_id": order_id,
        "status": r["order_status"],
        "customer_state": r["customer_state"],
        "seller_state": r["seller_state"],
        "product_category": r["product_category_name_english"],
        "purchase_date": str(r["order_purchase_timestamp"]),
        "estimated_delivery_date": str(r["order_estimated_delivery_date"]),
        "actual_delivery_date": str(r["order_delivered_customer_date"]) if pd.notna(r["order_delivered_customer_date"]) else None,
        "is_late": (None if pd.isna(r["is_late"]) else bool(r["is_late"])),
        "actual_delivery_days": (None if pd.isna(r["actual_delivery_days"]) else round(float(r["actual_delivery_days"]), 1)),
        "distance_km": round(float(r["distance_km"]), 1) if pd.notna(r["distance_km"]) else None,
        "review_score": (None if pd.isna(r["review_score"]) else float(r["review_score"])),
    }


def _build_feature_row(user: dict) -> dict:
    """
    Dựng đủ dòng đặc trưng cho model từ các trường người dùng CUNG CẤP, và TỰ TÍNH các đặc
    trưng phái sinh (same_state, is_peak_season, log-transform, tỉ lệ, tổng khối lượng,
    và đặc trưng LỊCH SỬ theo seller/ngành hàng/bang) từ đầu vào — thay vì để nguyên giá trị
    mặc định lỗi thời. Trường nào người dùng tự cung cấp thì được giữ nguyên, không bị ghi đè.
    `seller_id` (nếu có) chỉ dùng để tra cứu lịch sử, KHÔNG đưa thẳng vào model.
    """
    given = {k: v for k, v in user.items() if v is not None and k != "seller_id"}
    f = dict(FEATURE_DEFAULTS)
    f.update(_hist_defaults_from_priors())
    f.update(given)

    for col in ("customer_state", "seller_state"):
        f[col] = str(f[col]).upper().strip()
    n_items = max(int(f["n_items"]), 1)

    def missing(k):
        return k not in given

    if missing("payment_value"):
        f["payment_value"] = f["total_price"] + f["total_freight"]
    if missing("total_weight_g"):
        f["total_weight_g"] = f["product_weight_g"] * n_items
    if missing("max_weight_g"):
        f["max_weight_g"] = f["product_weight_g"]
    if missing("total_volume_cm3"):
        f["total_volume_cm3"] = f["product_volume_cm3"] * n_items
    # Chỉ suy ra same_state khi người dùng có nói ít nhất 1 bang; nếu không nói gì về bang
    # thì giữ mặc định (khác bang), tránh mâu thuẫn với khoảng cách mặc định 500km.
    if missing("same_state") and ("customer_state" in given or "seller_state" in given):
        f["same_state"] = int(f["customer_state"] == f["seller_state"])
    if missing("is_weekend"):
        f["is_weekend"] = int(int(f["purchase_dow"]) in (5, 6))
    if missing("is_peak_season"):
        f["is_peak_season"] = int(int(f["purchase_month"]) in (11, 12))
    if missing("price_per_item"):
        f["price_per_item"] = f["total_price"] / n_items
    if missing("freight_to_price_ratio"):
        f["freight_to_price_ratio"] = f["total_freight"] / max(f["total_price"], 0.01)
    for log_col, src in (("log_total_price", "total_price"), ("log_total_freight", "total_freight"),
                         ("log_distance_km", "distance_km"), ("log_total_weight_g", "total_weight_g")):
        if missing(log_col):
            f[log_col] = float(np.log1p(f[src]))

    # Tra cứu lịch sử THEO seller cụ thể (nếu biết seller_id) / ngành hàng / bang — chỉ ghi
    # đè các cột hist_* mà người dùng KHÔNG tự truyền vào.
    hist_from_lookup = _apply_history_lookup(dict(f), user.get("seller_id"))
    for k, v in hist_from_lookup.items():
        if "_hist_" in k and missing(k):
            f[k] = v
    return f


def predict_new_order(**kwargs) -> dict:
    """
    Dự đoán rủi ro giao trễ (%) và số ngày giao hàng ước tính cho một đơn MỚI,
    dựa trên các đặc trưng biết được tại thời điểm đặt hàng.
    Thiếu trường nào sẽ dùng giá trị mặc định hợp lý (trung bình dataset).
    """
    clf, reg = _load_models()
    features = _build_feature_row(kwargs)
    X = pd.DataFrame([features])

    late_proba = float(clf.predict_proba(X)[:, 1][0])
    pred_days = float(reg.predict(X)[0])

    # Dùng ngưỡng phân loại đã tối ưu F1 lúc train (thay vì mặc định 0.5) để chia mức rủi ro —
    # nếu chưa train lại (chưa có model_metadata.json mới) thì tự rơi về 0.5 an toàn.
    threshold = _load_model_metadata().get("classifier_threshold", 0.5)
    if late_proba >= threshold:
        risk_level = "CAO"
    elif late_proba >= threshold / 2:
        risk_level = "TRUNG BÌNH"
    else:
        risk_level = "THẤP"

    return {
        "late_probability_pct": round(late_proba * 100, 1),
        "risk_level": risk_level,
        "predicted_delivery_days": round(pred_days, 1),
        "input_used": features,
    }


def get_state_stats(state: str) -> dict:
    """Thống kê hiệu suất giao hàng cho khách hàng ở một bang (customer_state), ví dụ 'SP', 'RJ'."""
    df = _load_dataset()
    state = state.upper().strip()
    sub = df[(df["customer_state"] == state) & (df["order_status"] == "delivered")]
    if sub.empty:
        return {"found": False, "message": f"Không có dữ liệu cho bang '{state}'"}
    return {
        "found": True,
        "state": state,
        "n_orders": int(len(sub)),
        "late_rate_pct": round(sub["is_late"].mean() * 100, 2),
        "avg_delivery_days": round(sub["actual_delivery_days"].mean(), 2),
        "avg_distance_km": round(sub["distance_km"].mean(), 1),
        "avg_review_score": round(sub["review_score"].mean(), 2) if sub["review_score"].notna().any() else None,
    }


def get_seller_stats(seller_id: str) -> dict:
    """Thống kê hiệu suất giao hàng của một seller cụ thể."""
    df = _load_dataset()
    sub = df[(df["seller_id"] == seller_id) & (df["order_status"] == "delivered")]
    if sub.empty:
        return {"found": False, "message": f"Không có dữ liệu cho seller_id='{seller_id}'"}
    return {
        "found": True,
        "seller_id": seller_id,
        "seller_state": sub["seller_state"].iloc[0],
        "n_orders": int(len(sub)),
        "late_rate_pct": round(sub["is_late"].mean() * 100, 2),
        "avg_delivery_days": round(sub["actual_delivery_days"].mean(), 2),
        "avg_review_score": round(sub["review_score"].mean(), 2) if sub["review_score"].notna().any() else None,
    }


def get_category_stats(category: str) -> dict:
    """Thống kê hiệu suất giao hàng theo ngành hàng (product_category_name_english), ví dụ 'furniture_decor'."""
    df = _load_dataset()
    sub = df[(df["product_category_name_english"] == category) & (df["order_status"] == "delivered")]
    if sub.empty:
        return {"found": False, "message": f"Không có dữ liệu cho ngành hàng '{category}'"}
    return {
        "found": True,
        "category": category,
        "n_orders": int(len(sub)),
        "late_rate_pct": round(sub["is_late"].mean() * 100, 2),
        "avg_delivery_days": round(sub["actual_delivery_days"].mean(), 2),
        "avg_freight": round(sub["total_freight"].mean(), 2),
    }


def top_risky_states(n: int = 5, min_orders: int = 50) -> dict:
    """Trả về top N bang (customer_state) có tỉ lệ giao trễ cao nhất."""
    df = _load_dataset()
    sub = df[df["order_status"] == "delivered"]
    g = sub.groupby("customer_state").agg(
        n_orders=("order_id", "count"),
        late_rate_pct=("is_late", lambda x: round(x.mean() * 100, 2)),
        avg_delivery_days=("actual_delivery_days", lambda x: round(x.mean(), 2)),
    ).reset_index()
    g = g[g["n_orders"] >= min_orders].sort_values("late_rate_pct", ascending=False).head(n)
    return {"results": _df_to_json_safe_records(g)}


def top_risky_categories(n: int = 5, min_orders: int = 50) -> dict:
    """Trả về top N ngành hàng có tỉ lệ giao trễ cao nhất."""
    df = _load_dataset()
    sub = df[df["order_status"] == "delivered"]
    g = sub.groupby("product_category_name_english").agg(
        n_orders=("order_id", "count"),
        late_rate_pct=("is_late", lambda x: round(x.mean() * 100, 2)),
        avg_delivery_days=("actual_delivery_days", lambda x: round(x.mean(), 2)),
    ).reset_index()
    g = g[g["n_orders"] >= min_orders].sort_values("late_rate_pct", ascending=False).head(n)
    return {"results": _df_to_json_safe_records(g)}



def _map_date_filters(alias: str, start_date: str | None, end_date: str | None):
    """Tạo WHERE predicates + params dùng chung cho các API map theo ngày đặt hàng."""
    filters = [
        f"{alias}.order_status = 'delivered'",
        f"{alias}.seller_state IS NOT NULL",
        f"{alias}.customer_state IS NOT NULL",
    ]
    params = []
    if start_date:
        filters.append(f"CAST({alias}.order_purchase_timestamp AS DATE) >= CAST(? AS DATE)")
        params.append(start_date)
    if end_date:
        filters.append(f"CAST({alias}.order_purchase_timestamp AS DATE) <= CAST(? AS DATE)")
        params.append(end_date)
    return " AND ".join(filters), params


def _map_summary(con, where_sql: str, params: list) -> dict:
    row = con.execute(
        f"""
        SELECT
            COUNT(*) AS n_orders,
            COALESCE(SUM(CASE WHEN is_late = 1 THEN 1 ELSE 0 END), 0) AS n_late,
            AVG(is_late) * 100 AS late_rate_pct,
            AVG(actual_delivery_days) AS avg_delivery_days,
            AVG(distance_km) AS avg_distance_km
        FROM orders
        WHERE {where_sql}
        """,
        params,
    ).fetchone()
    n_orders, n_late, late_rate_pct, avg_delivery_days, avg_distance_km = row
    return {
        "n_orders": int(n_orders or 0),
        "n_late": int(n_late or 0),
        "late_rate_pct": round(float(late_rate_pct or 0), 2),
        "avg_delivery_days": round(float(avg_delivery_days or 0), 2),
        "avg_distance_km": round(float(avg_distance_km or 0), 1),
    }


def _map_date_bounds(con) -> dict:
    row = con.execute(
        """
        SELECT
            CAST(MIN(order_purchase_timestamp) AS DATE),
            CAST(MAX(order_purchase_timestamp) AS DATE)
        FROM orders
        WHERE order_status = 'delivered'
        """
    ).fetchone()
    return {
        "date_min": str(row[0]) if row[0] is not None else None,
        "date_max": str(row[1]) if row[1] is not None else None,
    }


def get_state_flow_map(start_date: str | None = None, end_date: str | None = None) -> dict:
    """
    Luồng seller_state -> customer_state, có bộ lọc theo ngày đặt hàng.
    Giữ nguyên semantics của map cũ, nhưng bổ sung summary + date bounds để frontend
    có thể lọc theo thời gian mà không phá endpoint hiện tại.
    """
    con = _get_duckdb_conn()
    where_sql, params = _map_date_filters("o", start_date, end_date)

    df = con.execute(
        f"""
        WITH state_centroid AS (
            SELECT state, AVG(lat) AS lat, AVG(lng) AS lng FROM (
                SELECT seller_state AS state, seller_lat AS lat, seller_lng AS lng
                FROM orders
                WHERE seller_lat IS NOT NULL
                UNION ALL
                SELECT customer_state AS state, cust_lat AS lat, cust_lng AS lng
                FROM orders
                WHERE cust_lat IS NOT NULL
            ) GROUP BY state
        )
        SELECT
            o.seller_state AS from_state,
            o.customer_state AS to_state,
            fc.lat AS from_lat,
            fc.lng AS from_lng,
            tc.lat AS to_lat,
            tc.lng AS to_lng,
            COUNT(*) AS n_orders,
            ROUND(AVG(o.is_late) * 100, 2) AS late_rate_pct,
            ROUND(AVG(o.distance_km), 1) AS avg_distance_km,
            ROUND(AVG(o.actual_delivery_days), 2) AS avg_delivery_days,
            SUM(CASE WHEN o.is_late = 1 THEN 1 ELSE 0 END) AS n_late
        FROM orders o
        JOIN state_centroid fc ON o.seller_state = fc.state
        JOIN state_centroid tc ON o.customer_state = tc.state
        WHERE {where_sql}
        GROUP BY
            o.seller_state, o.customer_state,
            fc.lat, fc.lng, tc.lat, tc.lng
        HAVING COUNT(*) >= 5
        ORDER BY n_orders DESC
        """,
        params,
    ).fetchdf()

    return {
        "flows": _df_to_json_safe_records(df),
        "summary": _map_summary(con, where_sql, params),
        **_map_date_bounds(con),
        "filter": {"start_date": start_date, "end_date": end_date},
    }


def get_state_risk_map(start_date: str | None = None, end_date: str | None = None) -> dict:
    """Thống kê risk theo customer_state để tô choropleth trên bản đồ Brazil."""
    con = _get_duckdb_conn()
    where_sql, params = _map_date_filters("o", start_date, end_date)

    df = con.execute(
        f"""
        SELECT
            o.customer_state AS state,
            COUNT(*) AS n_orders,
            SUM(CASE WHEN o.is_late = 1 THEN 1 ELSE 0 END) AS n_late,
            ROUND(AVG(o.is_late) * 100, 2) AS late_rate_pct,
            ROUND(AVG(o.actual_delivery_days), 2) AS avg_delivery_days,
            ROUND(AVG(o.distance_km), 1) AS avg_distance_km,
            ROUND(AVG(o.review_score), 2) AS avg_review_score
        FROM orders o
        WHERE {where_sql}
        GROUP BY o.customer_state
        HAVING COUNT(*) >= 5
        ORDER BY late_rate_pct DESC
        """,
        params,
    ).fetchdf()

    return {
        "states": _df_to_json_safe_records(df),
        "summary": _map_summary(con, where_sql, params),
        **_map_date_bounds(con),
        "filter": {"start_date": start_date, "end_date": end_date},
    }


def get_map_trend(start_date: str | None = None, end_date: str | None = None) -> dict:
    """Xu hướng theo tháng: orders, late rate và thời gian giao trung bình."""
    con = _get_duckdb_conn()
    where_sql, params = _map_date_filters("o", start_date, end_date)

    df = con.execute(
        f"""
        SELECT
            STRFTIME(DATE_TRUNC('month', o.order_purchase_timestamp), '%Y-%m') AS month,
            COUNT(*) AS n_orders,
            ROUND(AVG(o.is_late) * 100, 2) AS late_rate_pct,
            ROUND(AVG(o.actual_delivery_days), 2) AS avg_delivery_days
        FROM orders o
        WHERE {where_sql}
        GROUP BY 1
        ORDER BY 1
        """,
        params,
    ).fetchdf()

    return {
        "trend": _df_to_json_safe_records(df),
        **_map_date_bounds(con),
        "filter": {"start_date": start_date, "end_date": end_date},
    }


def get_order_sample_for_flow(
    from_state: str,
    to_state: str,
    limit: int = 40,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    """
    Lấy mẫu ngẫu nhiên các đơn thuộc 1 flow, có lọc theo thời gian.
    n_available là tổng số đơn khớp flow/filter, không còn bị giới hạn bởi LIMIT.
    """
    from_state = str(from_state).upper().strip()
    to_state = str(to_state).upper().strip()
    try:
        limit = max(1, min(int(limit), 200))
    except (TypeError, ValueError):
        limit = 40

    con = _get_duckdb_conn()
    filters = [
        "order_status = 'delivered'",
        "seller_state = ?",
        "customer_state = ?",
        "seller_lat IS NOT NULL",
        "cust_lat IS NOT NULL",
    ]
    base_params = [from_state, to_state]

    if start_date:
        filters.append("CAST(order_purchase_timestamp AS DATE) >= CAST(? AS DATE)")
        base_params.append(start_date)
    if end_date:
        filters.append("CAST(order_purchase_timestamp AS DATE) <= CAST(? AS DATE)")
        base_params.append(end_date)

    where_sql = " AND ".join(filters)
    count_row = con.execute(
        f"SELECT COUNT(*) FROM orders WHERE {where_sql}",
        base_params,
    ).fetchone()
    n_available = int(count_row[0] or 0)

    df = con.execute(
        f"""
        SELECT
            order_id, seller_id, seller_city, seller_state, seller_lat, seller_lng,
            customer_city, customer_state, cust_lat AS customer_lat, cust_lng AS customer_lng,
            is_late, actual_delivery_days, distance_km,
            order_purchase_timestamp, order_delivered_customer_date, order_estimated_delivery_date,
            product_category_name_english AS product_category,
            total_price, total_freight, payment_value, review_score
        FROM orders
        WHERE {where_sql}
        ORDER BY random()
        LIMIT ?
        """,
        base_params + [limit],
    ).fetchdf()

    return {
        "orders": _df_to_json_safe_records(df),
        "n_available": n_available,
        "filter": {"start_date": start_date, "end_date": end_date},
    }


def web_search(query: str, max_results: int = 5) -> dict:
    """
    Tìm kiếm thông tin trên web qua Tavily — CHỈ dùng cho câu hỏi NẰM NGOÀI phạm vi dataset
    Olist (ví dụ: tin tức logistics hiện tại, thông tin công ty vận chuyển, kiến thức chung).
    KHÔNG dùng cho câu hỏi có thể trả lời bằng dữ liệu nội bộ (predict_new_order,
    query_dataset_sql, get_state_stats...) — ưu tiên tool nội bộ trước vì dữ liệu chính xác
    và không tốn quota tìm kiếm.
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return {"error": "Chưa cấu hình TAVILY_API_KEY — không thể tìm kiếm web."}
    try:
        from tavily import TavilyClient
    except ImportError:
        return {"error": "Thiếu thư viện tavily-python. Cài bằng: pip install tavily-python"}

    try:
        client = TavilyClient(api_key=api_key)
        resp = client.search(query=query, max_results=max(1, min(int(max_results), 10)), include_answer=True)
    except Exception as e:  # noqa: BLE001
        detail = str(e) or type(e).__name__  # 1 số lỗi Tavily (vd key sai) có message rỗng
        return {"error": f"Lỗi khi gọi Tavily ({detail}) — kiểm tra lại TAVILY_API_KEY."}

    return {
        "answer": resp.get("answer"),
        "results": [
            {"title": r.get("title"), "url": r.get("url"), "content": r.get("content")}
            for r in resp.get("results", [])
        ],
    }


if __name__ == "__main__":
    import json
    print(json.dumps(get_state_stats("SP"), ensure_ascii=False, indent=2))
    print(json.dumps(predict_new_order(distance_km=2000, product_category_name_english="moveis_decoracao"), ensure_ascii=False, indent=2))
    print(json.dumps(top_risky_states(5), ensure_ascii=False, indent=2))
