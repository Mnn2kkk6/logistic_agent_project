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

# Các cột ngày giờ cần parse lại khi đọc từ CSV (CSV không giữ dtype datetime)
_DATE_COLUMNS = [
    "order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
    "order_delivered_customer_date", "order_estimated_delivery_date",
]
MODEL_DIR = BASE_DIR / "models"

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
}


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

    for table_name, filename in _RAW_TABLES.items():
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


def describe_dataset() -> dict:
    """
    Liệt kê schema của bảng 'orders' (đã gộp mỗi đơn 1 dòng) VÀ các bảng gốc chi tiết mức
    dòng (order_items, order_payments, order_reviews, products, sellers, customers,
    category_translation) để agent biết dataset có những trường/bảng nào trước khi viết SQL
    bằng tool `query_dataset_sql`. LUÔN gọi tool này trước nếu chưa chắc tên bảng/cột.
    """
    con = _get_duckdb_conn()
    tables = ["orders"] + [t for t in _RAW_TABLES if (BASE_DIR / "data" / _RAW_TABLES[t]).exists()]

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
    result["sample_rows"] = con.execute("SELECT * FROM orders LIMIT 3").fetchdf().to_dict(orient="records")
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


def predict_new_order(**kwargs) -> dict:
    """
    Dự đoán rủi ro giao trễ (%) và số ngày giao hàng ước tính cho một đơn MỚI,
    dựa trên các đặc trưng biết được tại thời điểm đặt hàng.
    Thiếu trường nào sẽ dùng giá trị mặc định hợp lý (trung bình dataset).
    """
    clf, reg = _load_models()
    features = dict(FEATURE_DEFAULTS)
    features.update({k: v for k, v in kwargs.items() if v is not None})
    X = pd.DataFrame([features])

    late_proba = float(clf.predict_proba(X)[:, 1][0])
    pred_days = float(reg.predict(X)[0])

    if late_proba >= 0.5:
        risk_level = "CAO"
    elif late_proba >= 0.25:
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
    return {"results": g.to_dict(orient="records")}


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
    return {"results": g.to_dict(orient="records")}


if __name__ == "__main__":
    import json
    print(json.dumps(get_state_stats("SP"), ensure_ascii=False, indent=2))
    print(json.dumps(predict_new_order(distance_km=2000, product_category_name_english="moveis_decoracao"), ensure_ascii=False, indent=2))
    print(json.dumps(top_risky_states(5), ensure_ascii=False, indent=2))
