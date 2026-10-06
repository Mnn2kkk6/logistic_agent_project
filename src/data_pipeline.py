"""
Data pipeline cho AI Logistics Agent (Olist E-commerce Dataset)
Load -> Merge -> Feature Engineering -> Xuất dataset đã xử lý.
"""
import pandas as pd
import numpy as np
from pathlib import Path

from src.history_features import add_history_features, build_lookup_tables

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def load_raw():
    orders = pd.read_csv(DATA_DIR / "olist_orders_dataset.csv", parse_dates=[
        "order_purchase_timestamp", "order_approved_at",
        "order_delivered_carrier_date", "order_delivered_customer_date",
        "order_estimated_delivery_date"
    ])
    items = pd.read_csv(DATA_DIR / "olist_order_items_dataset.csv", parse_dates=["shipping_limit_date"])
    payments = pd.read_csv(DATA_DIR / "olist_order_payments_dataset.csv")
    reviews = pd.read_csv(DATA_DIR / "olist_order_reviews_dataset.csv")
    products = pd.read_csv(DATA_DIR / "olist_products_dataset.csv")
    customers = pd.read_csv(DATA_DIR / "olist_customers_dataset.csv")
    sellers = pd.read_csv(DATA_DIR / "olist_sellers_dataset.csv")
    geo = pd.read_csv(DATA_DIR / "olist_geolocation_dataset.csv")
    cat_translation = pd.read_csv(DATA_DIR / "product_category_name_translation.csv")
    return orders, items, payments, reviews, products, customers, sellers, geo, cat_translation


def build_geo_lookup(geo: pd.DataFrame) -> pd.DataFrame:
    """Trung bình lat/lng theo zip prefix để tra cứu nhanh, tránh trùng lặp."""
    g = geo.groupby("geolocation_zip_code_prefix").agg(
        lat=("geolocation_lat", "mean"),
        lng=("geolocation_lng", "mean"),
    ).reset_index().rename(columns={"geolocation_zip_code_prefix": "zip_code_prefix"})
    return g


def build_dataset() -> pd.DataFrame:
    orders, items, payments, reviews, products, customers, sellers, geo, cat_translation = load_raw()
    geo_lookup = build_geo_lookup(geo)

    # Gộp items ở mức order: số lượng item, tổng giá, tổng phí ship, giá trị trung bình
    # + các đặc trưng MỚI: độ phức tạp đơn hàng (nhiều seller/category), tổng khối lượng/thể tích
    # thực tế (thay vì chỉ lấy của sản phẩm đầu tiên) — tất cả đều biết được TẠI THỜI ĐIỂM ĐẶT HÀNG.
    items_with_products = items.merge(products, on="product_id", how="left")
    item_agg = items_with_products.groupby("order_id").agg(
        n_items=("order_item_id", "count"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        n_distinct_sellers=("seller_id", "nunique"),
        n_distinct_categories=("product_category_name", "nunique"),
        total_weight_g=("product_weight_g", "sum"),
        max_weight_g=("product_weight_g", "max"),
    ).reset_index()
    item_agg["total_volume_cm3"] = (
        items_with_products.assign(vol=lambda d: d.product_length_cm * d.product_height_cm * d.product_width_cm)
        .groupby("order_id")["vol"].sum().reindex(item_agg["order_id"]).values
    )

    # Lấy seller & product của item đầu tiên mỗi order làm đại diện (đơn giản hoá bài toán)
    first_item = items.sort_values("order_item_id").drop_duplicates("order_id", keep="first")
    first_item = first_item.merge(products, on="product_id", how="left")
    first_item = first_item.merge(cat_translation, on="product_category_name", how="left")
    first_item = first_item.merge(sellers, on="seller_id", how="left")

    df = orders.merge(customers, on="customer_id", how="left")
    df = df.merge(item_agg, on="order_id", how="left")
    df = df.merge(
        first_item[[
            "order_id", "seller_id", "seller_zip_code_prefix", "seller_city", "seller_state",
            "product_category_name_english", "product_weight_g", "product_length_cm",
            "product_height_cm", "product_width_cm"
        ]],
        on="order_id", how="left"
    )

    # Thanh toán: tổng giá trị & số kỳ trả góp (lấy max installments)
    # + payment_type CHÍNH của đơn (payment_value lớn nhất trong các record) — đặc trưng MỚI.
    pay_agg = payments.groupby("order_id").agg(
        payment_value=("payment_value", "sum"),
        payment_installments=("payment_installments", "max"),
    ).reset_index()
    df = df.merge(pay_agg, on="order_id", how="left")

    primary_payment = (
        payments.sort_values("payment_value", ascending=False)
        .drop_duplicates("order_id", keep="first")[["order_id", "payment_type"]]
    )
    df = df.merge(primary_payment, on="order_id", how="left")

    # Review score (nếu có)
    rev_agg = reviews.groupby("order_id").agg(review_score=("review_score", "mean")).reset_index()
    df = df.merge(rev_agg, on="order_id", how="left")

    # Khoảng cách seller -> customer
    cust_geo = geo_lookup.rename(columns={"lat": "cust_lat", "lng": "cust_lng"})
    df = df.merge(cust_geo, left_on="customer_zip_code_prefix", right_on="zip_code_prefix", how="left")
    df = df.drop(columns=["zip_code_prefix"])

    sell_geo = geo_lookup.rename(columns={"lat": "seller_lat", "lng": "seller_lng"})
    df = df.merge(sell_geo, left_on="seller_zip_code_prefix", right_on="zip_code_prefix", how="left")
    df = df.drop(columns=["zip_code_prefix"])

    df["distance_km"] = haversine_km(df["cust_lat"], df["cust_lng"], df["seller_lat"], df["seller_lng"])

    # Đặc trưng thời gian
    df["purchase_dow"] = df["order_purchase_timestamp"].dt.dayofweek
    df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour
    df["is_weekend"] = df["purchase_dow"].isin([5, 6]).astype(int)
    # Mùa cao điểm mua sắm ở Brazil (Black Friday cuối T11, Giáng sinh T12) — logistics thường
    # quá tải hơn, biết được ngay tại thời điểm đặt hàng (chỉ cần xem tháng mua).
    df["is_peak_season"] = df["purchase_month"].isin([11, 12]).astype(int)

    # Thời gian xử lý nội bộ (approve -> giao cho carrier), tính bằng giờ
    df["approval_to_carrier_h"] = (
        df["order_delivered_carrier_date"] - df["order_approved_at"]
    ).dt.total_seconds() / 3600

    # Thời hạn giao hàng ước tính (ngày), tính từ lúc mua
    df["estimated_days"] = (
        df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]
    ).dt.days

    # Thể tích sản phẩm
    df["product_volume_cm3"] = (
        df["product_length_cm"] * df["product_height_cm"] * df["product_width_cm"]
    )

    # Đặc trưng phái sinh MỚI — tỉ lệ và log-transform giúp model bắt được quan hệ phi tuyến
    # tốt hơn với các biến lệch phân phối mạnh (giá, phí ship, khối lượng, khoảng cách).
    df["price_per_item"] = df["total_price"] / df["n_items"].clip(lower=1)
    df["freight_to_price_ratio"] = df["total_freight"] / df["total_price"].clip(lower=0.01)
    df["log_total_price"] = np.log1p(df["total_price"])
    df["log_total_freight"] = np.log1p(df["total_freight"])
    df["log_distance_km"] = np.log1p(df["distance_km"])
    df["log_total_weight_g"] = np.log1p(df["total_weight_g"])

    # ==== TARGETS (chỉ tính được với đơn đã giao - delivered) ====
    delivered_mask = df["order_status"] == "delivered"
    df["actual_delivery_days"] = np.nan
    df.loc[delivered_mask, "actual_delivery_days"] = (
        df.loc[delivered_mask, "order_delivered_customer_date"]
        - df.loc[delivered_mask, "order_purchase_timestamp"]
    ).dt.total_seconds() / 86400

    df["is_late"] = np.nan
    df.loc[delivered_mask, "is_late"] = (
        df.loc[delivered_mask, "order_delivered_customer_date"]
        > df.loc[delivered_mask, "order_estimated_delivery_date"]
    ).astype(int)

    cust_state = df["customer_state"]
    seller_state = df["seller_state"]
    df["same_state"] = (cust_state == seller_state).astype(int)

    # ==== Đặc trưng LỊCH SỬ (as-of, không rò rỉ tương lai) ====
    # Chỉ tính trên đơn "delivered" (nơi có is_late/actual_delivery_days thật), rồi ghép lại
    # vào toàn bộ df theo order_id. Đơn chưa giao không dùng để train nên để trống là ổn.
    delivered = df.loc[delivered_mask].reset_index(drop=True)
    delivered_hist = add_history_features(delivered)
    hist_cols = [c for c in delivered_hist.columns if "_hist_" in c]
    df = df.merge(delivered_hist[["order_id"] + hist_cols], on="order_id", how="left")

    return df, delivered_hist


def save_history_lookup(delivered_hist: pd.DataFrame, out_path: Path) -> None:
    """Lưu bảng tra cứu lịch sử (toàn bộ dữ liệu delivered, tính đến thời điểm hiện tại) ra
    JSON — dùng khi dự đoán đơn MỚI (tools.py đọc file này thay vì tính lại từ đầu)."""
    import json

    tables = build_lookup_tables(delivered_hist)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(tables, f, ensure_ascii=False)


if __name__ == "__main__":
    dataset, delivered_hist = build_dataset()
    out_path = OUT_DIR / "processed_dataset.csv"
    dataset.to_csv(out_path, index=False)
    print(f"Đã xây dựng dataset: {dataset.shape[0]} dòng, {dataset.shape[1]} cột")
    print(f"Lưu tại: {out_path}")
    print("\nTỉ lệ giao trễ (is_late) trên đơn delivered:")
    print(dataset["is_late"].value_counts(normalize=True, dropna=True))

    lookup_path = MODEL_DIR / "history_lookup.json"
    lookup_path.parent.mkdir(exist_ok=True)
    save_history_lookup(delivered_hist, lookup_path)
    print(f"Đã lưu bảng tra cứu lịch sử: {lookup_path}")