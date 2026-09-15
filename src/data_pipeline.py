"""
Data pipeline cho AI Logistics Agent (Olist E-commerce Dataset)
Load -> Merge -> Feature Engineering -> Xuất dataset đã xử lý.
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "data"


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
    item_agg = items.groupby("order_id").agg(
        n_items=("order_item_id", "count"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
    ).reset_index()

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
    pay_agg = payments.groupby("order_id").agg(
        payment_value=("payment_value", "sum"),
        payment_installments=("payment_installments", "max"),
    ).reset_index()
    df = df.merge(pay_agg, on="order_id", how="left")

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

    return df


if __name__ == "__main__":
    dataset = build_dataset()
    out_path = OUT_DIR / "processed_dataset.csv"
    dataset.to_csv(out_path, index=False)
    print(f"Đã xây dựng dataset: {dataset.shape[0]} dòng, {dataset.shape[1]} cột")
    print(f"Lưu tại: {out_path}")
    print("\nTỉ lệ giao trễ (is_late) trên đơn delivered:")
    print(dataset["is_late"].value_counts(normalize=True, dropna=True))
