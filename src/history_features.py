"""
Đặc trưng LỊCH SỬ "as-of": với mỗi đơn hàng, chỉ dùng các đơn ĐÃ GIAO XONG trước thời điểm
đặt đơn đó (order_delivered_customer_date < order_purchase_timestamp). Nhờ vậy không rò rỉ
thông tin tương lai — tại thời điểm đặt hàng thật, hệ thống cũng biết đúng những gì này.

Với mỗi nhóm (seller, ngành hàng, cặp bang seller→khách, bang khách) tính:
  {nhóm}_hist_late : tỉ lệ trễ lịch sử (làm mượt về mức toàn cục khi ít dữ liệu)
  {nhóm}_hist_days : số ngày giao TB lịch sử
  {nhóm}_hist_n    : số đơn đã có kết quả (độ tin cậy của 2 chỉ số trên)
Riêng seller có thêm seller_hist_carrier_h: giờ TB từ duyệt đơn đến giao cho vận chuyển.
"""
import numpy as np
import pandas as pd

SMOOTH_M = 20  # "số đơn ảo" kéo tỉ lệ về mức toàn cục khi nhóm có ít đơn

GROUPS = {
    "seller": ["seller_id"],
    "cat": ["product_category_name_english"],
    "pair": ["seller_state", "customer_state"],
    "cstate": ["customer_state"],
}


def _values(df):
    return {
        "late": df["is_late"].to_numpy(dtype=float),
        "days": df["actual_delivery_days"].to_numpy(dtype=float),
        "carrier_h": df["approval_to_carrier_h"].to_numpy(dtype=float),
    }


def _priors(df):
    return {
        "late": float(df["is_late"].mean()),
        "days": float(df["actual_delivery_days"].mean()),
        "carrier_h": float(df["approval_to_carrier_h"].mean()),
    }


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """df: chỉ gồm đơn 'delivered' đã có is_late / actual_delivery_days / ngày giao."""
    df = df.reset_index(drop=True).copy()
    t_buy = df["order_purchase_timestamp"].to_numpy(dtype="datetime64[ns]").astype("int64")
    t_done = df["order_delivered_customer_date"].to_numpy(dtype="datetime64[ns]").astype("int64")
    vals, priors = _values(df), _priors(df)

    for gname, keys in GROUPS.items():
        metrics = ["late", "days"] + (["carrier_h"] if gname == "seller" else [])
        out = {m: np.full(len(df), priors[m]) for m in metrics}
        cnt = np.zeros(len(df))
        for _, idx in df.groupby(keys, sort=False).indices.items():
            idx = np.asarray(idx)
            order = np.argsort(t_done[idx], kind="stable")
            done_sorted = t_done[idx][order]
            pos = np.searchsorted(done_sorted, t_buy[idx], side="left")  # số đơn đã giao xong trước lúc mua
            cnt[idx] = pos
            for m in metrics:
                v = vals[m][idx][order]
                ok = ~np.isnan(v)
                csum = np.concatenate([[0.0], np.cumsum(np.where(ok, v, 0.0))])[pos]
                ccnt = np.concatenate([[0.0], np.cumsum(ok)])[pos]
                out[m][idx] = (csum + priors[m] * SMOOTH_M) / (ccnt + SMOOTH_M)
        for m in metrics:
            df[f"{gname}_hist_{'late' if m == 'late' else ('days' if m == 'days' else 'carrier_h')}"] = out[m]
        df[f"{gname}_hist_n"] = cnt
    return df


def build_lookup_tables(df: pd.DataFrame) -> dict:
    """Bảng tra cứu 'trạng thái mới nhất' (toàn bộ lịch sử) dùng khi dự đoán đơn MỚI."""
    priors = _priors(df)
    tables = {"priors": priors, "smooth_m": SMOOTH_M}
    for gname, keys in GROUPS.items():
        g = df.groupby(keys)
        agg = g.agg(n=("is_late", "size"), late=("is_late", "sum"), days=("actual_delivery_days", "sum"))
        rec = {}
        for key, row in agg.iterrows():
            k = "|".join(map(str, key)) if isinstance(key, tuple) else str(key)
            rec[k] = {
                "late": (row["late"] + priors["late"] * SMOOTH_M) / (row["n"] + SMOOTH_M),
                "days": (row["days"] + priors["days"] * SMOOTH_M) / (row["n"] + SMOOTH_M),
                "n": int(row["n"]),
            }
        if gname == "seller":
            ch = g["approval_to_carrier_h"].agg(["sum", "count"])
            for key, row in ch.iterrows():
                rec[str(key)]["carrier_h"] = (row["sum"] + priors["carrier_h"] * SMOOTH_M) / (row["count"] + SMOOTH_M)
        tables[gname] = rec
    return tables
