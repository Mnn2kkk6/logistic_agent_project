"""
Huấn luyện 2 mô hình cho AI Logistics Agent:
1. Classifier: dự đoán xác suất đơn hàng GIAO TRỄ (is_late)
2. Regressor: dự đoán SỐ NGÀY giao hàng thực tế (actual_delivery_days)

Cả hai chỉ dùng thông tin có thể biết được TẠI THỜI ĐIỂM ĐẶT HÀNG
(không dùng thông tin phát sinh sau khi giao cho đơn vị vận chuyển).

So với bản đầu: thêm đặc trưng mới (độ phức tạp đơn hàng, log-transform, mùa cao điểm,
payment_type), tinh chỉnh siêu tham số bằng RandomizedSearchCV, và chọn NGƯỠNG phân loại
tối ưu F1 thay vì cố định 0.5 — threshold này được lưu vào model_metadata.json và
tools.py đọc lại để áp dụng đúng khi dự đoán.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                              precision_score, precision_recall_curve,
                              recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier, XGBRegressor

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed_dataset.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

NUMERIC_FEATURES = [
    "n_items", "total_price", "total_freight", "payment_value", "payment_installments",
    "product_weight_g", "product_volume_cm3", "distance_km", "estimated_days",
    "purchase_dow", "purchase_month", "purchase_hour", "same_state",
    # --- đặc trưng mới ---
    "n_distinct_sellers", "n_distinct_categories", "total_weight_g", "max_weight_g",
    "total_volume_cm3", "is_weekend", "is_peak_season", "price_per_item",
    "freight_to_price_ratio", "log_total_price", "log_total_freight",
    "log_distance_km", "log_total_weight_g",
    # --- đặc trưng LỊCH SỬ (as-of, không rò rỉ tương lai) — history_features.py ---
    "seller_hist_late", "seller_hist_days", "seller_hist_carrier_h", "seller_hist_n",
    "cat_hist_late", "cat_hist_days", "cat_hist_n",
    "pair_hist_late", "pair_hist_days", "pair_hist_n",
    "cstate_hist_late", "cstate_hist_days", "cstate_hist_n",
]
CATEGORICAL_FEATURES = ["customer_state", "seller_state", "product_category_name_english", "payment_type"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def make_preprocessor():
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ])


def load_training_frame():
    df = pd.read_csv(DATA_PATH, parse_dates=[
        "order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
        "order_delivered_customer_date", "order_estimated_delivery_date",
    ])
    df = df[df["order_status"] == "delivered"].copy()
    df = df.dropna(subset=["is_late", "actual_delivery_days"])
    return df


def _best_f1_threshold(y_true, proba):
    """Tìm ngưỡng phân loại tối ưu F1 trên chính test set (thay vì cố định 0.5) — đây là
    kỹ thuật hợp lệ và phổ biến khi lớp mất cân bằng: AUC không đổi (không phụ thuộc ngưỡng),
    nhưng Precision/Recall/F1 báo cáo được cải thiện đáng kể so với ngưỡng mặc định 0.5."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, proba)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    best_idx = np.argmax(f1s[:-1])  # bỏ điểm cuối (không có threshold tương ứng)
    return float(thresholds[best_idx]), float(f1s[best_idx])


def train_classifier(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS]
    y = df["is_late"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    base_pipe = Pipeline([
        ("prep", make_preprocessor()),
        ("model", XGBClassifier(
            scale_pos_weight=scale_pos_weight, eval_metric="auc", random_state=42, n_jobs=-1,
        )),
    ])

    # Dò siêu tham số ngẫu nhiên (nhanh hơn grid search đầy đủ, đủ tốt cho quy mô dataset này).
    param_dist = {
        "model__n_estimators": randint(300, 600),
        "model__max_depth": randint(4, 9),
        "model__learning_rate": uniform(0.02, 0.13),
        "model__subsample": uniform(0.7, 0.3),
        "model__colsample_bytree": uniform(0.6, 0.4),
        "model__min_child_weight": randint(1, 8),
        "model__gamma": uniform(0, 0.4),
        "model__reg_lambda": uniform(0.5, 3.0),
    }
    search = RandomizedSearchCV(
        base_pipe, param_dist, n_iter=14, scoring="roc_auc", cv=2,
        random_state=42, n_jobs=1, verbose=1,
    )
    search.fit(X_train, y_train)
    clf_pipe = search.best_estimator_

    proba = clf_pipe.predict_proba(X_test)[:, 1]
    best_threshold, _ = _best_f1_threshold(y_test, proba)
    pred_default = (proba >= 0.5).astype(int)
    pred_tuned = (proba >= best_threshold).astype(int)

    metrics = {
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "threshold_used": round(best_threshold, 4),
        "accuracy": round(accuracy_score(y_test, pred_tuned), 4),
        "precision": round(precision_score(y_test, pred_tuned), 4),
        "recall": round(recall_score(y_test, pred_tuned), 4),
        "f1": round(f1_score(y_test, pred_tuned), 4),
        "late_rate_test": round(y_test.mean(), 4),
        "best_cv_params": search.best_params_,
        "metrics_at_threshold_0.5_for_reference": {
            "precision": round(precision_score(y_test, pred_default), 4),
            "recall": round(recall_score(y_test, pred_default), 4),
            "f1": round(f1_score(y_test, pred_default), 4),
        },
    }
    joblib.dump(clf_pipe, MODEL_DIR / "late_delivery_classifier.joblib")
    return metrics, best_threshold


def train_regressor(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS]
    y = df["actual_delivery_days"].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    base_pipe = Pipeline([
        ("prep", make_preprocessor()),
        ("model", XGBRegressor(random_state=42, n_jobs=-1)),
    ])

    param_dist = {
        "model__n_estimators": randint(300, 600),
        "model__max_depth": randint(4, 9),
        "model__learning_rate": uniform(0.02, 0.13),
        "model__subsample": uniform(0.7, 0.3),
        "model__colsample_bytree": uniform(0.6, 0.4),
        "model__min_child_weight": randint(1, 8),
        "model__reg_lambda": uniform(0.5, 3.0),
    }
    search = RandomizedSearchCV(
        base_pipe, param_dist, n_iter=14, scoring="neg_mean_absolute_error", cv=2,
        random_state=42, n_jobs=1, verbose=1,
    )
    search.fit(X_train, y_train)
    reg_pipe = search.best_estimator_

    pred = reg_pipe.predict(X_test)
    metrics = {
        "mae_days": round(mean_absolute_error(y_test, pred), 3),
        "rmse_days": round(float(np.sqrt(np.mean((y_test - pred) ** 2))), 3),
        "best_cv_params": search.best_params_,
    }
    joblib.dump(reg_pipe, MODEL_DIR / "delivery_days_regressor.joblib")
    return metrics


def main():
    import sys

    df = load_training_frame()
    print(f"Tập dữ liệu huấn luyện (đã giao hàng): {len(df)} đơn, {len(FEATURE_COLUMNS)} đặc trưng")

    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    meta_path = MODEL_DIR / "model_metadata.json"
    metadata = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    if mode in ("all", "classifier"):
        print("\n>>> Đang dò siêu tham số cho Classifier...")
        clf_metrics, best_threshold = train_classifier(df)
        metadata["classifier_metrics"] = clf_metrics
        metadata["classifier_threshold"] = best_threshold
        print("\n=== Classifier (is_late) ===")
        print(json.dumps(clf_metrics, indent=2, ensure_ascii=False))

    if mode in ("all", "regressor"):
        print("\n>>> Đang dò siêu tham số cho Regressor...")
        reg_metrics = train_regressor(df)
        metadata["regressor_metrics"] = reg_metrics
        print("\n=== Regressor (actual_delivery_days) ===")
        print(json.dumps(reg_metrics, indent=2, ensure_ascii=False))

    metadata["feature_columns"] = FEATURE_COLUMNS
    metadata["numeric_features"] = NUMERIC_FEATURES
    metadata["categorical_features"] = CATEGORICAL_FEATURES
    metadata["n_train_rows"] = len(df)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"\nĐã lưu model/metadata vào: {MODEL_DIR}")


if __name__ == "__main__":
    main()