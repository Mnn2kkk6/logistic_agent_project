"""
Huấn luyện 2 mô hình cho AI Logistics Agent:
1. Classifier: dự đoán xác suất đơn hàng GIAO TRỄ (is_late)
2. Regressor: dự đoán SỐ NGÀY giao hàng thực tế (actual_delivery_days)

Cả hai chỉ dùng thông tin có thể biết được TẠI THỜI ĐIỂM ĐẶT HÀNG
(không dùng thông tin phát sinh sau khi giao cho đơn vị vận chuyển).
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                              precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
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
]
CATEGORICAL_FEATURES = ["customer_state", "seller_state", "product_category_name_english"]
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


def train_classifier(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS]
    y = df["is_late"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    clf_pipe = Pipeline([
        ("prep", make_preprocessor()),
        ("model", XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight,
            eval_metric="auc", random_state=42, n_jobs=-1,
        )),
    ])
    clf_pipe.fit(X_train, y_train)

    proba = clf_pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "late_rate_test": round(y_test.mean(), 4),
    }
    joblib.dump(clf_pipe, MODEL_DIR / "late_delivery_classifier.joblib")
    return metrics


def train_regressor(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS]
    y = df["actual_delivery_days"].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    reg_pipe = Pipeline([
        ("prep", make_preprocessor()),
        ("model", XGBRegressor(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            random_state=42, n_jobs=-1,
        )),
    ])
    reg_pipe.fit(X_train, y_train)
    pred = reg_pipe.predict(X_test)
    metrics = {
        "mae_days": round(mean_absolute_error(y_test, pred), 3),
        "rmse_days": round(float(np.sqrt(np.mean((y_test - pred) ** 2))), 3),
    }
    joblib.dump(reg_pipe, MODEL_DIR / "delivery_days_regressor.joblib")
    return metrics


def main():
    df = load_training_frame()
    print(f"Tập dữ liệu huấn luyện (đã giao hàng): {len(df)} đơn")

    clf_metrics = train_classifier(df)
    reg_metrics = train_regressor(df)

    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "classifier_metrics": clf_metrics,
        "regressor_metrics": reg_metrics,
        "n_train_rows": len(df),
    }
    with open(MODEL_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\n=== Classifier (is_late) ===")
    print(json.dumps(clf_metrics, indent=2, ensure_ascii=False))
    print("\n=== Regressor (actual_delivery_days) ===")
    print(json.dumps(reg_metrics, indent=2, ensure_ascii=False))
    print(f"\nĐã lưu model vào: {MODEL_DIR}")


if __name__ == "__main__":
    main()
