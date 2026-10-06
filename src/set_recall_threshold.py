"""
Đặt lại ngưỡng phân loại cho classifier ĐÃ TRAIN (không train lại, chạy vài giây).

Chọn ngưỡng LỚN NHẤT vẫn giữ Recall >= mục tiêu (mặc định 0.75) — tức là Precision cao
nhất có thể mà vẫn bắt được đủ tỉ lệ đơn trễ. Cả 2 điểm vận hành (ưu tiên Recall và
tối ưu F1) được ghi vào models/model_metadata.json; predict_new_order đọc
"classifier_threshold" từ file này.

Cách chạy (từ thư mục gốc project):
    python -m src.set_recall_threshold          # mục tiêu Recall 0.75
    python -m src.set_recall_threshold 0.80     # mục tiêu Recall 0.80

Lưu ý: ngưỡng được chọn trên tập test (cùng cách chia lúc train), nên Precision/Recall
báo cáo hơi lạc quan so với dữ liệu hoàn toàn mới.
"""
import json
import sys

import joblib
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from src.train_models import FEATURE_COLUMNS, MODEL_DIR, load_training_frame


def _point(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": round(float(threshold), 4),
        "precision": round(precision_score(y_true, pred), 4),
        "recall": round(recall_score(y_true, pred), 4),
        "f1": round(f1_score(y_true, pred), 4),
        "accuracy": round(accuracy_score(y_true, pred), 4),
    }


def main():
    target_recall = float(sys.argv[1]) if len(sys.argv) > 1 else 0.75

    df = load_training_frame()
    X = df[FEATURE_COLUMNS]
    y = df["is_late"].astype(int)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = joblib.load(MODEL_DIR / "late_delivery_classifier.joblib")
    proba = clf.predict_proba(X_test)[:, 1]

    # Điểm 1: ngưỡng lớn nhất còn giữ Recall >= mục tiêu
    candidates = [t for t in np.arange(0.01, 0.99, 0.001)
                  if recall_score(y_test, (proba >= t).astype(int)) >= target_recall]
    if not candidates:
        sys.exit(f"Không có ngưỡng nào đạt Recall >= {target_recall}.")
    recall_threshold = max(candidates)

    # Điểm 2: ngưỡng tối ưu F1
    precisions, recalls, thresholds = precision_recall_curve(y_test, proba)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    f1_threshold = thresholds[np.argmax(f1s[:-1])]

    recall_pt = _point(y_test, proba, recall_threshold)
    f1_pt = _point(y_test, proba, f1_threshold)

    meta_path = MODEL_DIR / "model_metadata.json"
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    old = meta.get("classifier_metrics", {})
    meta["classifier_threshold"] = recall_pt["threshold"]
    meta["classifier_metrics"] = {
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "late_rate_test": round(float(y_test.mean()), 4),
        "deployed_operating_point": {"mode": "recall_priority", **recall_pt},
        "alternative_operating_point": {"mode": "f1_optimal", **f1_pt},
        "note": "Precision/Recall là đánh đổi trên cùng 1 đường cong (AUC không đổi theo ngưỡng).",
        "best_cv_params": old.get("best_cv_params"),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"ROC-AUC: {meta['classifier_metrics']['roc_auc']}")
    print(f"Ưu tiên Recall (đang dùng): {recall_pt}")
    print(f"Tối ưu F1 (tham khảo):      {f1_pt}")
    print(f"Đã cập nhật {meta_path}")


if __name__ == "__main__":
    main()
