"""Danh mục dataset được hiển thị cho người dùng trong AI Logistics Agent.

Giai đoạn hiện tại chỉ đăng ký Olist. Khi bổ sung dữ liệu mới, thêm metadata vào
DATASETS; mỗi dataset mới cần adapter/tool tương ứng trước khi bật status="available".
"""
from __future__ import annotations

from typing import Any

DEFAULT_DATASET_ID = "olist"

DATASETS: dict[str, dict[str, Any]] = {
    "olist": {
        "id": "olist",
        "label": "Olist E-commerce (Brazil)",
        "description": (
            "Dữ liệu đơn hàng thương mại điện tử Brazil: giao hàng, seller, "
            "ngành hàng, rủi ro giao trễ và bản đồ luồng đơn hàng."
        ),
        "region": "Brazil",
        "status": "available",
        "capabilities": [
            "order_lookup",
            "delivery_statistics",
            "late_delivery_prediction",
            "delivery_time_prediction",
            "sql_analytics",
            "flow_map",
        ],
    },
}


def list_datasets() -> list[dict[str, Any]]:
    """Trả về metadata an toàn để frontend hiển thị, không lộ cấu hình nội bộ."""
    return [
        {
            "id": item["id"],
            "label": item["label"],
            "description": item["description"],
            "region": item.get("region"),
            "status": item["status"],
            "capabilities": list(item.get("capabilities", [])),
        }
        for item in DATASETS.values()
        if item.get("status") == "available"
    ]


def get_dataset(dataset_id: str | None) -> dict[str, Any] | None:
    """Lấy dataset đang được hỗ trợ; trả None nếu ID không hợp lệ/không khả dụng."""
    if not isinstance(dataset_id, str):
        return None
    item = DATASETS.get(dataset_id.strip())
    if not item or item.get("status") != "available":
        return None
    return item
