"""Dataset registry. DataCo is enabled only after its CSV is pushed."""
from __future__ import annotations
from pathlib import Path
from typing import Any
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_ID = "olist"
DATASETS: dict[str, dict[str, Any]] = {
 "olist": {"id":"olist","label":"Olist E-commerce (Brazil)","description":"Dữ liệu đơn hàng thương mại điện tử Brazil: giao hàng, seller, ngành hàng, rủi ro giao trễ và bản đồ luồng đơn hàng.","region":"Brazil","status":"available","capabilities":["order_lookup","delivery_statistics","late_delivery_prediction","delivery_time_prediction","sql_analytics","flow_map"]},
 "dataco": {"id":"dataco","label":"DataCo Supply Chain (Global)","description":"Dữ liệu chuỗi cung ứng quốc tế: thời gian vận chuyển thực tế/dự kiến, rủi ro giao trễ, quốc gia, khu vực, ngành hàng và phương thức giao hàng. Bản đồ luồng và dự đoán ML riêng chưa được bật.","region":"Global","status":"available","requires_file":BASE_DIR / "data_2" / "DataCoSupplyChainDataset.csv","capabilities":["order_lookup","delivery_statistics","late_delivery_risk_analysis","sql_analytics"]},
}
def _is_available(item: dict[str, Any]) -> bool:
 required=item.get("requires_file")
 return item.get("status")=="available" and (required is None or Path(required).is_file())
def list_datasets() -> list[dict[str, Any]]:
 return [{"id":i["id"],"label":i["label"],"description":i["description"],"region":i.get("region"),"status":i["status"],"capabilities":list(i.get("capabilities",[]))} for i in DATASETS.values() if _is_available(i)]
def get_dataset(dataset_id: str | None) -> dict[str, Any] | None:
 if not isinstance(dataset_id,str): return None
 item=DATASETS.get(dataset_id.strip())
 return item if item and _is_available(item) else None
