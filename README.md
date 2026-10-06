# AI Logistics Agent — Olist E-commerce Dataset

Dự án xây dựng một **AI Agent hỗ trợ nghiệp vụ logistics** cho sàn TMĐT, chạy như một
**web app local**, kết hợp:
1. **Mô hình Machine Learning** dự đoán rủi ro giao hàng (huấn luyện trên dữ liệu thật —
   bộ Olist Brazilian E-commerce, ~99K đơn hàng, từ Kaggle).
2. **Agent hội thoại dùng Gemini API (Google GenAI, function calling)** — người dùng hỏi
   bằng ngôn ngữ tự nhiên trên giao diện web, agent tự gọi đúng tool (model ML hoặc truy
   vấn thống kê) để trả lời.

## 1. Bài toán & dữ liệu

Dataset gốc gồm 9 bảng CSV (orders, order_items, payments, reviews, products, customers,
sellers, geolocation, product_category_translation). Sau khi gộp và xử lý, còn lại
**`processed_dataset.csv`** (99,442 dòng, 41 cột) — 1 bảng đặc trưng ở mức đơn hàng, gồm:
- Khoảng cách seller → khách hàng (haversine, dựa trên tọa độ zip code trung bình):
  `distance_km`, `cust_lat/lng`, `seller_lat/lng`
- Thể tích/khối lượng sản phẩm, số lượng item, tổng giá trị/phí ship/thanh toán
- Đặc trưng thời gian đặt hàng: `purchase_dow`, `purchase_month`, `purchase_hour`
- **Nhãn:**
  - `is_late`: đơn giao trễ so với ngày hẹn (`order_delivered_customer_date > order_estimated_delivery_date`)
  - `actual_delivery_days`: số ngày từ lúc mua đến lúc khách nhận hàng

Toàn bộ đặc trưng dùng để huấn luyện chỉ gồm thông tin **biết được ngay tại thời điểm
đặt hàng** — mô phỏng đúng tình huống thực tế: đánh giá rủi ro *trước khi* giao hàng.

`processed_dataset.parquet` là bản song song của cùng dataset (đọc nhanh hơn CSV).
`product_category_name_translation.csv` là bảng dịch tên category tiếng Bồ Đào Nha → tiếng Anh,
dùng trong bước xử lý dữ liệu gốc.

## 2. Mô hình ML (XGBoost)

| Mô hình | File | Mục tiêu | Kết quả (test set) |
|---|---|---|---|
| `late_delivery_classifier` | `late_delivery_classifier.joblib` | Xác suất giao trễ (binary) | ROC-AUC **0.788**, Accuracy **0.762**, Precision **0.20**, Recall **0.67**, F1 **0.31** |
| `delivery_days_regressor` | `delivery_days_regressor.joblib` | Số ngày giao hàng thực tế | MAE **4.74 ngày**, RMSE **7.58 ngày** |

Cả hai đều là `sklearn.Pipeline` (ColumnTransformer tiền xử lý + `XGBClassifier`/`XGBRegressor`),
huấn luyện trên 96,470 đơn đã giao (delivered), chia 80/20 train/test. Do dữ liệu mất cân bằng
(chỉ ~8% đơn trễ), classifier dùng `scale_pos_weight` để ưu tiên recall cao hơn precision —
tức ưu tiên "bắt" được đơn có nguy cơ trễ, chấp nhận báo động giả nhiều hơn. Chi tiết đầy đủ về
feature list và metrics nằm trong `model_metadata.json`.

## 3. Kiến trúc Agent (mục tiêu)

```
Người dùng (câu hỏi tiếng Việt tự nhiên, trên web UI local)
        │
        ▼
   Gemini API (Google GenAI SDK, function calling) ──► chọn tool phù hợp
        │
        ▼
   tools.py: predict_new_order / get_order_info /
             get_state_stats / get_seller_stats /
             get_category_stats / top_risky_states /
             top_risky_categories
        │
        ▼
   Trả kết quả (JSON) về cho Gemini → tổng hợp câu trả lời tiếng Việt → hiển thị trên web
```

Backend dự kiến dùng Flask để vừa serve web UI vừa expose các tool qua HTTP API.

## 4. Trạng thái hiện tại của repo

Hiện repo đang có:

```
.
├── processed_dataset.csv                    # dataset đã xử lý (99,442 dòng)
├── processed_dataset.parquet                 # bản parquet của cùng dataset
├── product_category_name_translation.csv     # bảng dịch category PT → EN
├── late_delivery_classifier.joblib            # model đã train
├── delivery_days_regressor.joblib             # model đã train
├── model_metadata.json                        # feature list + metrics đầy đủ
├── test_model.py                               # script test nhanh: load classifier
├── test_tools.py                               # script test nhanh: gọi tools.predict_new_order
├── requirements.txt
├── README.md
└── README_DATA.md
```

**Chưa có trong repo (cần triển khai tiếp):** `src/data_pipeline.py`, `src/train_models.py`,
`src/tools.py`, `src/agent.py`, `src/api.py` — tức toàn bộ phần code xử lý dữ liệu, training,
business-logic tools, agent Gemini và web/API server. `test_model.py` và `test_tools.py` đang
import từ các module này (`models/late_delivery_classifier.joblib`, `from src import tools`)
nên sẽ chưa chạy được cho đến khi các file trên được viết.

## 5. Cách chạy (khi đã có đủ source code)

```bash
pip install -r requirements.txt

# 1) Xây dataset đã xử lý (nếu chạy lại từ dữ liệu gốc — xem README_DATA.md)
python3 -m src.data_pipeline

# 2) Huấn luyện model
python3 -m src.train_models

# 3) Chạy web app (Flask) — agent dùng Gemini API
export GEMINI_API_KEY="..."
python3 -m src.api   # http://localhost:5000
```

### Ví dụ gọi API

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"distance_km": 1200, "customer_state": "BA", "seller_state": "SP"}'

curl http://localhost:5000/stats/top-risky-states?n=5
```

### Ví dụ hội thoại với agent (qua web UI)

```
Bạn: Đơn hàng từ seller ở SP giao tới khách ở AL, xa khoảng 2500km, có rủi ro trễ không?
Agent: [gọi tool predict_new_order] → Xác suất trễ ~XX%, dự kiến giao trong ~YY ngày.
       Khuyến nghị: bang AL vốn có tỉ lệ trễ lịch sử cao, nên cân nhắc chọn seller gần
       khách hơn hoặc cảnh báo thời gian giao dài hơn cho khách.
```

## 6. Hướng phát triển tiếp

- Viết `src/tools.py`, `src/agent.py` (tích hợp Gemini function calling), `src/api.py` (Flask)
  và giao diện web local để hoàn thiện luồng agent mô tả ở mục 3.
- Cân bằng lại precision/recall bằng threshold tuning hoặc cost-sensitive learning theo
  chi phí thực tế của việc giao trễ vs. cảnh báo nhầm.
- Thêm tool `recommend_seller` (chọn seller có lịch sử giao đúng hẹn tốt nhất theo khu vực).
- Thêm bộ nhớ hội thoại dài hạn / lưu log dự đoán để re-train định kỳ (giám sát model drift).
- Trực quan hoá bản đồ rủi ro theo bang (folium/plotly) cho phần dashboard web.

---
*Dataset: [Brazilian E-Commerce Public Dataset by Olist (Kaggle)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)*