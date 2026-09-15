# AI Logistics Agent — Olist E-commerce Dataset

Web app local kết hợp **Machine Learning** (dự đoán rủi ro giao hàng) và **AI Agent hội
thoại đa-model** (Gemini / GPT-4o / Grok — tự đổi model khi 1 provider hết quota) để hỗ
trợ nghiệp vụ logistics cho sàn TMĐT, xây trên bộ dữ liệu **Olist Brazilian E-commerce**
(~99K đơn hàng, Kaggle).

## 1. Tính năng chính

- **Dự đoán rủi ro giao trễ + số ngày giao hàng** cho 1 đơn hàng mới (trước khi giao)
- **Chat hỏi-đáp bằng tiếng Việt** trên giao diện web, agent tự gọi đúng tool để trả lời
- **Chọn model linh hoạt** ngay trên giao diện: Gemini Flash/Flash-Lite/2.0, GPT-4o/mini,
  Grok 4 Fast/4.6 — đổi model khi 1 provider hết quota mà không mất ngữ cảnh hội thoại
- **Tự viết SQL** để trả lời các câu hỏi phân tích không có tool cứng (so sánh theo
  tháng, phương thức thanh toán, cân nặng sản phẩm...)
- **Train nhanh model tạm thời** cho target/subset khác ngoài 2 model chính (ví dụ dự
  đoán review_score, hoặc chỉ train riêng cho 1 bang)
- **Đóng gói Docker** — chạy 1 lệnh, không cần tự cài Python/thư viện

## 2. Kiến trúc

```
Trình duyệt (giao diện web, templates/index.html)
        │  chọn model qua dropdown
        ▼
Flask API (src/api.py)  ──►  src/providers.py (lớp trừu tượng đa-model)
        │                         │
        │                         ├─ Gemini (google-genai)
        │                         ├─ OpenAI (GPT-4o / GPT-4o mini)
        │                         └─ xAI/Grok (OpenAI-compatible endpoint)
        │
        ▼
   src/tools.py — các hàm nghiệp vụ mà agent có thể gọi:
     predict_new_order · get_order_info · get_state_stats · get_seller_stats
     get_category_stats · top_risky_states · top_risky_categories
     describe_dataset · query_dataset_sql (DuckDB, chỉ SELECT) · train_custom_model
        │
        ▼
   data/processed_dataset.csv  +  models/*.joblib (XGBoost)
```

**Lịch sử hội thoại lưu dạng "canonical"** (chỉ text, không phụ thuộc provider) trong
`src/providers.py`, nên đổi model giữa chừng cuộc trò chuyện không bị vỡ format — đánh
đổi là model mới sẽ không "nhớ" các bước gọi tool nội bộ ở lượt trước, chỉ nhớ nội dung
hội thoại dạng text.

## 3. Bài toán & dữ liệu

Dataset gốc gồm 9 bảng CSV (orders, order_items, payments, reviews, products, customers,
sellers, geolocation, product_category_translation), gộp lại thành
**`data/processed_dataset.csv`** (99,442 dòng, 41 cột) qua `src/data_pipeline.py`:
- Khoảng cách seller → khách hàng (haversine, dựa trên tọa độ zip code trung bình)
- Thể tích/khối lượng sản phẩm, số lượng item, tổng giá trị/phí ship/thanh toán
- Đặc trưng thời gian đặt hàng: thứ, tháng, giờ
- **Nhãn:** `is_late` (giao trễ so với ngày hẹn) và `actual_delivery_days` (số ngày giao
  thực tế) — chỉ tính được với đơn đã giao (delivered)

Toàn bộ đặc trưng dùng để huấn luyện chỉ gồm thông tin **biết được ngay tại thời điểm
đặt hàng** — mô phỏng đúng tình huống thực tế: đánh giá rủi ro *trước khi* giao hàng.

## 4. Model ML (XGBoost)

| Model | File | Mục tiêu | Kết quả (test set) |
|---|---|---|---|
| `late_delivery_classifier` | `models/late_delivery_classifier.joblib` | Xác suất giao trễ (binary) | ROC-AUC **0.788**, Recall **0.67**, F1 **0.31** |
| `delivery_days_regressor` | `models/delivery_days_regressor.joblib` | Số ngày giao hàng thực tế | MAE **4.74 ngày**, RMSE **7.58 ngày** |

Cả hai là `sklearn.Pipeline` (ColumnTransformer + XGBClassifier/XGBRegressor), huấn luyện
trên 96,470 đơn đã giao, chia 80/20 train/test. Do dữ liệu mất cân bằng (~8% đơn trễ),
classifier dùng `scale_pos_weight` để ưu tiên recall cao hơn precision. Chi tiết đầy đủ:
`models/model_metadata.json`.

## 5. Model chat hỗ trợ

| Model | Cần key | Ghi chú |
|---|---|---|
| Gemini 2.5 Flash *(mặc định)* | `GEMINI_API_KEY` | Miễn phí — [lấy tại đây](https://aistudio.google.com/apikey) |
| Gemini 2.5 Flash-Lite | `GEMINI_API_KEY` | Miễn phí, hạn mức cao hơn — dùng khi Flash hết quota |
| Gemini 2.0 Flash | `GEMINI_API_KEY` | Miễn phí, bản dự phòng thế hệ trước |
| GPT-4o mini | `OPENAI_API_KEY` | Trả phí — [platform.openai.com](https://platform.openai.com/api-keys) |
| GPT-4o | `OPENAI_API_KEY` | Trả phí, chất lượng cao nhất |
| Grok 4 Fast | `XAI_API_KEY` | Trả phí — [console.x.ai](https://console.x.ai) (cần nạp credit trước) |
| Grok 4.6 | `XAI_API_KEY` | Trả phí, model xAI mạnh nhất |

Chỉ cần cấu hình **1 key** (Gemini, miễn phí) là dùng được toàn bộ tính năng. Các key
khác là tuỳ chọn, dùng làm dự phòng khi Gemini hết quota. Model nào thiếu key sẽ hiện
`(chưa có key)` trong dropdown và báo lỗi rõ ràng nếu chọn.

## 6. Cấu trúc thư mục

```
logistics_agent/
├── data/                          # processed_dataset.csv + bảng dịch category
├── models/                        # model .joblib đã train + model_metadata.json
├── src/
│   ├── data_pipeline.py           # load, merge, feature engineering
│   ├── train_models.py            # huấn luyện classifier + regressor
│   ├── tool_schemas.py            # TOOLS (JSON Schema) + SYSTEM_PROMPT dùng chung
│   ├── tools.py                   # hàm nghiệp vụ (ML predict, thống kê, SQL, train tạm)
│   ├── providers.py                # lớp trừu tượng đa-model (Gemini/OpenAI/xAI)
│   ├── api.py                      # Flask API + serve web UI — ĐIỂM VÀO CHÍNH
│   ├── agent.py, agent_gemini.py   # bản CLI cũ (chạy trong terminal, không có web UI)
├── templates/index.html            # giao diện chat web
├── requirements.txt                 # cài cho máy local (version mở, linh hoạt theo Python)
├── requirements-docker.txt          # cài cho Docker (version PIN khớp lúc train model)
├── Dockerfile, docker-compose.yml
├── .env.example                     # mẫu khai báo API key
└── README.md
```

## 7. Chạy trên máy (không cần Docker)

```bash
pip install -r requirements.txt
```

Đặt API key (PowerShell — chỉ cần Gemini là đủ):
```powershell
$env:GEMINI_API_KEY="AIza..."
```

Chạy web app:
```bash
python -m src.api
```
Mở trình duyệt: **http://localhost:5000**

Nếu muốn xây lại dataset/model từ đầu:
```bash
python -m src.data_pipeline
python -m src.train_models
```

## 8. Chạy bằng Docker

```bash
cp .env.example .env
# mở .env, điền GEMINI_API_KEY (bắt buộc); OPENAI_API_KEY, XAI_API_KEY (tuỳ chọn)

docker compose up --build
```
Mở trình duyệt: **http://localhost:5000**

Lần sau chỉ cần `docker compose up` (không cần `--build` nếu không đổi code/dependency).

`requirements-docker.txt` pin đúng version scikit-learn/xgboost khớp với lúc train model
— tránh cảnh báo `InconsistentVersionWarning` hay gặp khi máy local có version khác.

## 9. Các endpoint API

```
GET  /                              Giao diện chat web
GET  /health
GET  /models                        Danh sách model + trạng thái đã cấu hình key
POST /predict                       {distance_km, customer_state, seller_state, ...}
GET  /order/<order_id>
GET  /stats/state/<state>
GET  /stats/seller/<seller_id>
GET  /stats/category/<category>
GET  /stats/top-risky-states?n=5
GET  /stats/top-risky-categories?n=5
POST /chat                          {"message": "...", "model": "gemini-2.5-flash"}
POST /chat/reset                    Xoá lịch sử hội thoại của session hiện tại
```

Ví dụ:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"distance_km": 1200, "customer_state": "BA", "seller_state": "SP"}'
```

## 10. Hướng phát triển tiếp

- Cân bằng lại precision/recall bằng threshold tuning hoặc cost-sensitive learning
- Thêm tool `recommend_seller` (chọn seller có lịch sử giao đúng hẹn tốt nhất theo khu vực)
- Lưu lịch sử hội thoại vào DB thay vì RAM (để chạy được nhiều người dùng đồng thời)
- Trực quan hoá bản đồ rủi ro theo bang (folium/plotly)
- Loại bỏ dependency thừa `nvidia-nccl-cu13` (~250MB, không cần thiết cho project CPU-only)
  để giảm size Docker image

---
*Dataset: [Brazilian E-Commerce Public Dataset by Olist (Kaggle)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)*
