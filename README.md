# AI Logistics Agent — Olist E-commerce

AI Logistics Agent là một web application hỗ trợ phân tích logistics cho sàn thương mại điện tử. Dự án kết hợp **Machine Learning, LLM Function Calling, SQL analytics và bản đồ luồng đơn hàng** để trả lời câu hỏi nghiệp vụ từ dữ liệu Olist Brazilian E-Commerce.

> **Dataset:** [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

**English documentation:** [READMEENG.md](READMEENG.md)

## 1. Điểm nổi bật

- **Late-delivery prediction:** dự đoán xác suất giao hàng trễ cho một đơn hàng mới.
- **Delivery-time prediction:** ước tính số ngày giao hàng thực tế.
- **AI Agent:** người dùng hỏi bằng ngôn ngữ tự nhiên; LLM tự chọn tool phù hợp để lấy dữ liệu hoặc chạy phân tích.
- **Multi-provider LLM:** hỗ trợ Gemini, OpenAI, xAI/Grok, Groq và local rule-based fallback.
- **SQL analytics với DuckDB:** cho phép thực hiện các truy vấn `SELECT` / `WITH ... SELECT` trên dataset.
- **Order lookup & operational statistics:** tra cứu đơn hàng, seller, bang khách hàng, ngành hàng và các nhóm rủi ro.
- **Interactive flow map:** bản đồ luồng đơn hàng theo cặp `seller state → customer state`, có drill-down tới các đơn hàng mẫu và animation theo thời gian giao hàng.
- **Optional Tavily integration:** có utility tìm kiếm web cho thông tin nằm ngoài phạm vi dữ liệu Olist.
- **Dockerized deployment:** Flask app chạy trong container với healthcheck và multi-stage build.

## 2. Bài toán và dữ liệu

Dự án sử dụng bộ Olist Brazilian E-Commerce gồm **9 bảng CSV gốc**, bao gồm orders, order items, payments, reviews, products, customers, sellers, geolocation và category translation.

Pipeline chính:

~~~text
Raw Olist CSV
     |
     v
Load + Merge + Feature Engineering
     |
     +--> History features (seller/category/state-pair/customer-state)
     |
     v
processed_dataset.csv
     |
     +-------------------+
     |                   |
     v                   v
ML Training          DuckDB Analytics
     |                   |
     v                   v
2 Production Models   Agent Tools / SQL
     |                   |
     +---------+---------+
               |
               v
        Flask Web Application
          |             |
          v             v
       Chat UI       Flow Map
~~~

### Các đặc trưng chính

Các feature được thiết kế để mô phỏng quyết định **ngay tại thời điểm đặt hàng**, tránh dùng thông tin chỉ có sau khi đơn đã được giao.

Ví dụ:

- Khoảng cách seller → customer bằng Haversine.
- Giá sản phẩm, phí vận chuyển, payment value, installments.
- Số item, số seller/category khác nhau trong đơn.
- Tổng khối lượng, thể tích, giá/item và các log-transformed features.
- Bang seller, bang customer, category và payment type.
- Thứ, tháng, giờ đặt hàng, weekend và peak season.
- Historical features theo seller, category, state pair và customer state.

**Targets:**
- `is_late`: đơn có giao sau ngày dự kiến hay không.
- `actual_delivery_days`: thời gian giao thực tế tính từ lúc mua đến lúc khách nhận hàng.

## 3. Machine Learning

Hai model production được huấn luyện bằng **XGBoost** trong `sklearn.Pipeline`, kết hợp tiền xử lý numeric/categorical features.

| Model | Mục tiêu | Kết quả test |
|---|---|---|
| XGBClassifier | `is_late` | ROC-AUC **0.814** |
| XGBRegressor | `actual_delivery_days` | MAE **4.449 ngày**, RMSE **7.28 ngày** |

### Classifier: hai operating points

Do dữ liệu mất cân bằng (late-rate test khoảng **8.11%**), threshold có thể được điều chỉnh theo mục tiêu nghiệp vụ:

| Chế độ | Threshold | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Recall-priority (deployed) | **0.38** | 0.193 | **0.750** | 0.307 |
| F1-optimal | **0.6493** | 0.335 | 0.488 | **0.397** |

Cách tiếp cận này cho phép đánh đổi giữa **bắt được nhiều đơn có nguy cơ trễ** và **giảm số cảnh báo nhầm** thay vì cố định threshold = 0.5.

Metadata của model được lưu tại:

~~~text
models/model_metadata.json
~~~

## 4. AI Agent

Luồng hội thoại:

~~~text
User
  |
  v
Flask Chat API
  |
  v
Selected LLM Provider
  |
  +--> Function Calling
  |       |
  |       +--> get_order_info
  |       +--> predict_new_order
  |       +--> get_state_stats
  |       +--> get_seller_stats
  |       +--> get_category_stats
  |       +--> top_risky_states
  |       +--> top_risky_categories
  |       +--> describe_dataset
  |       +--> query_dataset_sql
  |       +--> train_custom_model
  |
  v
Tool result
  |
  v
LLM final answer
~~~

### Tool groups

**Prediction**
- `predict_new_order`

Dùng các thông tin có thể biết khi đặt hàng để trả về rủi ro giao trễ và số ngày giao dự kiến. Khi có `seller_id`, agent có thể bổ sung historical performance của seller.

**Lookup / statistics**
- `get_order_info`
- `get_state_stats`
- `get_seller_stats`
- `get_category_stats`
- `top_risky_states`
- `top_risky_categories`

**General analytics**
- `describe_dataset`
- `query_dataset_sql`
- `train_custom_model`

`query_dataset_sql` chỉ cho phép các truy vấn đọc dữ liệu và có thể truy cập bảng `orders` đã tổng hợp cùng các bảng raw cần thiết cho các bài toán join chi tiết.

## 5. Multi-provider LLM

Người dùng có thể chọn model trực tiếp trên web UI:

| Provider | Models in current project |
|---|---|
| Google Gemini | Gemini 2.5 Flash, 2.5 Flash-Lite, 2.0 Flash |
| OpenAI | GPT-4o Mini, GPT-4o |
| xAI | Grok 4 Fast, Grok 4.6 |
| Groq | GPT-OSS 120B, GPT-OSS 20B |
| Local | Rule-based offline fallback |

Lịch sử hội thoại được lưu theo dạng **canonical chat history**, không phụ thuộc provider. Nhờ đó có thể đổi model giữa các lượt chat mà không phải lưu format tool-call riêng của từng API.

## 6. Interactive Logistics Flow Map

Mở:

~~~text
http://localhost:5000/map
~~~

Bản đồ dùng **Leaflet + OpenStreetMap tiles**.

### Màn hình tổng thể

- Gộp đơn hàng theo cặp `seller_state → customer_state`.
- Độ dày luồng thể hiện quy mô đơn hàng.
- Màu sắc phản ánh late rate.
- Hover để xem nhanh số đơn và tỉ lệ trễ.
- Click vào một luồng để drill-down.

### Màn hình chi tiết

Sau khi click vào một luồng:

- Hiển thị các đơn hàng mẫu của cặp bang đó.
- Đường màu xanh biểu diễn đơn đúng hẹn; đỏ biểu diễn đơn trễ.
- Chấm chạy từ seller đến customer để mô phỏng luồng giao hàng.
- Tốc độ animation được chuẩn hóa theo thời gian giao thực tế của từng đơn.
- Click vào đơn để xem order id, seller/customer, khoảng cách, category, giá trị, ngày giao và review score.

> **Lưu ý:** vị trí trên bản đồ là **tọa độ trung bình theo khu vực/zip code**, dùng để minh họa luồng logistics. Đây không phải GPS thời gian thực của đơn hàng.

### Map endpoints

~~~text
GET /map
GET /api/map/flows
GET /api/map/sample?from=SP&to=RJ&limit=40
~~~

## 7. Tavily Web Search

Project có utility `web_search` sử dụng **Tavily** để lấy thông tin từ web cho các câu hỏi nằm ngoài dataset Olist, ví dụ kiến thức logistics hoặc thông tin bên ngoài dữ liệu nội bộ.

API key được cấu hình bằng:

~~~text
TAVILY_API_KEY=
~~~

Trong production, nên ưu tiên dữ liệu nội bộ trước khi tìm kiếm web vì dữ liệu Olist có tính nhất quán với các tool analytics của project.

## 8. Web API

Các endpoint chính:

~~~text
GET  /                         Web chat UI
GET  /health                   Healthcheck
GET  /models                   Available LLM models + key status

POST /chat                     Chat with selected model
POST /chat/reset               Reset current conversation

POST /predict                  Predict a new order
GET  /order/<order_id>         Lookup an order

GET  /stats/state/<state>
GET  /stats/seller/<seller_id>
GET  /stats/category/<category>
GET  /stats/top-risky-states?n=5
GET  /stats/top-risky-categories?n=5

GET  /map
GET  /api/map/flows
GET  /api/map/sample?from=SP&to=RJ&limit=40
~~~

## 9. Project structure

~~~text
.
├── data/
│   ├── olist_*_dataset.csv
│   ├── product_category_name_translation.csv
│   └── processed_dataset.csv
│
├── models/
│   ├── late_delivery_classifier.joblib
│   ├── delivery_days_regressor.joblib
│   ├── history_lookup.json
│   └── model_metadata.json
│
├── src/
│   ├── api.py
│   ├── agent.py
│   ├── agent_gemini.py
│   ├── providers.py
│   ├── tool_schemas.py
│   ├── tools.py
│   ├── data_pipeline.py
│   ├── history_features.py
│   ├── train_models.py
│   └── set_recall_threshold.py
│
├── templates/
│   ├── index.html
│   └── map.html
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-docker.txt
├── .env.example
├── README.md
└── READMEENG.md
~~~

## 10. Chạy local

### 10.1. Cài dependencies

~~~bash
pip install -r requirements.txt
~~~

### 10.2. Cấu hình environment

Copy:

~~~text
.env.example
~~~

thành:

~~~text
.env
~~~

Sau đó điền API key cần dùng.

> Không commit `.env` thật lên GitHub.

### 10.3. Chạy Flask

PowerShell:

~~~powershell
$env:GEMINI_API_KEY="your-key"
python -m src.api
~~~

Mở:

~~~text
http://localhost:5000
~~~

Bản đồ:

~~~text
http://localhost:5000/map
~~~

### 10.4. Chạy data pipeline / training

Tạo lại processed dataset:

~~~bash
python -m src.data_pipeline
~~~

Train classifier + regressor:

~~~bash
python -m src.train_models
~~~

Có thể train riêng:

~~~bash
python -m src.train_models classifier
python -m src.train_models regressor
~~~

## 11. Chạy bằng Docker

Build và start:

~~~bash
docker compose up --build
~~~

Kiểm tra:

~~~bash
docker compose ps
~~~

Healthcheck:

~~~text
http://localhost:5000/health
~~~

Ứng dụng:

~~~text
http://localhost:5000
~~~

Bản đồ:

~~~text
http://localhost:5000/map
~~~

Docker sử dụng multi-stage build: dependencies được cài ở builder stage, sau đó runtime image chỉ giữ môi trường cần thiết để chạy app.

Dataset và model được mount read-only trong Compose để có thể thay thế data/model mà không cần rebuild image.

## 12. Một vài câu hỏi mẫu

~~~text
Top 5 bang có tỉ lệ giao hàng trễ cao nhất?

Đơn hàng từ seller ở SP giao đến khách ở RJ cách khoảng 900km có rủi ro trễ không?

Seller này có hiệu suất giao hàng tốt không?

So sánh tỉ lệ giao trễ theo tháng.

Ngành hàng nào có thời gian giao hàng cao nhất?

Cho tôi xem luồng đơn hàng từ SP đến RJ trên bản đồ.
~~~

## 13. Hạn chế và hướng phát triển

- Dữ liệu Olist là dữ liệu lịch sử; bản đồ chỉ mang tính trực quan hóa.
- Chat history hiện lưu trong RAM của Flask process, phù hợp cho local/demo; triển khai nhiều instance nên chuyển sang Redis hoặc database.
- Các provider cần API key riêng; local fallback chỉ hỗ trợ một số mẫu câu cố định.
- Có thể mở rộng thêm seller recommendation, route-aware prediction, model monitoring, drift detection và long-term conversation memory.
- Có thể cải thiện classifier theo chi phí nghiệp vụ thực tế bằng threshold tuning hoặc cost-sensitive learning.

---

**Stack:** Python, Pandas, scikit-learn, XGBoost, DuckDB, Flask, Google GenAI, OpenAI API, xAI API, Groq, Tavily, Leaflet, Docker