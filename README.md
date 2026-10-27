# AI Logistics Agent — Dự đoán rủi ro giao hàng & Phân tích chuỗi cung ứng

[English documentation](READMEENG.md) · [GitHub repository](https://github.com/Mnn2kkk6/logistic_agent_project)

AI Logistics Agent là ứng dụng web phân tích nghiệp vụ logistics, kết hợp **Machine Learning, AI Agent với tool/function calling, SQL analytics và bản đồ luồng đơn hàng**. Người dùng đặt câu hỏi bằng ngôn ngữ tự nhiên, chọn bộ dữ liệu cần phân tích; hệ thống tự định tuyến model ngôn ngữ ở backend và gọi các công cụ phù hợp.

Dự án hiện hỗ trợ hai bộ dữ liệu: **Olist Brazilian E-Commerce** và **DataCo Supply Chain**. Hai bộ dữ liệu có pipeline phân tích riêng; model XGBoost và bản đồ bang Brazil của Olist không được dùng để suy diễn cho DataCo.

## Nội dung

- [Tính năng chính](#tính-năng-chính)
- [Giao diện](#giao-diện)
- [Kiến trúc hệ thống](#kiến-trúc-hệ-thống)
- [Bộ dữ liệu](#bộ-dữ-liệu)
- [Machine Learning với Olist](#machine-learning-với-olist)
- [AI Agent và tự động chọn LLM](#ai-agent-và-tự-động-chọn-llm)
- [RAG tích hợp trong AI Logistics Agent](#rag-tích-hợp-trong-ai-logistics-agent)
- [Bản đồ logistics](#bản-đồ-logistics)
- [API](#api)
- [Cài đặt và chạy local](#cài-đặt-và-chạy-local)
- [Chạy bằng Docker](#chạy-bằng-docker)
- [Pipeline và kiểm thử](#pipeline-và-kiểm-thử)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Giới hạn hiện tại](#giới-hạn-hiện-tại)

## Tính năng chính

### Phân tích Olist E-commerce (Brazil)

- Dự đoán xác suất giao hàng trễ và ước tính thời gian giao hàng theo ngày cho đơn hàng mới.
- Tra cứu đơn hàng và thống kê theo bang, seller, ngành hàng; xếp hạng khu vực/ngành hàng có rủi ro cao.
- Phân tích dữ liệu bằng DuckDB qua các truy vấn SQL chỉ đọc.
- Phân tích funnel marketing, hiệu suất kênh, hồ sơ seller 360 độ và mối liên hệ giữa acquisition với logistics.
- Bản đồ luồng đơn hàng theo cặp bang gửi → bang nhận, có bộ lọc thời gian/ngành hàng, drill-down và dự đoán tuyến cho đơn mới.

### Phân tích DataCo Supply Chain (quốc tế)

- Chọn DataCo từ bộ chọn dataset trên giao diện chat.
- Tra cứu đơn, phân tích theo quốc gia/khu vực/thị trường/phương thức vận chuyển và tổng hợp thời gian vận chuyển.
- Chạy truy vấn SQL chỉ đọc trên bảng đã chuẩn hóa tên cột là **dataco_orders**.
- Bản đồ luồng quốc gia với các chỉ số số đơn, nhãn late-delivery risk có sẵn, thời gian vận chuyển trung bình và sales.
- Có thể phân tích dữ liệu access log tùy chọn nếu file được đặt cục bộ; IP và URL thô không được đưa vào bảng mà agent truy vấn.

### AI Agent và ứng dụng

- Tool/function calling để kết nối câu hỏi ngôn ngữ tự nhiên với các hàm nghiệp vụ.
- RAG với LangChain, Qdrant và embedding đa ngôn ngữ để truy xuất tài liệu, hiển thị citations và chủ động từ chối khi thiếu bằng chứng.
- Trang Knowledge Assistant và API RAG riêng; tool truy xuất tri thức cũng dùng được trong chat agent Olist/DataCo.
- Tự động chọn model LLM ở backend theo loại tác vụ, API key đã cấu hình và thứ tự dự phòng.
- Thử model/provider tiếp theo khi lần gọi trước gặp lỗi; giao diện chat chỉ cho chọn dataset, không bắt người dùng chọn model.
- Rule-based local fallback xử lý một số mẫu câu Olist đã hỗ trợ mà không cần gọi API bên ngoài.
- Flask web app, endpoint healthcheck và Docker multi-stage build.

## Giao diện

Ảnh chụp giao diện được lưu trong thư mục MyAgentDemo.

### Chat và phân tích nghiệp vụ

<img width="1919" height="935" alt="image" src="https://github.com/user-attachments/assets/96db6384-1eae-4f79-a566-f7f266a1ab66" />


### Bản đồ luồng đơn hàng Olist

<img width="1919" height="935" alt="image" src="https://github.com/user-attachments/assets/81162530-deb7-43fc-9439-5408cc19dd5e" />

### Bản đồ luồng đơn hàng DataCo

<img width="1919" height="938" alt="image" src="https://github.com/user-attachments/assets/eaf0771e-85b4-4b25-8171-096a60528141" />



## Kiến trúc hệ thống

~~~text
                  Web UI (Flask)
                       |
               Chọn dataset Olist/DataCo
                       |
                  Chat API /chat
                       |
              Automatic LLM Router
               /                 \
      Local rules (Olist)    Online LLM providers
               \                 /
                 Tool / Function Calling
                          |
          +---------------+----------------+
          |                                |
   Olist business tools              DataCo tools
   XGBoost + DuckDB                  DuckDB + safe fields
          |                                |
   Olist state-flow map              Country-flow map
~~~

**Luồng RAG:** tài liệu được allowlist → chunking bằng LangChain → FastEmbed embedding đa ngôn ngữ chạy local → Qdrant → retriever → LLM hiện có tạo câu trả lời kèm nguồn.

**Luồng xử lý Olist:** CSV gốc → gộp đơn và tạo đặc trưng → dataset đã xử lý → model XGBoost và các công cụ DuckDB → Flask UI/API.

**Luồng xử lý DataCo:** DataCo CSV → chuẩn hóa encoding/tên cột và loại trường nhạy cảm khỏi bảng truy vấn → DuckDB analytics, tra cứu đơn và bản đồ quốc gia.

## Bộ dữ liệu

### 1. Olist Brazilian E-Commerce

Nguồn công khai: [Brazilian E-Commerce Public Dataset by Olist — Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).

Các bảng gốc gồm đơn hàng, sản phẩm trong đơn, thanh toán, đánh giá, sản phẩm, khách hàng, seller, địa lý và bản dịch tên ngành hàng. Bản đã xử lý được lưu tại **data/processed_dataset.csv**; các file model và bảng tra cứu đặc trưng lịch sử nằm trong **models/**.

Pipeline tạo đặc trưng như khoảng cách Haversine, giá trị đơn, phí vận chuyển, số sản phẩm/seller/ngành hàng, khối lượng/thể tích, thời điểm đặt đơn, bang gửi/nhận và thống kê lịch sử. Những đặc trưng lịch sử cần được kiểm tra cẩn thận khi huấn luyện lại để đảm bảo không sử dụng thông tin tương lai hoặc gây leakage.

### 2. DataCo Supply Chain

Thư mục **data_2/** chứa:

- **DataCoSupplyChainDataset.csv** — dữ liệu đơn hàng/line-item chính.
- **DescriptionDataCoSupplyChain.csv** — mô tả trường dữ liệu.
- **tokenized_access_logs.csv** — file tùy chọn, không bắt buộc; chỉ đặt vào máy local nếu cần phân tích lượt truy cập.

Backend tạo bảng truy vấn **dataco_orders** và chuẩn hóa tên cột để phục vụ SQL. Các trường như tên, email, mật khẩu, địa chỉ đường phố, mã bưu chính và URL ảnh sản phẩm bị loại khỏi bảng mà agent sử dụng. Tuy vậy, đây không phải cam kết ẩn danh hóa toàn bộ dữ liệu nguồn; hãy kiểm tra quyền sử dụng và dữ liệu trước khi chia sẻ hoặc triển khai công khai.

**Giới hạn diễn giải DataCo:** trường **late_delivery_risk** là nhãn có sẵn trong dataset, không phải xác suất được model ML của dự án dự đoán. Bản đồ dùng **Customer Country → Order Country** làm proxy từ các trường dữ liệu; không nên coi đó là tuyến đi từ kho hoặc seller đã xác minh.

## Machine Learning với Olist

Dự án có hai model XGBoost được lưu dưới dạng joblib và dùng qua pipeline tiền xử lý của scikit-learn:

| Model | Mục tiêu | Chỉ số trong metadata hiện có |
|---|---|---|
| XGBClassifier | Phân loại nguy cơ giao trễ (is_late) | ROC-AUC: **0.814** |
| XGBRegressor | Ước tính số ngày giao hàng thực tế | MAE: **4.449 ngày** · RMSE: **7.28 ngày** |

Các con số trên được đọc từ **models/model_metadata.json**, không phải cam kết hiệu suất khi áp dụng cho dữ liệu ngoài Olist. Kết quả có thể thay đổi khi thay đổi dữ liệu, cách chia train/test hoặc pipeline.

### Chọn ngưỡng phân loại

Tỷ lệ đơn giao trễ trong tập test được lưu trong metadata là khoảng **8.11%**. Ngưỡng phân loại được điều chỉnh theo mục tiêu nghiệp vụ:

| Chế độ | Threshold | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Ưu tiên Recall — ngưỡng đang lưu | 0.38 | 0.193 | 0.750 | 0.307 |
| Ngưỡng tối ưu F1 trong metadata | 0.6493 | 0.335 | 0.488 | 0.397 |

Ngưỡng Recall cao giúp phát hiện nhiều đơn có nguy cơ hơn nhưng cũng tạo nhiều cảnh báo nhầm hơn. Cần chọn ngưỡng theo chi phí thực tế của bỏ sót đơn trễ và xử lý cảnh báo sai.

Các file model:

~~~text
models/
├── late_delivery_classifier.joblib
├── delivery_days_regressor.joblib
├── history_lookup.json
└── model_metadata.json
~~~

## AI Agent và tự động chọn LLM

Ứng dụng dùng **src/model_router.py** để định tuyến câu hỏi. Giao diện cho phép chọn bộ dữ liệu; model LLM không phải lựa chọn hiển thị trong chat.

Quy trình hiện tại:

1. Với Olist, kiểm tra trước xem rule-based agent có nhận diện được mẫu câu hỗ trợ hay không.
2. Nếu cần LLM trực tuyến, phân loại tác vụ ở mức quy tắc thành nhóm nhanh hoặc suy luận.
3. Lọc các model theo API key đã cấu hình, thử theo thứ tự ưu tiên và chuyển sang provider khác khi có lỗi.
4. Giới hạn số lần thử bằng **MODEL_ROUTER_MAX_ATTEMPTS** (tối đa 8 trong code hiện tại).

Các tích hợp trong code gồm **Google Gemini, OpenAI, xAI, Groq và OpenRouter**, cùng local rule-based fallback. Model/provider có thể thay đổi theo cấu hình, API key, quota và khả năng truy cập tại thời điểm chạy; không đảm bảo mọi model luôn hoạt động hoặc có gói miễn phí.

### Nhóm công cụ tiêu biểu

**Olist:** tra cứu đơn hàng; dự đoán đơn mới; thống kê theo state/seller/category; xếp hạng khu vực rủi ro; mô tả schema; truy vấn SQL; xem luồng trên bản đồ; các công cụ marketing.

**DataCo:** mô tả dataset; tra cứu đơn; thống kê khu vực/quốc gia; so sánh shipping mode; truy vấn SQL trên bảng đã làm sạch; thống kê access log tùy chọn; xem luồng quốc gia.

Truy vấn SQL được giới hạn số dòng trả về. Với DataCo, agent chỉ được truy vấn bảng **dataco_orders** và bị chặn nhiều từ khóa/hàm có khả năng thay đổi dữ liệu hoặc đọc file tùy ý. Đây là lớp bảo vệ ở mức ứng dụng, không thay thế việc kiểm soát quyền truy cập, xác thực và bảo vệ hạ tầng khi triển khai production.

## RAG tích hợp trong AI Logistics Agent

RAG là **một công cụ bên trong Agent**, không phải một giao diện chat riêng. Người dùng đặt tất cả câu hỏi tại **http://localhost:5000**; Agent tự lựa chọn giữa SQL/data tools, model ML và RAG dựa trên nội dung câu hỏi.

- Câu hỏi cần số liệu, thống kê hoặc phân tích đơn hàng: Agent dùng công cụ dữ liệu/SQL phù hợp.
- Câu hỏi về định nghĩa trường, tài liệu dự án hoặc quy trình: Agent gọi `query_knowledge_base`, truy xuất tài liệu liên quan từ Qdrant và dùng nguồn tìm được để trả lời.
- Câu hỏi kết hợp số liệu và tài liệu có thể dùng cả hai loại công cụ trong cùng một lượt hội thoại.
- Agent được hướng dẫn dẫn nguồn tài liệu và nói rõ khi kho tri thức không có bằng chứng đủ liên quan; RAG không thay thế SQL và không được dùng để bịa số liệu.

### Tạo hoặc cập nhật chỉ mục tài liệu

Chỉ ingest các nguồn đã được cho phép: README song ngữ, mô tả trường DataCo, metadata model và tài liệu được tuyển chọn trong `knowledge_base/`. CSV đơn hàng Olist/DataCo thô không được ingest. `knowledge_base/sample_delivery_sop.md` là SOP hư cấu để demo, không phải chính sách chính thức của Olist/DataCo.

**Docker:**

~~~bash
docker compose up --build -d
docker compose exec logistics-agent python -m src.rag_ingest
~~~

Mở **http://localhost:5000** rồi hỏi trực tiếp, ví dụ: “Trường `late_delivery_risk` trong DataCo có phải xác suất dự đoán của model không?”. Lần ingest đầu sẽ tải embedding model đa ngôn ngữ qua FastEmbed (ONNX, chạy CPU) và lưu cache vào volume `fastembed_cache`.

**Python chạy trực tiếp:**

~~~powershell
python -m src.rag_ingest
python -m src.api
~~~

Sau đó mở **http://localhost:5000**. Python chạy trực tiếp không tự động nạp `.env`; hãy cấu hình API key cho LLM trong terminal hoặc IDE. Embedding được tính local, còn việc sinh câu trả lời cần một LLM provider đã cấu hình.

### API RAG

| Method | Endpoint | Mục đích |
|---|---|---|
| GET | /api/v1/rag/status | Kiểm tra chỉ mục, số chunks và cấu hình backend |
| POST | /api/v1/rag/query | API hỏi đáp tài liệu trực tiếp, trả về nguồn; hữu ích cho tích hợp hoặc kiểm thử |

Ví dụ gọi API trực tiếp (không phải giao diện chat thứ hai):

~~~bash
curl -X POST http://localhost:5000/api/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{"question":"Trường late_delivery_risk có phải xác suất model dự đoán không?","top_k":4}'
~~~

API nhận `question` (hoặc `query`) và `top_k` từ 1 đến 8. Nguồn trả về gồm đường dẫn, điểm tương đồng và đoạn trích. URL cũ `/knowledge` sẽ chuyển về chat Agent chính.

### Cấu hình và giới hạn

- `RAG_EMBEDDING_MODEL` mặc định là `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`; FastEmbed dùng ONNX Runtime, không cần cài PyTorch.
- `RAG_LLM_PROVIDER=auto` chọn provider có cấu hình phù hợp; có thể dùng `gemini`, `groq`, `openrouter`, `openai` hoặc `xai`. `RAG_LLM_MODEL` là tuỳ chọn.
- `RAG_MIN_RELEVANCE_SCORE` điều chỉnh ngưỡng từ chối khi bằng chứng không đủ liên quan; nên đánh giá trên bộ câu hỏi thực tế.
- Chỉ lập chỉ mục tài liệu được phép sử dụng. Không commit secret, dữ liệu cá nhân hay tài liệu doanh nghiệp mật vào repository công khai.
- Demo chưa có authentication và ACL theo tài liệu; không đưa kho tài liệu riêng tư lên Internet trước khi bổ sung kiểm soát truy cập.

## Bản đồ logistics

### Olist: luồng đơn hàng giữa các bang Brazil

Mở **/map**. Bản đồ hỗ trợ:

- Luồng từ seller state đến customer state; độ dày và màu sắc thể hiện chỉ số đang chọn.
- Các chỉ số như số đơn, tỷ lệ giao trễ lịch sử, rủi ro dự đoán và thời gian giao trung bình.
- Bộ lọc tháng, ngành hàng và số đơn tối thiểu; lớp choropleth theo bang.
- Xem các đơn mẫu, xu hướng theo tháng và dự đoán cho đơn mới giữa hai bang.
- Điều hướng từ chat agent đến luồng được yêu cầu khi công cụ hỗ trợ.

Tọa độ địa lý dùng để tổng hợp và minh họa khu vực, không phải GPS hoặc theo dõi xe theo thời gian thực. Rủi ro dự đoán trên một luồng là kết quả cho đơn đại diện theo các giá trị đầu vào tổng hợp, không tương đương trực tiếp với tỷ lệ trễ lịch sử.

### DataCo: luồng quốc gia

Mở **/map?dataset=dataco**. Bản đồ hỗ trợ lọc theo ngày, market và shipping mode; chọn chỉ số hiển thị và drill-down để xem đơn mẫu.

**Lưu ý:** đây là hình ảnh hóa quan hệ giữa trường quốc gia khách hàng và quốc gia đích theo dữ liệu nguồn, không xác nhận tuyến đường vận tải vật lý.

## API

Ứng dụng chạy mặc định tại **http://localhost:5000**.

### API chung và Olist

| Method | Endpoint | Mục đích |
|---|---|---|
| GET | / | Giao diện chat |
| GET | /health | Healthcheck |
| GET | /models | Thông tin model phục vụ chẩn đoán |
| GET | /datasets | Dataset hiện có để chọn trên UI |
| POST | /chat | Gửi câu hỏi; JSON gồm message và dataset_id |
| POST | /chat/restore | Khôi phục lịch sử chat cho session |
| POST | /chat/reset | Xóa lịch sử chat của dataset trong session |
| POST | /predict | Dự đoán cho đơn mới theo input Olist |
| GET | /order/{order_id} | Tra cứu đơn Olist |
| GET | /stats/state/{state} | Thống kê theo bang |
| GET | /stats/seller/{seller_id} | Thống kê theo seller |
| GET | /stats/category/{category} | Thống kê theo ngành hàng |
| GET | /stats/top-risky-states?n=5 | Bang có tỷ lệ trễ cao |
| GET | /stats/top-risky-categories?n=5 | Ngành hàng có tỷ lệ trễ cao |
| GET | /marketing/funnel | Tóm tắt funnel marketing |
| GET | /marketing/channels | Hiệu suất kênh marketing |
| GET | /seller/{seller_id}/360 | Seller 360 |
| GET | /marketing/acquisition-logistics | Acquisition và logistics |

### API bản đồ Olist

~~~text
GET  /map
GET  /api/map/filters
GET  /api/map/flows
GET  /api/map/sample?from=SP&to=RJ&limit=40
GET  /api/map/trend?from=SP&to=RJ
POST /api/map/predict-route
GET  /api/map/geojson
~~~

### API bản đồ DataCo

~~~text
GET  /api/dataco/map/health
GET  /api/dataco/map/filters
GET  /api/dataco/map/summary
GET  /api/dataco/map/flows
GET  /api/dataco/map/orders?from=Brazil&to=Mexico
~~~

Ví dụ gọi chat:

~~~bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Top 5 bang có tỷ lệ giao trễ cao nhất","dataset_id":"olist"}'
~~~

Trên Windows PowerShell, có thể dùng giao diện web để thử API thay vì chạy lệnh curl ở trên.

## Cài đặt và chạy local

### Yêu cầu

- Git
- Python 3.12 được khuyến nghị để gần với môi trường Docker
- Tài nguyên đủ để đọc các CSV; lần đầu sử dụng DataCo có thể tốn thời gian để chuẩn hóa và nạp dữ liệu vào DuckDB
- API key của ít nhất một LLM provider trực tuyến nếu cần hỏi các câu không được rule-based agent Olist xử lý. DataCo cần model trực tuyến.

### 1. Tải source

~~~bash
git clone https://github.com/Mnn2kkk6/logistic_agent_project.git
cd logistic_agent_project
python -m venv .venv
~~~

Kích hoạt môi trường ảo:

~~~powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
~~~

~~~bash
# macOS / Linux
source .venv/bin/activate
~~~

### 2. Cài thư viện

Cách cài linh hoạt cho môi trường local:

~~~bash
pip install -r requirements.txt
~~~

Để dùng phiên bản thư viện đã pin tương ứng với image Docker, dùng Python 3.12 và:

~~~bash
pip install -r requirements-docker.txt
~~~

### 3. Cấu hình API key và chạy app

Ví dụ Windows PowerShell:

~~~powershell
$env:GEMINI_API_KEY="your-gemini-api-key"
$env:FLASK_SECRET_KEY="replace-with-a-random-secret"
python -m src.api
~~~

Có thể cấu hình các biến tương ứng cho OpenAI, xAI, Groq hoặc OpenRouter nếu muốn dùng những provider đó. **Chạy Python trực tiếp không tự động nạp file .env**; hãy đặt biến môi trường trong terminal hoặc cấu hình môi trường của IDE. Docker Compose tự đọc file .env để nội suy biến.

Mở:

- Chat: http://localhost:5000
- Healthcheck: http://localhost:5000/health
- Dataset list: http://localhost:5000/datasets
- Bản đồ Olist: http://localhost:5000/map
- Bản đồ DataCo: http://localhost:5000/map?dataset=dataco

Các biến thường dùng:

| Biến | Mục đích |
|---|---|
| GEMINI_API_KEY hoặc GOOGLE_API_KEY | Gemini |
| OPENAI_API_KEY | OpenAI |
| XAI_API_KEY | xAI |
| GROQ_API_KEY | Groq |
| OPENROUTER_API_KEY | OpenRouter |
| TAVILY_API_KEY | Tìm kiếm web tùy chọn |
| CARTO_API_KEY | Tùy chọn cho nền bản đồ CARTO |
| FLASK_SECRET_KEY | Ký Flask session |
| MODEL_ROUTER_FAST_MODEL | Ưu tiên model cho tác vụ nhanh |
| MODEL_ROUTER_REASONING_MODEL | Ưu tiên model cho tác vụ suy luận |
| MODEL_ROUTER_MAX_ATTEMPTS | Giới hạn số lượt thử model |

Không commit API key hoặc file **.env** chứa secret lên GitHub.

## Chạy bằng Docker

Docker là cách được khuyến nghị để chạy trong môi trường gần với cấu hình đã định nghĩa trong repo.

1. Tạo file **.env** từ **.env.example** và điền API key cần thiết.
2. Build và chạy:

~~~bash
docker compose up --build -d
~~~

3. Kiểm tra trạng thái:

~~~bash
docker compose ps
docker compose logs -f logistics-agent
~~~

4. Truy cập **http://localhost:5000**. Healthcheck là **http://localhost:5000/health**.

Để dùng RAG, chạy lệnh ingest trong mục RAG ở trên rồi mở **http://localhost:5000**.

Dừng ứng dụng:

~~~bash
docker compose down
~~~

Dockerfile dùng multi-stage build, runtime chạy bằng user không phải root và có healthcheck. Compose mount **data/** và **models/** ở chế độ chỉ đọc. Thư mục **data_2/** được copy vào image trong bước build, vì vậy cần build lại image khi muốn cập nhật file DataCo bên trong container.

## Pipeline và kiểm thử

Để tạo lại dataset Olist đã xử lý từ CSV gốc:

~~~bash
python -m src.data_pipeline
~~~

Huấn luyện lại model phân loại và hồi quy:

~~~bash
python -m src.train_models
~~~

Có thể huấn luyện riêng:

~~~bash
python -m src.train_models classifier
python -m src.train_models regressor
~~~

Đặt đầy đủ CSV Olist gốc vào **data/** trước khi chạy pipeline. Các model và dataset đã xử lý hiện được lưu sẵn trong repository để thuận tiện chạy thử; chỉ chạy training khi cần tái tạo hoặc thay đổi pipeline.

Một bộ kiểm tra DataCo có tại **tests/test_dataco_map.py**. Chạy bằng:

~~~bash
python -m unittest discover -s tests -p "test_dataco_map.py" -v
~~~

~~~bash
python -m unittest discover -s tests -p "test_rag.py" -v
~~~

Workflow GitHub Actions **.github/workflows/dataco-backend-tests.yml** có các bước compile Python, kiểm tra cú pháp JavaScript của bản đồ DataCo và chạy các test tương ứng khi trigger theo cấu hình workflow.

## Cấu trúc dự án

~~~text
.
├── data/                     # CSV Olist, dataset đã xử lý, GeoJSON Brazil
├── data_2/                   # CSV DataCo và mô tả dữ liệu
├── models/                   # XGBoost models, metadata, history lookup
├── knowledge_base/           # Tài liệu tri thức curated; SOP mẫu được ghi rõ là giả lập
├── src/
│   ├── api.py                # Flask UI/API và route bản đồ
│   ├── rag.py                # Ingest, embedding, Qdrant retrieval, citations và grounded generation
│   ├── rag_ingest.py         # CLI tạo/rebuild index
│   ├── agent.py              # Agent orchestration
│   ├── agent_gemini.py       # Tích hợp Gemini
│   ├── providers.py           # Provider adapters và tool calling
│   ├── model_router.py        # Tự chọn LLM và fallback
│   ├── dataset_registry.py    # Dataset selector/availability
│   ├── tools.py               # Olist tools, ML inference, DuckDB, map
│   ├── dataco_tools.py        # DataCo analytics và truy vấn an toàn
│   ├── dataco_map.py          # Tổng hợp luồng quốc gia
│   ├── dataco_country_coords.py
│   ├── marketing_tools.py     # Phân tích marketing và seller
│   ├── data_pipeline.py      # Tiền xử lý Olist
│   ├── history_features.py   # Đặc trưng lịch sử
│   └── train_models.py       # Huấn luyện model
├── templates/
│   ├── index.html            # Giao diện chat
│   ├── map.html              # Bản đồ Olist
│   └── dataco_map.html       # Bản đồ DataCo
├── tests/
│   └── test_dataco_map.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-docker.txt
├── .env.example
├── README.md
└── READMEENG.md
~~~

## Giới hạn hiện tại

- Chat history được lưu trong RAM của tiến trình Flask. Khi restart, lịch sử trong RAM mất; triển khai nhiều instance nên chuyển sang Redis/database và bổ sung cơ chế session phù hợp.
- Chất lượng và khả năng gọi tool phụ thuộc provider, API key, quota, mạng và model còn khả dụng.
- Rule-based local fallback chỉ xử lý các mẫu câu đã được lập trình và hiện chỉ dành cho Olist; DataCo yêu cầu provider trực tuyến.
- Dataset Olist và DataCo là dữ liệu lịch sử/phục vụ phân tích, không phải feed logistics thời gian thực.
- Tọa độ bản đồ phục vụ minh họa dữ liệu tổng hợp; không nên dùng như GPS hay bằng chứng về tuyến vận chuyển thực tế.
- Trước khi dùng trong vận hành thật, cần kiểm định model trên dữ liệu mới, đánh giá data leakage, bổ sung authentication/authorization, quản lý secret, logging/monitoring và lưu lịch sử bền vững.

## Công nghệ

**Python · Pandas · NumPy · scikit-learn · XGBoost · DuckDB · Flask · Google GenAI · OpenAI API · xAI · Groq · OpenRouter · Tavily · Leaflet · Docker Compose**
