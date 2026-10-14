🚚 AI Logistics Agent — Olist E-commerceWeb application hỗ trợ phân tích logistics cho sàn thương mại điện tử kết hợp Machine Learning, LLM Function Calling, SQL analytics và bản đồ luồng đơn hàng từ dữ liệu Olist Brazilian E-Commerce.English Documentation (READMEENG.md)📑 Mục lụcĐiểm nổi bậtBài toán và dữ liệuMachine LearningAI AgentMulti-provider LLMInteractive Logistics Flow MapTavily Web SearchWeb APIProject StructureChạy localChạy bằng DockerMột vài câu hỏi mẫuHạn chế và hướng phát triển1. Điểm nổi bậtLate-delivery prediction: Dự đoán xác suất giao hàng trễ cho một đơn hàng mới.Delivery-time prediction: Ước tính số ngày giao hàng thực tế.AI Agent: Người dùng hỏi bằng ngôn ngữ tự nhiên; LLM tự chọn tool phù hợp để lấy dữ liệu hoặc chạy phân tích.Multi-provider LLM: Hỗ trợ Gemini, OpenAI, xAI/Grok, Groq và local rule-based fallback.SQL analytics với DuckDB: Cho phép thực hiện các truy vấn SELECT / WITH ... SELECT trên dataset.Order lookup & operational statistics: Tra cứu đơn hàng, seller, bang khách hàng, ngành hàng và các nhóm rủi ro.Interactive flow map: Bản đồ luồng đơn hàng theo cặp seller state → customer state, có drill-down tới các đơn hàng mẫu và animation theo thời gian giao hàng.Optional Tavily integration: Có utility tìm kiếm web cho thông tin nằm ngoài phạm vi dữ liệu Olist.Dockerized deployment: Flask app chạy trong container với healthcheck và multi-stage build.2. Bài toán và dữ liệuDự án sử dụng bộ Olist Brazilian E-Commerce gồm 9 bảng CSV gốc, bao gồm orders, order items, payments, reviews, products, customers, sellers, geolocation và category translation.Pipeline chính:PlaintextRaw Olist CSV 
  │
  ▼ Load + Merge + Feature Engineering
  ├──> History features (seller / category / state-pair / customer-state)
  │
  ▼ processed_dataset.csv
  ├───────────────────┬───────────────────┐
  │                   │                   │
  ▼                   ▼                   ▼
ML Training     DuckDB Analytics      Agent Tools / SQL
  │                   │
  ▼                   ▼
2 Production Models ──┴──> Flask Web Application ──> Chat UI <--> Flow Map
Các đặc trưng chínhCác feature được thiết kế để mô phỏng quyết định ngay tại thời điểm đặt hàng, tránh dùng thông tin chỉ có sau khi đơn đã được giao. Ví dụ:Khoảng cách seller → customer bằng Haversine.Giá sản phẩm, phí vận chuyển, payment value, installments.Số item, số seller/category khác nhau trong đơn.Tổng khối lượng, thể tích, giá/item và các log-transformed features.Bang seller, bang customer, category và payment type.Thứ, tháng, giờ đặt hàng, weekend và peak season.Historical features theo seller, category, state pair và customer state.Targets:is_late: Đơn có giao sau ngày dự kiến hay không.actual_delivery_days: Thời gian giao thực tế tính từ lúc mua đến lúc khách nhận hàng.3. Machine LearningHai model production được huấn luyện bằng XGBoost trong sklearn.Pipeline, kết hợp tiền xử lý numeric/categorical features.ModelMục tiêuKết quả testXGBClassifieris_lateROC-AUC: 0.814XGBRegressoractual_delivery_daysMAE: 4.449 ngày, RMSE: 7.28 ngàyClassifier: Hai operating pointsDo dữ liệu mất cân bằng (late-rate test khoảng 8.11%), threshold có thể được điều chỉnh theo mục tiêu nghiệp vụ:Chế độThresholdPrecisionRecallF1Recall-priority (deployed)0.380.1930.7500.307F1-optimal0.64930.3350.4880.397Cách tiếp cận này cho phép đánh đổi giữa bắt được nhiều đơn có nguy cơ trễ và giảm số cảnh báo nhầm thay vì cố định threshold = 0.5.Metadata của model được lưu tại: models/model_metadata.json4. AI AgentLuồng hội thoại:PlaintextUser 
  │
  ▼ Flask Chat API
  │
  ▼ Selected LLM Provider
  ├──> Function Calling
  │      ├── get_order_info
  │      ├── predict_new_order
  │      ├── get_state_stats
  │      ├── get_seller_stats
  │      ├── get_category_stats
  │      ├── top_risky_states
  │      ├── top_risky_categories
  │      ├── describe_dataset
  │      ├── query_dataset_sql
  │      └── train_custom_model
  │
  ▼ Tool result
  │
  ▼ LLM final answer
Tool groups:Prediction: predict_new_order — Dùng các thông tin có thể biết khi đặt hàng để trả về rủi ro giao trễ và số ngày giao dự kiến. Khi có seller_id, agent có thể bổ sung historical performance của seller.Lookup / statistics: get_order_info, get_state_stats, get_seller_stats, get_category_stats, top_risky_states, top_risky_categories.General analytics: describe_dataset, query_dataset_sql, train_custom_model, show_flow_on_map.query_dataset_sql chỉ cho phép các truy vấn đọc dữ liệu và có thể truy cập bảng orders đã tổng hợp cùng các bảng raw cần thiết cho các bài toán join chi tiết.5. Multi-provider LLMNgười dùng có thể chọn model trực tiếp trên web UI:ProviderModels in current projectGoogle GeminiGemini 2.5 Flash, 2.5 Flash-Lite, 2.0 FlashOpenAIGPT-4o Mini, GPT-4oxAI GrokGrok 4 Fast, Grok 4.6GroqGPT-OSS 120B, GPT-OSS 20BLocalRule-based offline fallbackLịch sử hội thoại được lưu theo dạng canonical chat history, không phụ thuộc provider. Nhờ đó có thể đổi model giữa các lượt chat mà không phải lưu format tool-call riêng của từng API.6. Interactive Logistics Flow MapMở: http://localhost:5000/mapBản đồ dùng Leaflet (nền tối CARTO hoặc OpenStreetMap, đổi được ở góc phải) và gồm:Luồng theo bang: Các cung cong seller_state → customer_state; độ dày = số đơn, màu = chỉ số đang chọn.Bốn chỉ số tô màu: Tỉ lệ trễ lịch sử, rủi ro dự đoán (XGBoost), thời gian giao trung bình, số đơn. Legend tự co giãn theo dữ liệu.Choropleth theo bang (nhận hàng hoặc gửi hàng) từ GeoJSON 27 bang (data/brazil_states.geojson). Bấm vào một bang để chỉ xem các luồng của bang đó.Bộ lọc: Khoảng tháng, ngành hàng, số đơn tối thiểu mỗi luồng.Top 10 luồng nổi bật theo chỉ số đang chọn (bấm để drill-down).Hover: Tooltip số liệu + sparkline late rate theo tháng của luồng.Drill-down: 40 đơn mẫu lấy phân tầng (có cả đơn trễ lẫn đúng hẹn), chấm chạy dọc cung cong với tốc độ tỉ lệ nghịch số ngày giao thực tế; có nút Tạm dừng và chọn tốc độ 0.5×–4×.Dự đoán đơn mới: Chọn bang seller/khách, ngành hàng, giá, phí ship, cân nặng → vẽ tuyến lên bản đồ kèm xác suất trễ, mức rủi ro và số ngày dự kiến.Kết nối với AI Agent: Tool show_flow_on_map — khi bạn hỏi "cho tôi xem luồng SP → RJ trên bản đồ", giao diện chat tự mở bản đồ cạnh khung chat và zoom tới luồng đó. Chiều ngược lại, nút "Hỏi agent" trong màn hình chi tiết gửi sẵn câu hỏi vào chat.Có thể mở thẳng bằng URL: /map?from=SP&to=RJ&mode=pred (mode: late | pred | days | orders).Lưu ý: Vị trí trên bản đồ là tọa độ trung bình theo khu vực/zip code, dùng để minh họa luồng logistics, không phải GPS thời gian thực. "Rủi ro dự đoán" là kết quả model cho đơn điển hình của luồng (khoảng cách, giá, ship, cân nặng, hạn giao lấy theo trung vị/trung bình thật của luồng), nên không phải con số so sánh trực tiếp 1-1 với late rate lịch sử.Kết quả tổng hợp được cache trong bộ nhớ và được tính sẵn khi server khởi động (tắt bằng MAP_WARMUP=0).Map Endpoints:GET /mapGET /api/map/filtersGET /api/map/flows?start=2017-01&end=2018-08&category=bed_bath_table&min_orders=5GET /api/map/sample?from=SP&to=RJ&limit=40[&start=&end=&category=]GET /api/map/trend?from=SP&to=RJ[&start=&end=&category=]POST /api/map/predict-routeGET /api/map/geojson7. Tavily Web SearchProject có utility web_search sử dụng Tavily để lấy thông tin từ web cho các câu hỏi nằm ngoài dataset Olist, ví dụ kiến thức logistics hoặc thông tin bên ngoài dữ liệu nội bộ.API key được cấu hình bằng:Đoạn mãTAVILY_API_KEY=your-tavily-api-key
Trong production, nên ưu tiên dữ liệu nội bộ trước khi tìm kiếm web vì dữ liệu Olist có tính nhất quán với các tool analytics của project.8. Web APICác endpoint chính:Core & ChatGET / — Web chat UIGET /health — HealthcheckGET /models — Available LLM models + key statusPOST /chat — Chat with selected modelPOST /chat/reset — Reset current conversationPrediction & LookupPOST /predict — Predict a new orderGET /order/<order_id> — Lookup an orderGET /stats/state/GET /stats/seller/<seller_id>GET /stats/category/GET /stats/top-risky-states?n=5GET /stats/top-risky-categories?n=5Map API (Xem Mục 6)9. Project StructurePlaintext.
├── data/
│   ├── olist_*_dataset.csv
│   ├── product_category_name_translation.csv
│   ├── brazil_states.geojson
│   └── processed_dataset.csv
├── models/
│   ├── late_delivery_classifier.joblib
│   ├── delivery_days_regressor.joblib
│   ├── history_lookup.json
│   └── model_metadata.json
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
├── templates/
│   ├── index.html
│   └── map.html
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-docker.txt
├── .env.example
├── README.md
└── READMEENG.md
10. Chạy local10.1. Cài dependenciesBashpip install -r requirements.txt
10.2. Cấu hình environmentCopy file mẫu:Bashcp .env.example .env
Sau đó điền API key cần dùng.⚠️ Không commit file .env thật lên GitHub.10.3. Chạy FlaskPowerShell:PowerShell$env:GEMINI_API_KEY="your-key"
python -m src.api
Mở web chat: http://localhost:5000Mở bản đồ: http://localhost:5000/map10.4. Chạy data pipeline / trainingTạo lại processed dataset:Bashpython -m src.data_pipeline
Train classifier + regressor:Bashpython -m src.train_models
Có thể train riêng:Bashpython -m src.train_models classifier
python -m src.train_models regressor
11. Chạy bằng DockerBuild và start:Bashdocker compose up --build
Kiểm tra:Bashdocker compose ps
Healthcheck: http://localhost:5000/healthỨng dụng: http://localhost:5000Bản đồ: http://localhost:5000/mapDocker sử dụng multi-stage build: dependencies được cài ở builder stage, sau đó runtime image chỉ giữ môi trường cần thiết để chạy app.Dataset và model được mount read-only trong Compose để có thể thay thế data/model mà không cần rebuild image.12. Một vài câu hỏi mẫuTop 5 bang có tỉ lệ giao hàng trễ cao nhất?Đơn hàng từ seller ở SP giao đến khách ở RJ cách khoảng 900km có rủi ro trễ không?Seller này có hiệu suất giao hàng tốt không?So sánh tỉ lệ giao trễ theo tháng.Ngành hàng nào có thời gian giao hàng cao nhất?Cho tôi xem luồng đơn hàng từ SP đến RJ trên bản đồ.13. Hạn chế và hướng phát triểnDữ liệu Olist là dữ liệu lịch sử; bản đồ chỉ mang tính trực quan hóa.Chat history hiện lưu trong RAM của Flask process, phù hợp cho local/demo; triển khai nhiều instance nên chuyển sang Redis hoặc database.Các provider cần API key riêng; local fallback chỉ hỗ trợ một số mẫu câu cố định.Có thể mở rộng thêm seller recommendation, route-aware prediction, model monitoring, drift detection và long-term conversation memory.Có thể cải thiện classifier theo chi phí nghiệp vụ thực tế bằng threshold tuning hoặc cost-sensitive learning.Tech Stack: Python, Pandas, scikit-learn, XGBoost, DuckDB, Flask, Google GenAI, OpenAI API, xAI API, Groq, Tavily, Leaflet, Docker
