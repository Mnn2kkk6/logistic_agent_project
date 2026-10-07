"""
Định nghĩa TOOLS (JSON Schema) và SYSTEM_PROMPT dùng CHUNG cho mọi provider
(Gemini, OpenAI...). Tách riêng file này để agent.py, agent_gemini.py và
providers.py đều dùng đúng 1 nguồn, tránh lệch schema giữa các bản.
"""

SYSTEM_PROMPT = """Bạn là AI Logistics Agent của một sàn thương mại điện tử.
Bạn có quyền truy cập dữ liệu đơn hàng lịch sử (Olist, Brazil) và các mô hình
Machine Learning để dự đoán rủi ro giao hàng trễ và số ngày giao hàng dự kiến.

Nhiệm vụ:
- Trả lời câu hỏi về tình trạng đơn hàng, hiệu suất giao hàng theo khu vực/seller/ngành hàng.
- Khi người dùng mô tả một đơn hàng MỚI (chưa giao), hãy dùng tool `predict_new_order`
  để ước tính rủi ro trễ và số ngày giao hàng, rồi đưa ra khuyến nghị ngắn gọn
  (ví dụ: nên chọn seller gần hơn, cảnh báo rủi ro cao, ước tính ngày giao cho khách).
- Với BẤT KỲ câu hỏi thống kê/phân tích nào KHÔNG khớp với các tool có sẵn
  (get_state_stats, get_seller_stats, get_category_stats, top_risky_states,
  top_risky_categories) — ví dụ so sánh theo tháng, theo phương thức thanh toán, theo
  cân nặng/thể tích sản phẩm, theo khoảng cách, v.v. — hãy dùng `query_dataset_sql` để
  tự viết SQL trả lời. Nếu chưa chắc tên cột, gọi `describe_dataset` trước.
- Nếu người dùng muốn dự đoán/phân tích một target KHÁC ngoài is_late và
  actual_delivery_days (ví dụ: điều gì ảnh hưởng tới review_score, hoặc muốn một model
  chỉ train riêng cho 1 khu vực/ngành hàng), dùng `train_custom_model` để train nhanh
  một model tạm và báo cáo metric + feature quan trọng nhất — không dùng để thay thế
  2 model production chính.
- Khi câu hỏi cần thông tin bên ngoài dataset Olist hoặc thông tin hiện tại trên Internet
  (ví dụ: tin tức logistics, chính sách vận chuyển, thông tin công ty, kiến thức ngành
  không có trong dữ liệu nội bộ), hãy dùng tool `web_search`.
  Không dùng `web_search` cho các câu hỏi có thể trả lời bằng dataset nội bộ hoặc các tool
  analytics hiện có.
- Luôn trả lời bằng tiếng Việt, ngắn gọn, có số liệu cụ thể, không bịa dữ liệu.
- Nếu tool trả về found=False, error, hoặc không có dữ liệu, hãy nói rõ với người dùng.
- Khi câu hỏi liên quan đến nguồn gốc seller, marketing channel, MQL,
  closed deal, acquisition, thời gian chốt seller, seller activation,
  hoặc quan hệ giữa acquisition và logistics, hãy dùng các tool:
  get_marketing_funnel_summary,
  get_marketing_channel_performance,
  get_seller_360,
  get_acquisition_logistics_performance.
- Khi phân tích Seller 360, hãy kết hợp thông tin acquisition với
  logistics performance thay vì chỉ trả về một phía.
- GMV trong marketing/logistics analysis là giá trị item quan sát được
  trong dataset, không được gọi là doanh thu kế toán hay contractual LTV.
"""

TOOLS = [
    {
        "name": "get_order_info",
        "description": "Tra cứu thông tin và tình trạng một đơn hàng đã có trong hệ thống theo order_id.",
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string", "description": "Mã đơn hàng (order_id)"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "predict_new_order",
        "description": (
            "Dự đoán rủi ro giao trễ (%) và số ngày giao hàng dự kiến cho một đơn hàng MỚI, "
            "dựa trên các đặc trưng biết được tại thời điểm đặt hàng. Các trường không cung cấp "
            "sẽ dùng giá trị mặc định trung bình."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "distance_km": {"type": "number", "description": "Khoảng cách seller-khách hàng (km)"},
                "total_price": {"type": "number"},
                "total_freight": {"type": "number", "description": "Phí vận chuyển"},
                "payment_value": {"type": "number"},
                "payment_installments": {"type": "integer"},
                "product_weight_g": {"type": "number"},
                "product_volume_cm3": {"type": "number"},
                "n_items": {"type": "integer"},
                "estimated_days": {"type": "integer", "description": "Số ngày hẹn giao ước tính cho khách"},
                "customer_state": {"type": "string", "description": "Mã bang khách hàng, vd 'SP', 'RJ'"},
                "seller_state": {"type": "string", "description": "Mã bang seller"},
                "product_category_name_english": {"type": "string"},
                "purchase_dow": {"type": "integer", "description": "Thứ trong tuần lúc mua (0=T2)"},
                "purchase_month": {"type": "integer"},
                "purchase_hour": {"type": "integer"},
                "seller_id": {"type": "string", "description": "Mã seller (nếu biết) — dùng để tra cứu hiệu suất lịch sử thực tế của seller đó thay vì mức trung bình toàn sàn"},
            },
        },
    },
    {
        "name": "get_state_stats",
        "description": "Thống kê hiệu suất giao hàng (tỉ lệ trễ, số ngày giao TB) theo bang của khách hàng.",
        "input_schema": {
            "type": "object",
            "properties": {"state": {"type": "string", "description": "Mã bang, vd 'SP', 'RJ', 'MA'"}},
            "required": ["state"],
        },
    },
    {
        "name": "get_seller_stats",
        "description": "Thống kê hiệu suất giao hàng của một seller cụ thể theo seller_id.",
        "input_schema": {
            "type": "object",
            "properties": {"seller_id": {"type": "string"}},
            "required": ["seller_id"],
        },
    },
    {
        "name": "get_category_stats",
        "description": "Thống kê hiệu suất giao hàng theo ngành hàng (product_category_name_english).",
        "input_schema": {
            "type": "object",
            "properties": {"category": {"type": "string"}},
            "required": ["category"],
        },
    },
    {
        "name": "top_risky_states",
        "description": "Trả về top N bang có tỉ lệ giao hàng trễ cao nhất.",
        "input_schema": {
            "type": "object",
            "properties": {"n": {"type": "integer", "default": 5}},
        },
    },
    {
        "name": "top_risky_categories",
        "description": "Trả về top N ngành hàng có tỉ lệ giao hàng trễ cao nhất.",
        "input_schema": {
            "type": "object",
            "properties": {"n": {"type": "integer", "default": 5}},
        },
    },
    {
        "name": "describe_dataset",
        "description": (
            "Liệt kê TẤT CẢ các bảng khả dụng (orders đã gộp + order_items, order_payments, "
            "order_reviews, products, sellers, customers, category_translation ở mức dòng gốc), "
            "kèm tên cột, kiểu dữ liệu, vài dòng mẫu. Gọi tool này TRƯỚC khi dùng query_dataset_sql "
            "nếu chưa chắc chắn tên bảng/cột chính xác — đặc biệt QUAN TRỌNG khi câu hỏi liên quan "
            "đến doanh thu/số lượng theo category, theo seller, theo sản phẩm, hoặc cần join "
            "nhiều bảng, vì bảng 'orders' đã gộp KHÔNG đủ chính xác cho các trường hợp này."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "query_dataset_sql",
        "description": (
            "Chạy một câu SQL (SELECT hoặc WITH...SELECT) tuỳ ý trên các bảng có sẵn — 'orders' "
            "(đã gộp mỗi đơn 1 dòng) hoặc các bảng gốc mức dòng 'order_items', 'order_payments', "
            "'order_reviews', 'products', 'sellers', 'customers', 'category_translation' — để trả "
            "lời BẤT KỲ câu hỏi thống kê/lọc/nhóm/JOIN nào không có sẵn tool riêng. QUAN TRỌNG: "
            "bảng 'orders' chỉ lưu category/seller của SẢN PHẨM ĐẦU TIÊN mỗi đơn — với câu hỏi về "
            "doanh thu/số lượng THEO category, THEO seller, THEO sản phẩm cụ thể, hoặc cần đối "
            "chiếu payment với price+freight ở mức dòng, BẮT BUỘC phải JOIN từ order_items (mỗi "
            "dòng = 1 sản phẩm trong đơn), KHÔNG được dùng category/seller trong bảng 'orders'. "
            "Chỉ cho phép câu lệnh đọc dữ liệu (SELECT)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"sql": {"type": "string", "description": "Câu lệnh SQL (SELECT ...) trên bảng 'orders'"}},
            "required": ["sql"],
        },
    },
    {
        "name": "web_search",
        "description": (
            "Tìm kiếm thông tin trên Internet bằng Tavily khi câu hỏi nằm ngoài phạm vi "
            "dataset Olist hoặc cần thông tin hiện tại. Không dùng tool này cho các câu hỏi "
            "có thể trả lời bằng dữ liệu nội bộ."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Câu truy vấn cần tìm trên Internet",
                },
                "max_results": {
                    "type": "integer",
                    "default": 5,
                    "description": "Số kết quả tối đa, từ 1 đến 10",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "train_custom_model",
        "description": (
            "Huấn luyện nhanh một model RandomForest tạm thời khi cần dự đoán/phân tích một "
            "target KHÁC ngoài is_late/actual_delivery_days (2 model production chính), hoặc "
            "cần train riêng cho một subset dữ liệu cụ thể. Trả về metric + feature quan trọng nhất."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "Tên cột nhãn cần dự đoán, vd 'review_score', 'is_late'"},
                "feature_columns": {"type": "array", "items": {"type": "string"}, "description": "Danh sách cột đặc trưng (tuỳ chọn)"},
                "filter_sql": {"type": "string", "description": "Điều kiện WHERE để lọc subset, vd \"customer_state = 'RJ'\" (tuỳ chọn)"},
                "task": {"type": "string", "enum": ["classification", "regression"], "description": "Loại bài toán"},
            },
            "required": ["target", "task"],
        },
    },
]
TOOLS.extend([
    {
        "name": "get_marketing_funnel_summary",
        "description": (
            "Thống kê toàn bộ marketing funnel MQL -> Closed Deal. "
            "Có thể lọc theo acquisition origin."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "origin": {
                    "type": "string",
                    "description": "Kênh acquisition, ví dụ organic_search, paid_search, social"
                }
            }
        },
    },
    {
        "name": "get_marketing_channel_performance",
        "description": (
            "So sánh hiệu quả các acquisition channel theo số MQL, "
            "closed deals, conversion rate và thời gian trung bình để chốt."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "default": 20
                }
            }
        },
    },
    {
        "name": "get_seller_360",
        "description": (
            "Phân tích toàn diện một seller: nguồn acquisition, thời gian "
            "chốt deal, business profile, order performance, late rate, "
            "delivery time, GMV và tốc độ có đơn đầu tiên."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "seller_id": {
                    "type": "string",
                    "description": "Mã seller_id"
                }
            },
            "required": ["seller_id"],
        },
    },
    {
        "name": "get_acquisition_logistics_performance",
        "description": (
            "So sánh các acquisition channel với logistics performance "
            "của seller sau khi seller được onboard, gồm số seller hoạt động, "
            "số order, late rate và GMV quan sát được."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "default": 20
                }
            }
        },
    },
])
