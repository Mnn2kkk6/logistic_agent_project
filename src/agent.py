"""
AI Logistics Agent
===================
Agent hội thoại (tiếng Việt) trả lời câu hỏi về logistics dựa trên bộ dữ liệu
Olist E-commerce, sử dụng Gemini (Google Gen AI SDK) với Function Calling để
gọi các hàm nghiệp vụ trong tools.py (dự đoán ML + tra cứu thống kê + SQL tuỳ ý).

Cách chạy:
    export GEMINI_API_KEY="AIza..."
    python3 -m src.agent

Nếu KHÔNG có GEMINI_API_KEY, chương trình sẽ tự chuyển sang chế độ
"offline demo" (menu chọn sẵn) để vẫn có thể trình diễn được các tool.
"""
import json
import os
import sys

from src import tools

MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")

_GENAI_CLIENT = None

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
- Luôn trả lời bằng tiếng Việt, ngắn gọn, có số liệu cụ thể, không bịa dữ liệu.
- Nếu tool trả về found=False, error, hoặc không có dữ liệu, hãy nói rõ với người dùng.
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
            "Liệt kê tên cột, kiểu dữ liệu và vài dòng mẫu của dataset. Gọi tool này TRƯỚC "
            "khi dùng query_dataset_sql nếu chưa chắc chắn tên cột chính xác."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "query_dataset_sql",
        "description": (
            "Chạy một câu SQL (SELECT hoặc WITH...SELECT) tuỳ ý trên bảng 'orders' để trả lời "
            "BẤT KỲ câu hỏi thống kê/lọc/nhóm nào không có sẵn tool riêng — ví dụ so sánh theo "
            "tháng, theo phương thức thanh toán, theo cân nặng sản phẩm, tương quan giữa 2 biến, "
            "v.v. Chỉ cho phép câu lệnh đọc dữ liệu (SELECT)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"sql": {"type": "string", "description": "Câu lệnh SQL (SELECT ...) trên bảng 'orders'"}},
            "required": ["sql"],
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

TOOL_FUNCTIONS = {
    "get_order_info": tools.get_order_info,
    "predict_new_order": tools.predict_new_order,
    "get_state_stats": tools.get_state_stats,
    "get_seller_stats": tools.get_seller_stats,
    "get_category_stats": tools.get_category_stats,
    "top_risky_states": tools.top_risky_states,
    "top_risky_categories": tools.top_risky_categories,
    "describe_dataset": tools.describe_dataset,
    "query_dataset_sql": tools.query_dataset_sql,
    "train_custom_model": tools.train_custom_model,
}


def run_tool(name: str, tool_input: dict) -> dict:
    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        return {"error": f"Không tìm thấy tool '{name}'"}
    try:
        return func(**tool_input)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def _build_gemini_tools():
    """Chuyển TOOLS (JSON Schema kiểu Claude) sang định dạng Gemini FunctionDeclaration.
    Gemini SDK nhận thẳng JSON Schema qua `parameters_json_schema` nên không cần đổi
    "type": "string"/"object" sang enum viết hoa như Schema thủ công."""
    from google.genai import types

    declarations = [
        types.FunctionDeclaration(
            name=t["name"],
            description=t["description"],
            parameters_json_schema=t["input_schema"],
        )
        for t in TOOLS
    ]
    return [types.Tool(function_declarations=declarations)]


def _generate_with_retry(client, config, contents, max_retries: int = 4):
    """Gọi generate_content, tự retry (backoff) khi gặp lỗi tạm thời từ server Gemini
    (503 quá tải, 429 rate limit) — lỗi thật (400 sai request, 401/403 sai key) thì raise ngay."""
    import time

    from google.genai import errors

    delay = 2.0
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(model=MODEL, contents=contents, config=config)
        except errors.APIError as e:
            if e.code not in (429, 503) or attempt == max_retries - 1:
                raise
            print(f"  [retry] Gemini trả lỗi {e.code} (quá tải/rate limit), thử lại sau {delay:.0f}s...")
            time.sleep(delay)
            delay *= 2


def _get_genai_client():
    """Tạo (và cache) 1 genai.Client dùng chung, đọc GEMINI_API_KEY/GOOGLE_API_KEY từ env."""
    global _GENAI_CLIENT
    if _GENAI_CLIENT is None:
        from google import genai
        _GENAI_CLIENT = genai.Client()
    return _GENAI_CLIENT


def chat_loop_with_gemini():
    from google.genai import types

    client = _get_genai_client()
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=_build_gemini_tools(),
    )
    contents = []
    print("=== AI Logistics Agent (Gemini) — gõ 'exit' để thoát ===")
    while True:
        user_input = input("\nBạn: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        contents.append(types.Content(role="user", parts=[types.Part(text=user_input)]))

        while True:  # vòng lặp tool-use, tối đa tới khi model trả lời text
            response = _generate_with_retry(client, config, contents)
            candidate = response.candidates[0]
            contents.append(candidate.content)

            function_calls = [p.function_call for p in candidate.content.parts if p.function_call]
            if not function_calls:
                print(f"\nAgent: {response.text}")
                break

            response_parts = []
            for fc in function_calls:
                args = dict(fc.args) if fc.args else {}
                result = run_tool(fc.name, args)
                print(f"  [tool_use] {fc.name}({args}) -> {json.dumps(result, ensure_ascii=False)[:200]}")
                response_parts.append(types.Part.from_function_response(name=fc.name, response={"result": result}))
            contents.append(types.Content(role="user", parts=response_parts))


def offline_demo():
    """Chế độ demo không cần API key: gọi trực tiếp các tool để minh hoạ."""
    print("=== AI Logistics Agent — CHẾ ĐỘ OFFLINE DEMO ===")
    print("(Không tìm thấy GEMINI_API_KEY, minh hoạ trực tiếp các tool)\n")

    print("1) Top 5 bang có tỉ lệ giao trễ cao nhất:")
    print(json.dumps(tools.top_risky_states(5), ensure_ascii=False, indent=2))

    print("\n2) Thống kê bang 'SP':")
    print(json.dumps(tools.get_state_stats("SP"), ensure_ascii=False, indent=2))

    print("\n3) Dự đoán cho một đơn hàng mới (khách ở RJ, seller ở SP, xa 900km):")
    result = tools.predict_new_order(
        distance_km=900, customer_state="RJ", seller_state="SP",
        product_category_name_english="moveis_decoracao", total_freight=45,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))

    print("\nĐể dùng agent hội thoại tự nhiên bằng tiếng Việt, hãy đặt biến môi trường")
    print("GEMINI_API_KEY rồi chạy lại: python3 -m src.agent")


if __name__ == "__main__":
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        try:
            chat_loop_with_gemini()
        except Exception as e:  # noqa: BLE001
            print(f"Lỗi khi gọi Gemini API: {e}", file=sys.stderr)
            print("Chuyển sang chế độ offline demo...\n")
            offline_demo()
    else:
        offline_demo()