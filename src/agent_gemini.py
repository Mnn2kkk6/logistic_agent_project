"""
AI Logistics Agent — bản chạy bằng Google Gemini (MIỄN PHÍ qua Google AI Studio)
================================================================================
Thay vì dùng Anthropic Claude (trả phí theo API), bản này dùng Gemini API,
vốn có gói miễn phí (free tier) khi lấy key tại: https://aistudio.google.com/apikey
(không cần thẻ tín dụng, có giới hạn số lượt gọi/phút/ngày).

Cách chạy:
    1) Lấy API key miễn phí tại https://aistudio.google.com/apikey
    2) pip install google-genai
    3) Windows PowerShell:  $env:GOOGLE_API_KEY="dán-key-vào-đây"
       (Mac/Linux:          export GOOGLE_API_KEY="dán-key-vào-đây")
    4) python -m src.agent_gemini
"""
import os
import sys

from src import tools

MODEL = "gemini-2.5-flash"  # nằm trong nhóm model có hạn mức miễn phí

SYSTEM_PROMPT = """Bạn là AI Logistics Agent của một sàn thương mại điện tử.
Bạn có quyền truy cập dữ liệu đơn hàng lịch sử (Olist, Brazil) và các mô hình
Machine Learning để dự đoán rủi ro giao hàng trễ và số ngày giao hàng dự kiến.

Nhiệm vụ:
- Trả lời câu hỏi về tình trạng đơn hàng, hiệu suất giao hàng theo khu vực/seller/ngành hàng.
- Khi người dùng mô tả một đơn hàng MỚI (chưa giao), hãy dùng tool predict_new_order
  để ước tính rủi ro trễ và số ngày giao hàng, rồi đưa ra khuyến nghị ngắn gọn.
- Luôn trả lời bằng tiếng Việt, ngắn gọn, có số liệu cụ thể, không bịa dữ liệu.
- Nếu tool trả về found=False hoặc không có dữ liệu, hãy nói rõ với người dùng.
"""


# ==== Các hàm "tool" — có type hint + docstring rõ ràng để Gemini tự đọc schema ====

def get_order_info(order_id: str) -> dict:
    """Tra cứu thông tin và tình trạng một đơn hàng đã có trong hệ thống.

    Args:
        order_id: Mã đơn hàng (order_id) cần tra cứu.
    """
    return tools.get_order_info(order_id)


def predict_new_order(
    distance_km: float = None,
    total_price: float = None,
    total_freight: float = None,
    payment_value: float = None,
    payment_installments: int = None,
    product_weight_g: float = None,
    product_volume_cm3: float = None,
    n_items: int = None,
    estimated_days: int = None,
    customer_state: str = None,
    seller_state: str = None,
    product_category_name_english: str = None,
    purchase_dow: int = None,
    purchase_month: int = None,
    purchase_hour: int = None,
) -> dict:
    """Dự đoán rủi ro giao trễ (%) và số ngày giao hàng dự kiến cho một đơn hàng MỚI.

    Các trường không cung cấp sẽ dùng giá trị mặc định trung bình của dataset.

    Args:
        distance_km: Khoảng cách seller-khách hàng, tính bằng km.
        total_price: Tổng giá trị sản phẩm.
        total_freight: Tổng phí vận chuyển.
        payment_value: Tổng giá trị thanh toán.
        payment_installments: Số kỳ trả góp.
        product_weight_g: Khối lượng sản phẩm (gram).
        product_volume_cm3: Thể tích sản phẩm (cm3).
        n_items: Số lượng sản phẩm trong đơn.
        estimated_days: Số ngày hẹn giao ước tính cho khách.
        customer_state: Mã bang khách hàng, ví dụ 'SP', 'RJ', 'BA'.
        seller_state: Mã bang seller.
        product_category_name_english: Tên ngành hàng (tiếng Anh).
        purchase_dow: Thứ trong tuần lúc mua (0 = Thứ 2).
        purchase_month: Tháng lúc mua (1-12).
        purchase_hour: Giờ lúc mua (0-23).
    """
    kwargs = {k: v for k, v in locals().items() if v is not None}
    return tools.predict_new_order(**kwargs)


def get_state_stats(state: str) -> dict:
    """Thống kê hiệu suất giao hàng theo bang của khách hàng.

    Args:
        state: Mã bang, ví dụ 'SP', 'RJ', 'MA'.
    """
    return tools.get_state_stats(state)


def get_seller_stats(seller_id: str) -> dict:
    """Thống kê hiệu suất giao hàng của một seller cụ thể.

    Args:
        seller_id: Mã seller cần tra cứu.
    """
    return tools.get_seller_stats(seller_id)


def get_category_stats(category: str) -> dict:
    """Thống kê hiệu suất giao hàng theo ngành hàng.

    Args:
        category: Tên ngành hàng bằng tiếng Anh, ví dụ 'furniture_decor'.
    """
    return tools.get_category_stats(category)


def top_risky_states(n: int = 5) -> dict:
    """Trả về top N bang có tỉ lệ giao hàng trễ cao nhất.

    Args:
        n: Số lượng bang muốn lấy (mặc định 5).
    """
    return tools.top_risky_states(n)


def top_risky_categories(n: int = 5) -> dict:
    """Trả về top N ngành hàng có tỉ lệ giao hàng trễ cao nhất.

    Args:
        n: Số lượng ngành hàng muốn lấy (mặc định 5).
    """
    return tools.top_risky_categories(n)


AGENT_TOOLS = [
    get_order_info, predict_new_order, get_state_stats,
    get_seller_stats, get_category_stats, top_risky_states, top_risky_categories,
]


def chat_loop():
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Chưa có GOOGLE_API_KEY. Lấy key miễn phí tại: https://aistudio.google.com/apikey")
        print("Rồi chạy: $env:GOOGLE_API_KEY=\"key-cua-ban\"  (PowerShell)")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    chat = client.chats.create(
        model=MODEL,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=AGENT_TOOLS,
        ),
    )

    print("=== AI Logistics Agent (Gemini, miễn phí) — gõ 'exit' để thoát ===")
    while True:
        user_input = input("\nBạn: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        try:
            response = chat.send_message(user_input)
            print(f"\nAgent: {response.text}")
        except Exception as e:  # noqa: BLE001
            print(f"Lỗi khi gọi Gemini API: {e}")


if __name__ == "__main__":
    chat_loop()