"""
Lớp trừu tượng đa-provider cho AI Logistics Agent.
================================================

Cho phép người dùng CHỌN MODEL (Gemini Flash / Flash-Lite / GPT-4o / GPT-4o mini...)
để tránh bị chặn khi 1 provider hết quota miễn phí. Business logic (tools.py, model ML)
dùng chung 100% — chỉ phần "bộ não" điều phối tool là khác nhau giữa các provider.

THIẾT KẾ QUAN TRỌNG — lịch sử hội thoại "canonical":
Lịch sử chat lưu ở dạng đơn giản, KHÔNG phụ thuộc provider:
    [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
Các bước gọi tool (tool_use/tool_result) chỉ tồn tại TẠM THỜI trong nội bộ 1 lượt xử lý
(1 lần gọi chat_turn), không lưu vào lịch sử lâu dài. Nhờ vậy người dùng có thể đổi model
giữa chừng cuộc trò chuyện mà không bị lỗi format, vì mỗi provider tự dựng lại request
của mình từ lịch sử canonical này ở đầu mỗi lượt.

Đánh đổi: nếu đổi model giữa chừng, model mới sẽ không "nhớ" các tool đã gọi ở lượt
trước (chỉ nhớ nội dung hội thoại dạng text) — chấp nhận được vì mục tiêu chính là
tránh nghẽn quota, không phải giữ nguyên vẹn 100% ngữ cảnh nội bộ.
"""
import json
import os
import re
import time

from src.tool_schemas import SYSTEM_PROMPT, TOOLS
from src.tools import (
    describe_dataset,
    get_category_stats,
    get_order_info,
    get_seller_stats,
    get_state_stats,
    predict_new_order,
    query_dataset_sql,
    top_risky_categories,
    top_risky_states,
    train_custom_model,
)

TOOL_FUNCTIONS = {
    "get_order_info": get_order_info,
    "predict_new_order": predict_new_order,
    "get_state_stats": get_state_stats,
    "get_seller_stats": get_seller_stats,
    "get_category_stats": get_category_stats,
    "top_risky_states": top_risky_states,
    "top_risky_categories": top_risky_categories,
    "describe_dataset": describe_dataset,
    "query_dataset_sql": query_dataset_sql,
    "train_custom_model": train_custom_model,
}

# ==== Danh sách model hỗ trợ ====
# "env_key": biến môi trường chứa API key cần thiết cho provider đó.
AVAILABLE_MODELS = {
    "gemini-2.5-flash": {
        "provider": "gemini", "label": "Gemini 2.5 Flash", "env_key": "GEMINI_API_KEY",
        "note": "Cân bằng tốc độ/chất lượng — free tier Google AI Studio",
    },
    "gemini-2.5-flash-lite": {
        "provider": "gemini", "label": "Gemini 2.5 Flash-Lite", "env_key": "GEMINI_API_KEY",
        "note": "Nhanh & rẻ nhất, hạn mức free tier cao hơn — dùng khi model kia hết quota",
    },
    "gemini-2.0-flash": {
        "provider": "gemini", "label": "Gemini 2.0 Flash", "env_key": "GEMINI_API_KEY",
        "note": "Bản dự phòng thế hệ trước",
    },
    "gpt-4o-mini": {
        "provider": "openai", "label": "GPT-4o Mini", "env_key": "OPENAI_API_KEY",
        "note": "Cần OPENAI_API_KEY (trả phí, nhưng rẻ) — dùng khi cả 2 Gemini đều hết quota",
    },
    "gpt-4o": {
        "provider": "openai", "label": "GPT-4o", "env_key": "OPENAI_API_KEY",
        "note": "Cần OPENAI_API_KEY (trả phí) — chất lượng cao nhất",
    },
    "grok-4-fast": {
        "provider": "xai", "label": "Grok 4 Fast", "env_key": "XAI_API_KEY",
        "note": "Cần XAI_API_KEY (trả phí, nhưng rẻ) — dự phòng khi Gemini/GPT đều hết quota",
    },
    "grok-4.6": {
        "provider": "xai", "label": "Grok 4.6", "env_key": "XAI_API_KEY",
        "note": "Cần XAI_API_KEY (trả phí) — model xAI mạnh nhất hiện tại",
    },
    "local-rule-based": {
        "provider": "local", "label": "Local (miễn phí, offline)", "env_key": None,
        "note": "Không cần API key, không gọi mạng — chỉ nhận diện vài mẫu câu cố định "
                "(không hiểu ngôn ngữ tự nhiên linh hoạt). Dùng khi MỌI model khác đều hết quota.",
    },
}
DEFAULT_MODEL = "gemini-2.5-flash"


class ProviderError(Exception):
    """Lỗi có thể hiển thị trực tiếp cho người dùng (thiếu key, hết quota, model sai...)."""


def list_models() -> list:
    """Trả về danh sách model kèm trạng thái đã cấu hình API key hay chưa (để FE hiện rõ)."""
    result = []
    for model_id, info in AVAILABLE_MODELS.items():
        env_key = info["env_key"]
        configured = True if env_key is None else bool(os.environ.get(env_key))
        result.append({
            "id": model_id,
            "label": info["label"],
            "provider": info["provider"],
            "note": info["note"],
            "configured": configured,
        })
    return result


def _run_tool(name: str, args: dict) -> dict:
    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        return {"error": f"Không tìm thấy tool '{name}'"}
    try:
        return func(**args)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def chat_turn(model: str, history: list, user_message: str, max_tool_rounds: int = 5):
    """
    history: list[{"role": "user"|"assistant", "content": str}] — lịch sử canonical.
    Trả về: (reply_text, tool_calls_log, new_history)
    Raise ProviderError với thông báo tiếng Việt rõ ràng nếu model không hỗ trợ,
    thiếu API key, hoặc provider trả lỗi không phục hồi được.
    """
    info = AVAILABLE_MODELS.get(model)
    if info is None:
        raise ProviderError(f"Model '{model}' không được hỗ trợ. Model hợp lệ: {', '.join(AVAILABLE_MODELS)}")

    if info["env_key"] is not None and not os.environ.get(info["env_key"]):
        raise ProviderError(f"Chưa cấu hình biến môi trường {info['env_key']} cho model '{model}'.")

    if info["provider"] == "gemini":
        reply, tool_calls = _gemini_turn(model, history, user_message, max_tool_rounds)
    elif info["provider"] == "openai":
        reply, tool_calls = _openai_turn(model, history, user_message, max_tool_rounds)
    elif info["provider"] == "xai":
        reply, tool_calls = _xai_turn(model, history, user_message, max_tool_rounds)
    elif info["provider"] == "local":
        reply, tool_calls = _local_turn(model, history, user_message, max_tool_rounds)
    else:  # pragma: no cover — không nên xảy ra vì AVAILABLE_MODELS kiểm soát chặt
        raise ProviderError(f"Provider '{info['provider']}' chưa được cài đặt.")

    new_history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": reply},
    ]
    return reply, tool_calls, new_history


# ============================== GEMINI ==============================

def _gemini_tools():
    from google.genai import types
    declarations = [
        types.FunctionDeclaration(
            name=t["name"], description=t["description"], parameters_json_schema=t["input_schema"],
        )
        for t in TOOLS
    ]
    return [types.Tool(function_declarations=declarations)]


def _gemini_client():
    from google import genai
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    return genai.Client(api_key=api_key)


def _gemini_generate_with_retry(client, model, contents, config, max_retries=4):
    from google.genai import errors
    delay = 2.0
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(model=model, contents=contents, config=config)
        except errors.APIError as e:
            if e.code not in (429, 503):
                raise ProviderError(f"Gemini API lỗi {e.code}: {e.message if hasattr(e, 'message') else e}")
            if attempt == max_retries - 1:
                raise ProviderError(f"Gemini quá tải/hết quota (lỗi {e.code}) sau {max_retries} lần thử — hãy đổi sang model khác.")
            time.sleep(delay)
            delay *= 2


def _gemini_turn(model, history, user_message, max_tool_rounds):
    from google.genai import types

    client = _gemini_client()
    config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, tools=_gemini_tools())

    contents = []
    for turn in history:
        role = "user" if turn["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=turn["content"])]))
    contents.append(types.Content(role="user", parts=[types.Part(text=user_message)]))

    tool_calls_log = []
    for _ in range(max_tool_rounds):
        response = _gemini_generate_with_retry(client, model, contents, config)
        candidate = response.candidates[0]
        contents.append(candidate.content)
        function_calls = [p.function_call for p in candidate.content.parts if p.function_call]
        if not function_calls:
            return response.text or "", tool_calls_log
        response_parts = []
        for fc in function_calls:
            args = dict(fc.args) if fc.args else {}
            result = _run_tool(fc.name, args)
            tool_calls_log.append({"name": fc.name, "args": args})
            response_parts.append(types.Part.from_function_response(name=fc.name, response={"result": result}))
        contents.append(types.Content(role="user", parts=response_parts))
    return "(Đã đạt giới hạn số vòng gọi tool cho 1 câu hỏi, vui lòng hỏi lại cụ thể hơn.)", tool_calls_log


# =================== OPENAI-COMPATIBLE (OpenAI + xAI/Grok) ===================
# Cả OpenAI và xAI đều dùng chung format API (client "openai" SDK), chỉ khác
# base_url + API key, nên gộp chung 1 hàm để tránh lặp code.

def _openai_compatible_tools():
    return [
        {"type": "function", "function": {
            "name": t["name"], "description": t["description"], "parameters": t["input_schema"],
        }}
        for t in TOOLS
    ]


def _openai_compatible_turn(model, history, user_message, max_tool_rounds, api_key, base_url, provider_label):
    from openai import APIError, OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend({"role": t["role"], "content": t["content"]} for t in history)
    messages.append({"role": "user", "content": user_message})

    tool_calls_log = []
    for _ in range(max_tool_rounds):
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, tools=_openai_compatible_tools(), tool_choice="auto",
            )
        except APIError as e:
            raise ProviderError(f"{provider_label} API lỗi: {e}")
        msg = response.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))
        if not msg.tool_calls:
            return msg.content or "", tool_calls_log
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            result = _run_tool(tc.function.name, args)
            tool_calls_log.append({"name": tc.function.name, "args": args})
            messages.append({
                "role": "tool", "tool_call_id": tc.id, "content": json.dumps(result, ensure_ascii=False),
            })
    return "(Đã đạt giới hạn số vòng gọi tool cho 1 câu hỏi, vui lòng hỏi lại cụ thể hơn.)", tool_calls_log


def _openai_turn(model, history, user_message, max_tool_rounds):
    return _openai_compatible_turn(
        model, history, user_message, max_tool_rounds,
        api_key=os.environ.get("OPENAI_API_KEY"), base_url=None, provider_label="OpenAI",
    )


def _xai_turn(model, history, user_message, max_tool_rounds):
    return _openai_compatible_turn(
        model, history, user_message, max_tool_rounds,
        api_key=os.environ.get("XAI_API_KEY"), base_url="https://api.x.ai/v1", provider_label="xAI (Grok)",
    )


# ============================== LOCAL (rule-based, miễn phí) ==============================
# KHÔNG gọi bất kỳ API nào — chỉ nhận diện từ khoá/mẫu câu đơn giản (regex) để quyết định
# gọi tool nào. Không hiểu ngôn ngữ tự nhiên linh hoạt như LLM thật, nhưng luôn dùng được
# kể cả khi mọi provider trả phí/miễn phí khác đều hết quota — dự phòng "chót" an toàn.

BRAZIL_STATES = {
    "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT", "MS", "MG",
    "PA", "PB", "PR", "PE", "PI", "RJ", "RN", "RS", "RO", "RR", "SC", "SP", "SE", "TO",
}

WEEKDAY_NAMES_VN = {0: "Thứ 2", 1: "Thứ 3", 2: "Thứ 4", 3: "Thứ 5", 4: "Thứ 6", 5: "Thứ 7", 6: "Chủ nhật"}

_known_categories_cache = None


def _get_known_categories() -> list:
    """Danh sách ngành hàng có thật trong dataset (cache lại, chỉ truy vấn 1 lần)."""
    global _known_categories_cache
    if _known_categories_cache is None:
        result = _run_tool("query_dataset_sql", {
            "sql": "SELECT DISTINCT product_category_name_english AS c FROM orders "
                   "WHERE product_category_name_english IS NOT NULL",
        })
        _known_categories_cache = [r["c"] for r in result.get("rows", [])]
    return _known_categories_cache


def _find_categories_in_text(text: str) -> list:
    """Tìm các ngành hàng có thật xuất hiện trong text (so khớp dạng có gạch dưới hoặc
    có khoảng trắng thay cho gạch dưới), ưu tiên tên dài hơn trước để tránh khớp nhầm
    tên ngắn nằm lọt bên trong tên dài."""
    lower = text.lower()
    found = []
    for cat in sorted(_get_known_categories(), key=len, reverse=True):
        cat_l = cat.lower()
        if cat_l in lower or cat_l.replace("_", " ") in lower:
            if cat not in found:
                found.append(cat)
    return found


def _extract_states_in_order(text: str) -> list:
    """Trả về danh sách mã bang Brazil hợp lệ xuất hiện trong text, theo đúng thứ tự."""
    found = []
    for tok in re.findall(r"\b[A-Za-z]{2}\b", text):
        up = tok.upper()
        if up in BRAZIL_STATES:
            found.append(up)
    return found


def _format_predict_reply(result: dict, provided_kwargs: dict) -> str:
    if "error" in result:
        return f"Không dự đoán được: {result['error']}"
    chi_tiet = []
    if "distance_km" in provided_kwargs:
        chi_tiet.append(f"khoảng cách {provided_kwargs['distance_km']}km")
    if "customer_state" in provided_kwargs:
        chi_tiet.append(f"khách ở {provided_kwargs['customer_state']}")
    if "seller_state" in provided_kwargs:
        chi_tiet.append(f"seller ở {provided_kwargs['seller_state']}")
    dieu_kien = " (" + ", ".join(chi_tiet) + ")" if chi_tiet else " (không trích được thông tin cụ thể — dùng toàn bộ giá trị mặc định)"
    return (
        f"Dự đoán cho đơn hàng{dieu_kien}:\n"
        f"- Xác suất giao trễ: {result['late_probability_pct']}% (mức rủi ro: {result['risk_level']})\n"
        f"- Số ngày giao dự kiến: {result['predicted_delivery_days']} ngày\n\n"
        f"(Lưu ý: các trường không được cung cấp đã dùng giá trị mặc định trung bình của dataset —"
        f" độ chính xác sẽ thấp hơn so với khi hỏi qua model AI thật.)"
    )


def _format_order_reply(result: dict) -> str:
    trang_thai_tre = "chưa rõ" if result.get("is_late") is None else ("CÓ" if result["is_late"] else "KHÔNG")
    return (
        f"Đơn hàng {result['order_id']}:\n"
        f"- Trạng thái: {result['status']}\n"
        f"- Khách ở bang: {result['customer_state']} | Seller ở bang: {result['seller_state']}\n"
        f"- Ngành hàng: {result['product_category']}\n"
        f"- Giao trễ: {trang_thai_tre}"
        + (f" (thực giao trong {result['actual_delivery_days']} ngày)" if result.get("actual_delivery_days") else "")
    )


def _local_turn(model, history, user_message, max_tool_rounds):
    text = user_message.strip()
    lower = text.lower()
    tool_calls_log = []

    def call(name, args):
        result = _run_tool(name, args)
        tool_calls_log.append({"name": name, "args": args})
        return result

    # 1) Dự đoán đơn hàng mới — có "dự đoán"/"rủi ro" hoặc có số + "km"
    if any(k in lower for k in ["dự đoán", "du doan", "rủi ro trễ", "rui ro tre", "có trễ không", "co tre khong"]) \
            or re.search(r"\d+\s*km", lower):
        kwargs = {}
        m = re.search(r"(\d+(?:[.,]\d+)?)\s*km", lower)
        if m:
            kwargs["distance_km"] = float(m.group(1).replace(",", "."))
        m2 = re.search(r"ph[íi]\s*ship\D{0,10}(\d+(?:[.,]\d+)?)", lower)
        if m2:
            kwargs["total_freight"] = float(m2.group(1).replace(",", "."))

        cust_m = re.search(r"kh[aá]ch[^,.;\n]{0,20}?\b([A-Za-z]{2})\b", text, re.IGNORECASE)
        seller_m = re.search(r"seller[^,.;\n]{0,20}?\b([A-Za-z]{2})\b", text, re.IGNORECASE)
        if cust_m and cust_m.group(1).upper() in BRAZIL_STATES:
            kwargs["customer_state"] = cust_m.group(1).upper()
        if seller_m and seller_m.group(1).upper() in BRAZIL_STATES:
            kwargs["seller_state"] = seller_m.group(1).upper()
        if "customer_state" not in kwargs or "seller_state" not in kwargs:
            states = _extract_states_in_order(text)
            states = [s for s in states if s not in kwargs.values()]
            if "seller_state" not in kwargs and states:
                kwargs["seller_state"] = states.pop(0)
            if "customer_state" not in kwargs and states:
                kwargs["customer_state"] = states.pop(0)

        result = call("predict_new_order", kwargs)
        return _format_predict_reply(result, kwargs), tool_calls_log

    # 2) Tra cứu 1 đơn hàng cụ thể (order_id dạng chuỗi hex dài)
    if "đơn hàng" in lower or "order" in lower or "mã đơn" in lower:
        m = re.search(r"\b([0-9a-fA-F]{16,32})\b", text)
        if m:
            result = call("get_order_info", {"order_id": m.group(1)})
            if not result.get("found"):
                return f"Không tìm thấy đơn hàng '{m.group(1)}' trong dữ liệu.", tool_calls_log
            return _format_order_reply(result), tool_calls_log

    # 3) Thống kê 1 seller cụ thể (seller_id dạng chuỗi hex dài)
    if "seller" in lower:
        m = re.search(r"\b([0-9a-fA-F]{16,32})\b", text)
        if m:
            result = call("get_seller_stats", {"seller_id": m.group(1)})
            if not result.get("found"):
                return f"Không có dữ liệu cho seller_id '{m.group(1)}'.", tool_calls_log
            return (
                f"Thống kê seller {m.group(1)} (bang {result['seller_state']}): "
                f"{result['n_orders']} đơn, tỉ lệ trễ {result['late_rate_pct']}%, "
                f"TB {result['avg_delivery_days']} ngày giao"
                + (f", điểm đánh giá TB {result['avg_review_score']}." if result.get("avg_review_score") is not None else ".")
            ), tool_calls_log

    # 4) So sánh 2 bang
    if any(k in lower for k in ["so sánh", "so sanh"]):
        states = _extract_states_in_order(text)
        if len(states) >= 2:
            a, b = call("get_state_stats", {"state": states[0]}), call("get_state_stats", {"state": states[1]})
            if a.get("found") and b.get("found"):
                return (
                    f"So sánh {states[0]} vs {states[1]}:\n"
                    f"- {states[0]}: {a['n_orders']} đơn, trễ {a['late_rate_pct']}%, TB {a['avg_delivery_days']} ngày giao\n"
                    f"- {states[1]}: {b['n_orders']} đơn, trễ {b['late_rate_pct']}%, TB {b['avg_delivery_days']} ngày giao"
                ), tool_calls_log

        cats = _find_categories_in_text(text)
        if len(cats) >= 2:
            a, b = call("get_category_stats", {"category": cats[0]}), call("get_category_stats", {"category": cats[1]})
            if a.get("found") and b.get("found"):
                return (
                    f"So sánh {cats[0]} vs {cats[1]}:\n"
                    f"- {cats[0]}: {a['n_orders']} đơn, trễ {a['late_rate_pct']}%, TB {a['avg_delivery_days']} ngày giao\n"
                    f"- {cats[1]}: {b['n_orders']} đơn, trễ {b['late_rate_pct']}%, TB {b['avg_delivery_days']} ngày giao"
                ), tool_calls_log

    # 5) Tỉ lệ trễ theo tháng
    if ("tháng" in lower or "thang" in lower) and any(k in lower for k in ["trễ", "tre", "tỉ lệ", "ti le"]):
        result = call("query_dataset_sql", {
            "sql": "SELECT purchase_month, ROUND(AVG(is_late)*100, 2) AS late_rate_pct, COUNT(*) AS n "
                   "FROM orders WHERE order_status='delivered' GROUP BY purchase_month ORDER BY purchase_month",
        })
        rows = result.get("rows", [])
        if not rows:
            return "Không lấy được dữ liệu theo tháng.", tool_calls_log
        lines = [f"Tháng {r['purchase_month']}: {r['late_rate_pct']}% trễ ({r['n']} đơn)" for r in rows]
        return "Tỉ lệ giao trễ theo tháng:\n" + "\n".join(lines), tool_calls_log

    # 6) Tỉ lệ trễ theo thứ trong tuần
    if "trong tuần" in lower or "trong tuan" in lower:
        result = call("query_dataset_sql", {
            "sql": "SELECT purchase_dow, ROUND(AVG(is_late)*100, 2) AS late_rate_pct, COUNT(*) AS n "
                   "FROM orders WHERE order_status='delivered' GROUP BY purchase_dow ORDER BY purchase_dow",
        })
        rows = result.get("rows", [])
        if not rows:
            return "Không lấy được dữ liệu theo thứ trong tuần.", tool_calls_log
        lines = [f"{WEEKDAY_NAMES_VN.get(r['purchase_dow'], r['purchase_dow'])}: {r['late_rate_pct']}% trễ ({r['n']} đơn)" for r in rows]
        return "Tỉ lệ giao trễ theo thứ trong tuần (lúc đặt hàng):\n" + "\n".join(lines), tool_calls_log

    # 7) Tỉ lệ trễ theo khung giờ đặt hàng
    if any(k in lower for k in ["khung giờ", "khung gio", "giờ đặt hàng", "gio dat hang"]):
        result = call("query_dataset_sql", {
            "sql": "SELECT purchase_hour, ROUND(AVG(is_late)*100, 2) AS late_rate_pct, COUNT(*) AS n "
                   "FROM orders WHERE order_status='delivered' GROUP BY purchase_hour ORDER BY late_rate_pct DESC LIMIT 5",
        })
        rows = result.get("rows", [])
        if not rows:
            return "Không lấy được dữ liệu theo khung giờ.", tool_calls_log
        lines = [f"{r['purchase_hour']}h: {r['late_rate_pct']}% trễ ({r['n']} đơn)" for r in rows]
        return "Top khung giờ đặt hàng có tỉ lệ giao trễ cao nhất:\n" + "\n".join(lines), tool_calls_log

    # 8) Cùng bang vs khác bang
    if "cùng bang" in lower or "cung bang" in lower or "khác bang" in lower or "khac bang" in lower:
        result = call("query_dataset_sql", {
            "sql": "SELECT same_state, ROUND(AVG(is_late)*100, 2) AS late_rate_pct, "
                   "ROUND(AVG(actual_delivery_days), 2) AS avg_days, COUNT(*) AS n "
                   "FROM orders WHERE order_status='delivered' GROUP BY same_state",
        })
        rows = {r["same_state"]: r for r in result.get("rows", [])}
        cung = rows.get(1)
        khac = rows.get(0)
        if not cung or not khac:
            return "Không lấy được dữ liệu so sánh cùng bang/khác bang.", tool_calls_log
        return (
            f"So sánh đơn hàng CÙNG bang vs KHÁC bang (seller-khách hàng):\n"
            f"- Cùng bang: {cung['late_rate_pct']}% trễ, TB {cung['avg_days']} ngày giao ({cung['n']} đơn)\n"
            f"- Khác bang: {khac['late_rate_pct']}% trễ, TB {khac['avg_days']} ngày giao ({khac['n']} đơn)"
        ), tool_calls_log

    # 9) Ảnh hưởng số kỳ trả góp
    if "trả góp" in lower or "tra gop" in lower or "kỳ hạn" in lower or "ky han" in lower:
        result = call("query_dataset_sql", {
            "sql": "SELECT payment_installments, ROUND(AVG(is_late)*100, 2) AS late_rate_pct, COUNT(*) AS n "
                   "FROM orders WHERE order_status='delivered' GROUP BY payment_installments "
                   "HAVING COUNT(*) >= 50 ORDER BY payment_installments LIMIT 10",
        })
        rows = result.get("rows", [])
        if not rows:
            return "Không lấy được dữ liệu theo số kỳ trả góp.", tool_calls_log
        lines = [f"{r['payment_installments']} kỳ: {r['late_rate_pct']}% trễ ({r['n']} đơn)" for r in rows]
        return "Tỉ lệ giao trễ theo số kỳ trả góp:\n" + "\n".join(lines), tool_calls_log

    # 10) Ảnh hưởng cân nặng sản phẩm
    if any(k in lower for k in ["cân nặng", "can nang", "nặng", "nang nhe", "nặng nhẹ"]):
        result = call("query_dataset_sql", {
            "sql": """
                SELECT
                    CASE
                        WHEN product_weight_g < 500 THEN '< 500g'
                        WHEN product_weight_g < 2000 THEN '500g - 2kg'
                        WHEN product_weight_g < 10000 THEN '2kg - 10kg'
                        ELSE '>= 10kg'
                    END AS nhom_can_nang,
                    ROUND(AVG(is_late)*100, 2) AS late_rate_pct, COUNT(*) AS n
                FROM orders WHERE order_status='delivered' AND product_weight_g IS NOT NULL
                GROUP BY nhom_can_nang ORDER BY MIN(product_weight_g)
            """,
        })
        rows = result.get("rows", [])
        if not rows:
            return "Không lấy được dữ liệu theo cân nặng sản phẩm.", tool_calls_log
        lines = [f"{r['nhom_can_nang']}: {r['late_rate_pct']}% trễ ({r['n']} đơn)" for r in rows]
        return "Tỉ lệ giao trễ theo nhóm cân nặng sản phẩm:\n" + "\n".join(lines), tool_calls_log

    # 11) Điểm đánh giá (review score) thấp/cao nhất theo bang hoặc ngành hàng
    if any(k in lower for k in ["đánh giá", "danh gia", "review"]):
        if "ngành hàng" in lower or "nganh hang" in lower or "category" in lower:
            result = call("query_dataset_sql", {
                "sql": "SELECT product_category_name_english AS nhom, ROUND(AVG(review_score), 2) AS avg_review, COUNT(*) AS n "
                       "FROM orders WHERE order_status='delivered' AND review_score IS NOT NULL "
                       "GROUP BY nhom HAVING COUNT(*) >= 50 ORDER BY avg_review ASC LIMIT 5",
            })
            nhan = "ngành hàng"
        else:
            result = call("query_dataset_sql", {
                "sql": "SELECT customer_state AS nhom, ROUND(AVG(review_score), 2) AS avg_review, COUNT(*) AS n "
                       "FROM orders WHERE order_status='delivered' AND review_score IS NOT NULL "
                       "GROUP BY nhom HAVING COUNT(*) >= 50 ORDER BY avg_review ASC LIMIT 5",
            })
            nhan = "bang"
        rows = result.get("rows", [])
        if not rows:
            return "Không lấy được dữ liệu điểm đánh giá.", tool_calls_log
        lines = [f"{r['nhom']}: điểm TB {r['avg_review']}/5 ({r['n']} đơn)" for r in rows]
        return f"Top {nhan} có điểm đánh giá (review score) thấp nhất:\n" + "\n".join(lines), tool_calls_log

    # 12) Tổng quan dataset: tổng số đơn / khoảng thời gian
    if any(k in lower for k in ["bao nhiêu đơn", "bao nhieu don", "tổng số đơn", "tong so don"]):
        result = call("query_dataset_sql", {"sql": "SELECT COUNT(*) AS n FROM orders"})
        n = result.get("rows", [{}])[0].get("n")
        return f"Dataset có tổng cộng {n} đơn hàng.", tool_calls_log

    if any(k in lower for k in ["năm nào", "nam nao", "khoảng thời gian", "khoang thoi gian"]):
        result = call("query_dataset_sql", {
            "sql": "SELECT MIN(order_purchase_timestamp) AS tu, MAX(order_purchase_timestamp) AS den FROM orders",
        })
        row = result.get("rows", [{}])[0]
        return f"Dữ liệu trải dài từ {row.get('tu')} đến {row.get('den')}.", tool_calls_log

    # 13) Top N bang trễ nhất
    if "bang" in lower and any(k in lower for k in ["trễ", "tre", "rủi ro", "rui ro"]):
        n_match = re.search(r"top\s*(\d+)|(\d+)\s*bang", lower)
        n = int(next(g for g in (n_match.groups() if n_match else []) if g)) if n_match else 5
        result = call("top_risky_states", {"n": n})
        rows = result.get("results", [])
        if not rows:
            return "Không lấy được dữ liệu thống kê bang.", tool_calls_log
        lines = [
            f"{i+1}. {r['customer_state']}: {r['late_rate_pct']}% trễ "
            f"({r['n_orders']} đơn, TB {r['avg_delivery_days']} ngày giao)"
            for i, r in enumerate(rows)
        ]
        return f"Top {len(rows)} bang có tỉ lệ giao trễ cao nhất:\n" + "\n".join(lines), tool_calls_log

    # 14) Top N ngành hàng trễ nhất
    if any(k in lower for k in ["ngành hàng", "nganh hang", "category"]) and any(k in lower for k in ["trễ", "tre", "rủi ro", "rui ro"]):
        n_match = re.search(r"top\s*(\d+)", lower)
        n = int(n_match.group(1)) if n_match else 5
        result = call("top_risky_categories", {"n": n})
        rows = result.get("results", [])
        if not rows:
            return "Không lấy được dữ liệu thống kê ngành hàng.", tool_calls_log
        lines = [
            f"{i+1}. {r['product_category_name_english']}: {r['late_rate_pct']}% trễ ({r['n_orders']} đơn)"
            for i, r in enumerate(rows)
        ]
        return f"Top {len(rows)} ngành hàng có tỉ lệ giao trễ cao nhất:\n" + "\n".join(lines), tool_calls_log

    # 15) Thống kê 1 ngành hàng cụ thể (không phải top N)
    found_cats = _find_categories_in_text(text)
    if found_cats:
        result = call("get_category_stats", {"category": found_cats[0]})
        if not result.get("found"):
            return f"Không có dữ liệu cho ngành hàng '{found_cats[0]}'.", tool_calls_log
        return (
            f"Thống kê ngành hàng {found_cats[0]}: {result['n_orders']} đơn, "
            f"tỉ lệ trễ {result['late_rate_pct']}%, TB {result['avg_delivery_days']} ngày giao, "
            f"phí ship TB {result['avg_freight']}."
        ), tool_calls_log

    # 16) Thống kê 1 bang cụ thể (không match các nhánh trên)
    states = _extract_states_in_order(text)
    if states:
        state = states[0]
        result = call("get_state_stats", {"state": state})
        if not result.get("found"):
            return f"Không có dữ liệu cho bang '{state}'.", tool_calls_log
        return (
            f"Thống kê bang {state}: {result['n_orders']} đơn, tỉ lệ trễ {result['late_rate_pct']}%, "
            f"TB {result['avg_delivery_days']} ngày giao, khoảng cách TB {result['avg_distance_km']}km."
        ), tool_calls_log

    # 17) Mô tả cấu trúc dataset
    if any(k in lower for k in ["cột", "cot", "schema", "describe", "mô tả dữ liệu", "mo ta du lieu"]):
        result = call("describe_dataset", {})
        cols = ", ".join(c["name"] for c in result.get("columns", [])[:15])
        return f"Dataset có {result.get('n_rows')} dòng, {len(result.get('columns', []))} cột. Một số cột: {cols}...", tool_calls_log

    # Không khớp mẫu nào — hướng dẫn người dùng
    return (
        "Mình đang chạy ở chế độ **Local** (rule-based, miễn phí, không gọi API) nên chỉ nhận "
        "diện được vài mẫu câu cố định, chưa hiểu ngôn ngữ tự nhiên linh hoạt như Gemini/GPT/Grok. "
        "Bạn có thể thử các mẫu câu sau:\n"
        "- \"Top 5 bang trễ nhất\"\n"
        "- \"Ngành hàng nào trễ nhiều nhất\"\n"
        "- \"Thống kê bang RJ\"\n"
        "- \"Dự đoán đơn hàng khách RJ seller SP cách 900km\"\n"
        "- \"Tra cứu đơn hàng <order_id>\"\n\n"
        "Muốn hỏi tự do, phức tạp hơn, hãy đổi sang model Gemini/GPT/Grok ở dropdown khi có quota."
    ), tool_calls_log
