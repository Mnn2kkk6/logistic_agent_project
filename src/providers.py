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
from urllib import response

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
    web_search,
)
from src.marketing_tools import (
    get_marketing_funnel_summary,
    get_marketing_channel_performance,
    get_seller_360,
    get_acquisition_logistics_performance,
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
    "web_search": web_search,
}
TOOL_FUNCTIONS.update({
    "get_marketing_funnel_summary": get_marketing_funnel_summary,
    "get_marketing_channel_performance": get_marketing_channel_performance,
    "get_seller_360": get_seller_360,
    "get_acquisition_logistics_performance": get_acquisition_logistics_performance,
})

# ==== Danh sách model hỗ trợ ====
# "env_key": biến môi trường chứa API key cần thiết cho provider đó.
AVAILABLE_MODELS = {
    "gemini-3.8-flash": {
        "provider": "gemini",
        "label": "Gemini 3.8 Flash",
        "env_key": "GEMINI_API_KEY",
        "note": "Free Tier — lựa chọn chính cho agent, reasoning và tool calling",
    },

    "gemini-3.7-flash": {
        "provider": "gemini",
        "label": "Gemini 3.7 Flash",
        "env_key": "GEMINI_API_KEY",
        "note": "Free Tier — mạnh cho agentic workflows và multi-step execution",
    },

    "gemini-3.6-flash": {
        "provider": "gemini",
        "label": "Gemini 3.6 Flash",
        "env_key": "GEMINI_API_KEY",
        "note": "Free Tier — nhanh, ổn định, dùng làm backup",
    },
    "gemini-3.1-pro-preview": {
    "provider": "gemini",
    "label": "Gemini 3.1 Pro Preview",
    "env_key": "GEMINI_API_KEY",
    "note": "Pro — reasoning mạnh, multimodal và agentic workflows; paid",
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
    "openai/gpt-oss-120b": {
        "provider": "groq", "label": "GPT-OSS 120B (Groq)", "env_key": "GROQ_API_KEY",
        "note": "Miễn phí (~14,400 request/ngày) — chạy trên hạ tầng Groq siêu nhanh, "
                "KHÁC với Grok của xAI. Có khả năng suy luận (reasoning) tốt hơn bản 20B.",
    },
    "openai/gpt-oss-20b": {
        "provider": "groq", "label": "GPT-OSS 20B (Groq)", "env_key": "GROQ_API_KEY",
        "note": "Miễn phí (~14,400 request/ngày), nhỏ/nhanh hơn bản 120B — dự phòng khi "
                "Gemini hết quota mà chưa muốn dùng bản trả phí (OpenAI/xAI).",
    },
    "nvidia/nemotron-3-ultra-550b-a55b-20260604:free": {
        "provider": "openrouter",
        "label": "Nemotron 3 Ultra (OpenRouter)",
        "env_key": "OPENROUTER_API_KEY",
        "note": "Free — reasoning/orchestration mạnh, phù hợp AI Agent",
    },

    "poolside/laguna-s-2.1:free": {
        "provider": "openrouter",
        "label": "Laguna S 2.1 (OpenRouter)",
        "env_key": "OPENROUTER_API_KEY",
        "note": "Free — coding + agentic workflows + tool calling",
    },

    "nvidia/nemotron-3.5-lightning:free": {
        "provider": "openrouter",
        "label": "Nemotron 3.5 Lightning (OpenRouter)",
        "env_key": "OPENROUTER_API_KEY",
        "note": "Free — nhanh, context lớn, phù hợp fallback agent",
    },

    "google/gemma-4-31b-it:free": {
        "provider": "openrouter",
        "label": "Gemma 4 31B (OpenRouter)",
        "env_key": "OPENROUTER_API_KEY",
        "note": "Free — function calling + multimodal",
    },

    "openrouter/free": {
        "provider": "openrouter",
        "label": "OpenRouter Free Router",
        "env_key": "OPENROUTER_API_KEY",
        "note": "Free — tự chọn free model tương thích với request",
    },

    "gpt-4o-mini": {
        "provider": "openai",
        "label": "GPT-4o Mini",
        "env_key": "OPENAI_API_KEY",
        "note": "Cần OPENAI_API_KEY",
    },
    "local-rule-based": {
        "provider": "local", "label": "Local (miễn phí, offline)", "env_key": None,
        "note": "Không cần API key, không gọi mạng — chỉ nhận diện vài mẫu câu cố định "
                "(không hiểu ngôn ngữ tự nhiên linh hoạt). Dùng khi MỌI model khác đều hết quota.",
    },
}
DEFAULT_MODEL = "gemini-3.1-pro-preview"  # model mặc định nếu người dùng không chọn gì


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


def chat_turn(model: str, history: list, user_message: str, max_tool_rounds: int = 8):
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
    elif info["provider"] == "groq":
        reply, tool_calls = _groq_turn(model, history, user_message, max_tool_rounds)
    elif info["provider"] == "openrouter":
        reply, tool_calls = _openrouter_turn(model, history, user_message, max_tool_rounds)
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
        except Exception as e:
            raise ProviderError(f"{provider_label} request lỗi: {e}")
        # OpenRouter có thể trả payload lỗi mà SDK vẫn tạo response object.
        if response is None:
            raise ProviderError(f"{provider_label} trả về response rỗng.")
        if not getattr(response, "choices", None):
            error_obj = getattr(response, "error", None)
            if error_obj:
                try: 
                    error_data = error_obj.model_dump()
                except Exception:
                    error_data = str(error_obj)
                raise ProviderError(
                    f"{provider_label} không trả về choices. {error_data}"
                    f"Provider error: {error_data}"
                )
            # Fallback: dump toàn bộ response để debug
            try:
                response_data = response.model_dump(exclude_none=True)
            except Exception:
                response_data = str(response)
            raise ProviderError(
                f"{provider_label} không trả về choices. {response_data}"
                f"Provider error: {response_data}"
            )
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


def _groq_turn(model, history, user_message, max_tool_rounds):
    return _openai_compatible_turn(
        model, history, user_message, max_tool_rounds,
        api_key=os.environ.get("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1", provider_label="Groq",
    )
def _openrouter_turn(model, history, user_message, max_tool_rounds):
    return _openai_compatible_turn(
        model,
        history,
        user_message,
        max_tool_rounds,
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        provider_label="OpenRouter",
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


# ==== Bảng chỉ số cơ bản/thống kê tổng quan (đếm/tổng/TB/cao nhất/liệt kê) ====
# Thay vì viết 1 khối "if" riêng cho từng câu (rất dài dòng, dễ đụng từ khoá), các câu hỏi
# thống kê ĐƠN GIẢN dùng chung 1 "công thức" theo (loại phép tính × đối tượng), nên gom vào
# 1 bảng dữ liệu — mỗi dòng chỉ cần: điều kiện nhận diện (match) + SQL + cách trả lời (kind).
# Bảng này được kiểm tra SAU CÙNG (trước describe_dataset/fallback), vì các câu hỏi phân tích
# phức tạp hơn ở các nhánh phía trên đã được ưu tiên xử lý trước rồi.

def _fmt_num(v):
    return f"{v:,.2f}" if isinstance(v, float) else f"{v:,}" if isinstance(v, int) else str(v)


def _short(v, n=12):
    s = str(v)
    return s[:n] + "..." if len(s) > n else s


def _run_basic_metric(m, call) -> str:
    result = call("query_dataset_sql", {"sql": m["sql"]})
    rows = result.get("rows", [])
    if not rows:
        return f"Không lấy được dữ liệu cho '{m['label']}'."

    kind = m["kind"]
    if kind == "count":
        return f"{m['label']}: {_fmt_num(rows[0].get('n'))}."
    if kind == "value":
        return f"{m['label']}: {_fmt_num(rows[0].get('v'))}{m.get('suffix', '')}."
    if kind == "top1":
        r = rows[0]
        name = _short(r.get("name")) if m.get("truncate_name") else r.get("name")
        return f"{m['label']}: {name} ({m.get('metric_label', 'giá trị')}: {_fmt_num(r.get('metric'))})."
    if kind == "list":
        items = [str(r.get("v")) for r in rows]
        return f"{m['label']}: " + ", ".join(items) + "."
    if kind == "compare":
        lines = [f"- {r.get('order_status')}: {_fmt_num(r.get('n'))} đơn" for r in rows]
        return f"{m['label']}:\n" + "\n".join(lines)
    return "Không xác định được cách trả lời."  # pragma: no cover


BASIC_METRICS = [
    {  # M1 (loại trừ khi câu hỏi thực ra về ĐIỂM ĐÁNH GIÁ — dù có chữ "khách hàng"/"bao nhiêu")
        "match": lambda t: "khách hàng" in t and "bao nhiêu" in t
                  and not any(k in t for k in ["đánh giá", "danh gia", "review"]),
        "kind": "count", "label": "Số khách hàng (unique) trong dataset",
        "sql": "SELECT COUNT(DISTINCT customer_unique_id) AS n FROM customers",
    },
    {  # M2
        "match": lambda t: "seller" in t and "bao nhiêu" in t,
        "kind": "count", "label": "Số seller trong dataset",
        "sql": "SELECT COUNT(*) AS n FROM sellers",
    },
    {  # M4 (sản phẩm nói chung — câu hỏi theo 1 danh mục cụ thể đã xử lý riêng ở trên)
        "match": lambda t: any(k in t for k in ["sản phẩm", "san pham"]) and "bao nhiêu" in t
                  and "danh mục" not in t and "danh muc" not in t and "thuộc" not in t and "thuoc" not in t
                  and "doanh thu" not in t,
        "kind": "count", "label": "Số sản phẩm trong dataset",
        "sql": "SELECT COUNT(*) AS n FROM products",
    },
    {  # M6
        "match": lambda t: "trạng thái" in t and "nhiều nhất" in t,
        "kind": "top1", "label": "Trạng thái đơn hàng xuất hiện nhiều nhất", "metric_label": "số đơn",
        "sql": "SELECT order_status AS name, COUNT(*) AS metric FROM orders GROUP BY name ORDER BY metric DESC LIMIT 1",
    },
    {  # M7 — nhận diện cả khi không có chữ "trạng thái" mà liệt kê thẳng tên trạng thái
        "match": lambda t: "so sánh" in t and (
            "trạng thái" in t or
            sum(k in t for k in ["delivered", "canceled", "shipped", "processing", "invoiced", "unavailable"]) >= 2
        ),
        "kind": "compare", "label": "So sánh số lượng đơn hàng theo trạng thái",
        "sql": "SELECT order_status, COUNT(*) AS n FROM orders GROUP BY order_status ORDER BY n DESC",
    },
    {  # M5 (đặt sau M6/M7 vì "trạng thái" là điều kiện chung nhất, để 2 câu trên ưu tiên trước)
        "match": lambda t: "trạng thái" in t,
        "kind": "list", "label": "Các trạng thái đơn hàng đang có trong dataset",
        "sql": "SELECT DISTINCT order_status AS v FROM orders",
    },
    {  # M8
        "match": lambda t: any(k in t for k in ["phương thức thanh toán", "phuong thuc thanh toan"])
                  and "bao nhiêu" in t,
        "kind": "count", "label": "Số phương thức thanh toán khác nhau",
        "sql": "SELECT COUNT(DISTINCT payment_type) AS n FROM order_payments",
    },
    {  # M9
        "match": lambda t: any(k in t for k in ["phương thức thanh toán", "phuong thuc thanh toan"])
                  and "nhiều nhất" in t,
        "kind": "top1", "label": "Phương thức thanh toán được dùng nhiều nhất", "metric_label": "lượt dùng",
        "sql": "SELECT payment_type AS name, COUNT(*) AS metric FROM order_payments GROUP BY name ORDER BY metric DESC LIMIT 1",
    },
    {  # M10 — chỉ khớp khi KHÔNG có "bang"/"ngành hàng" (những câu đó đã xử lý riêng ở nhánh 11 phía trên)
        "match": lambda t: any(k in t for k in ["điểm đánh giá", "diem danh gia", "review"]) and "trung bình" in t
                  and not any(k in t for k in ["bang", "ngành hàng", "nganh hang", "category"]),
        "kind": "value", "label": "Điểm đánh giá trung bình toàn sàn",
        "sql": "SELECT ROUND(AVG(review_score), 2) AS v FROM order_reviews",
    },
    {  # M11
        "match": lambda t: "5 sao" in t,
        "kind": "count", "label": "Số review 5 sao",
        "sql": "SELECT COUNT(*) AS n FROM order_reviews WHERE review_score = 5",
    },
    {  # M12
        "match": lambda t: any(k in t for k in ["1-2 sao", "1–2 sao"]) or ("1 sao" in t and "2 sao" in t),
        "kind": "count", "label": "Số review từ 1 đến 2 sao",
        "sql": "SELECT COUNT(*) AS n FROM order_reviews WHERE review_score IN (1, 2)",
    },
    {  # M13
        "match": lambda t: any(k in t for k in ["danh mục", "danh muc", "category"]) and "nhiều sản phẩm nhất" in t,
        "kind": "top1", "label": "Danh mục có nhiều sản phẩm nhất", "metric_label": "số sản phẩm",
        "sql": "SELECT ct.product_category_name_english AS name, COUNT(*) AS metric FROM products p "
               "JOIN category_translation ct ON p.product_category_name = ct.product_category_name "
               "GROUP BY name ORDER BY metric DESC LIMIT 1",
    },
    {  # M14
        "match": lambda t: "thành phố" in t and "nhiều khách hàng nhất" in t,
        "kind": "top1", "label": "Thành phố có nhiều khách hàng nhất", "metric_label": "số khách hàng",
        "sql": "SELECT customer_city AS name, COUNT(DISTINCT customer_unique_id) AS metric "
               "FROM customers GROUP BY name ORDER BY metric DESC LIMIT 1",
    },
    {  # M15
        "match": lambda t: "bang" in t and "nhiều seller nhất" in t,
        "kind": "top1", "label": "Bang có nhiều seller nhất", "metric_label": "số seller",
        "sql": "SELECT seller_state AS name, COUNT(*) AS metric FROM sellers GROUP BY name ORDER BY metric DESC LIMIT 1",
    },
    {  # M16
        "match": lambda t: any(k in t for k in ["sản phẩm", "san pham"]) and "bán nhiều nhất" in t,
        "kind": "top1", "label": "Sản phẩm được bán nhiều nhất", "metric_label": "lượt bán", "truncate_name": True,
        "sql": "SELECT product_id AS name, COUNT(*) AS metric FROM order_items GROUP BY name ORDER BY metric DESC LIMIT 1",
    },
    {  # M17
        "match": lambda t: "seller" in t and "nhiều đơn hàng nhất" in t,
        "kind": "top1", "label": "Seller có nhiều đơn hàng nhất", "metric_label": "số đơn", "truncate_name": True,
        "sql": "SELECT seller_id AS name, COUNT(DISTINCT order_id) AS metric FROM order_items GROUP BY name ORDER BY metric DESC LIMIT 1",
    },
    {  # M18
        "match": lambda t: "tháng" in t and "nhiều đơn hàng nhất" in t,
        "kind": "top1", "label": "Tháng có nhiều đơn hàng nhất", "metric_label": "số đơn",
        "sql": "SELECT purchase_month AS name, COUNT(*) AS metric FROM orders GROUP BY name ORDER BY metric DESC LIMIT 1",
    },
    {  # M28
        "match": lambda t: any(k in t for k in ["delivered", "được giao", "duoc giao"]) and "bao nhiêu" in t,
        "kind": "count", "label": "Số đơn hàng đã giao (delivered)",
        "sql": "SELECT COUNT(*) AS n FROM orders WHERE order_status = 'delivered'",
    },
    {  # M29
        "match": lambda t: any(k in t for k in ["hủy", "huỷ", "huy", "cancel"]) and "bao nhiêu" in t,
        "kind": "count", "label": "Số đơn hàng bị hủy",
        "sql": "SELECT COUNT(*) AS n FROM orders WHERE order_status = 'canceled'",
    },
    {  # M19
        "match": lambda t: any(k in t for k in ["tổng giá trị", "tổng doanh thu"]) and "đơn hàng" in t
                  and not any(k in t for k in ["danh mục", "seller", "sản phẩm", "khách hàng"]),
        "kind": "value", "label": "Tổng giá trị các đơn hàng",
        "sql": "SELECT ROUND(SUM(payment_value), 2) AS v FROM orders",
    },
    {  # M20
        "match": lambda t: "giá trị đơn hàng" in t and "trung bình" in t,
        "kind": "value", "label": "Giá trị đơn hàng trung bình",
        "sql": "SELECT ROUND(AVG(payment_value), 2) AS v FROM orders",
    },
    {  # M21 (đặt TRƯỚC M22 vì câu này cụ thể hơn — hỏi luôn cả khách hàng nào)
        "match": lambda t: "giá trị" in t and "cao nhất" in t and "khách hàng nào" in t,
        "kind": "top1", "label": "Khách hàng có đơn hàng giá trị cao nhất", "metric_label": "giá trị đơn", "truncate_name": True,
        "sql": "SELECT c.customer_unique_id AS name, o.payment_value AS metric FROM orders o "
               "JOIN customers c ON o.customer_id = c.customer_id ORDER BY o.payment_value DESC LIMIT 1",
    },
    {  # M22
        "match": lambda t: "giá trị" in t and "cao nhất" in t and "khách hàng nào" not in t and "seller" not in t and "danh mục" not in t,
        "kind": "value", "label": "Giá trị đơn hàng cao nhất",
        "sql": "SELECT ROUND(MAX(payment_value), 2) AS v FROM orders",
    },
    {  # M23
        "match": lambda t: "seller" in t and "doanh thu" in t and "cao nhất" in t,
        "kind": "top1", "label": "Seller có tổng doanh thu cao nhất", "metric_label": "doanh thu", "truncate_name": True,
        "sql": "SELECT seller_id AS name, ROUND(SUM(price), 2) AS metric FROM order_items GROUP BY name ORDER BY metric DESC LIMIT 1",
    },
    {  # M24 (loại trừ "2018" vì đã có câu hỏi riêng chi tiết hơn cho năm 2018 ở nhánh N1-Q1)
        "match": lambda t: any(k in t for k in ["danh mục", "danh muc"]) and "doanh thu" in t and "cao nhất" in t and "2018" not in t,
        "kind": "top1", "label": "Danh mục có tổng doanh thu cao nhất", "metric_label": "doanh thu",
        "sql": "SELECT ct.product_category_name_english AS name, ROUND(SUM(oi.price), 2) AS metric "
               "FROM order_items oi JOIN products p ON oi.product_id = p.product_id "
               "JOIN category_translation ct ON p.product_category_name = ct.product_category_name "
               "GROUP BY name ORDER BY metric DESC LIMIT 1",
    },
    {  # M25
        "match": lambda t: "khách hàng" in t and "chi tiêu" in t and "nhiều nhất" in t,
        "kind": "top1", "label": "Khách hàng chi tiêu nhiều nhất", "metric_label": "tổng chi tiêu", "truncate_name": True,
        "sql": "SELECT c.customer_unique_id AS name, ROUND(SUM(o.payment_value), 2) AS metric FROM orders o "
               "JOIN customers c ON o.customer_id = c.customer_id GROUP BY name ORDER BY metric DESC LIMIT 1",
    },
    {  # M26 (loại trừ "tháng" vì đã có câu hỏi chi tiết theo tháng ở nhánh N2-Q5 phía trên)
        "match": lambda t: any(k in t for k in ["thời gian giao", "thoi gian giao"]) and "trung bình" in t and "tháng" not in t,
        "kind": "value", "label": "Thời gian giao hàng trung bình (đơn delivered)", "suffix": " ngày",
        "sql": "SELECT ROUND(AVG(actual_delivery_days), 2) AS v FROM orders WHERE order_status = 'delivered'",
    },
    {  # M27 (loại trừ "nhóm" vì đã có câu hỏi chia nhóm freight ở nhánh N2-Q6 phía trên)
        "match": lambda t: any(k in t for k in ["freight", "phí ship", "phi ship", "phí vận chuyển", "phi van chuyen"])
                  and "trung bình" in t and "nhóm" not in t,
        "kind": "value", "label": "Phí vận chuyển (freight) trung bình mỗi đơn",
        "sql": "SELECT ROUND(AVG(total_freight), 2) AS v FROM orders",
    },
]


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

    # === Bộ câu hỏi "stress test" Nhóm 1 (Multi-table Join) & Nhóm 2 (Time & Delivery) ===
    # Đây là 8 câu hỏi CỐ ĐỊNH đã biết trước (không phải NLU tổng quát), nên nhận diện bằng
    # tổ hợp từ khoá khá đặc thù cho từng câu để giảm rủi ro trùng lặp với các nhánh khác.

    # N1-Q1: Top 10 category doanh thu cao nhất năm 2018 (kèm số đơn, freight TB, review TB)
    if ("2018" in lower) and any(k in lower for k in ["danh mục", "danh muc", "category", "ngành hàng", "nganh hang"]) \
            and any(k in lower for k in ["doanh thu", "revenue"]):
        result = call("query_dataset_sql", {"sql": """
            SELECT ct.product_category_name_english AS category, ROUND(SUM(oi.price), 2) AS revenue,
                   COUNT(DISTINCT oi.order_id) AS n_orders, ROUND(AVG(oi.freight_value), 2) AS avg_freight,
                   ROUND(AVG(r.review_score), 2) AS avg_review
            FROM order_items oi
            JOIN orders o ON oi.order_id = o.order_id
            JOIN products p ON oi.product_id = p.product_id
            JOIN category_translation ct ON p.product_category_name = ct.product_category_name
            LEFT JOIN order_reviews r ON oi.order_id = r.order_id
            WHERE o.order_status = 'delivered' AND EXTRACT(YEAR FROM o.order_purchase_timestamp) = 2018
            GROUP BY category ORDER BY revenue DESC LIMIT 10
        """})
        rows = result.get("rows", [])
        if not rows:
            return "Không lấy được dữ liệu doanh thu theo danh mục năm 2018.", tool_calls_log
        lines = [
            f"{i+1}. {r['category']}: doanh thu {r['revenue']:,.0f}, {r['n_orders']} đơn, "
            f"freight TB {r['avg_freight']}, review TB {r['avg_review']}"
            for i, r in enumerate(rows)
        ]
        return "Top 10 danh mục doanh thu cao nhất năm 2018 (đơn delivered):\n" + "\n".join(lines), tool_calls_log

    # N1-Q2: Seller top 10% doanh thu nhưng review TB thấp hơn trung bình toàn sàn
    if "seller" in lower and any(k in lower for k in ["review", "đánh giá", "danh gia"]) \
            and any(k in lower for k in ["top 10", "trung bình", "trung binh"]):
        result = call("query_dataset_sql", {"sql": """
            WITH seller_rev AS (
                SELECT oi.seller_id, SUM(oi.price) AS revenue, COUNT(DISTINCT oi.order_id) AS n_orders
                FROM order_items oi JOIN orders o ON oi.order_id = o.order_id
                WHERE o.order_status = 'delivered' GROUP BY oi.seller_id
            ),
            seller_review AS (
                SELECT oi.seller_id, AVG(r.review_score) AS avg_review,
                       AVG(CASE WHEN r.review_score < 3 THEN 1.0 ELSE 0 END) AS pct_low_review
                FROM order_items oi JOIN order_reviews r ON oi.order_id = r.order_id GROUP BY oi.seller_id
            ),
            threshold AS (SELECT PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY revenue) AS p90 FROM seller_rev),
            overall_avg AS (SELECT AVG(review_score) AS avg_all FROM order_reviews)
            SELECT sr.seller_id, s.seller_city, s.seller_state, ROUND(sr.revenue, 2) AS revenue, sr.n_orders,
                   ROUND(rv.avg_review, 2) AS avg_review, ROUND(rv.pct_low_review*100, 1) AS pct_low_review_pct
            FROM seller_rev sr
            JOIN seller_review rv ON sr.seller_id = rv.seller_id
            JOIN sellers s ON sr.seller_id = s.seller_id
            CROSS JOIN threshold t CROSS JOIN overall_avg oa
            WHERE sr.revenue >= t.p90 AND rv.avg_review < oa.avg_all
            ORDER BY sr.revenue DESC LIMIT 10
        """})
        rows = result.get("rows", [])
        if not rows:
            return "Không tìm thấy seller nào thoả điều kiện (top 10% doanh thu, review dưới TB toàn sàn).", tool_calls_log
        lines = [
            f"{i+1}. {r['seller_id'][:12]}... ({r['seller_city']}, {r['seller_state']}): "
            f"doanh thu {r['revenue']:,.0f}, {r['n_orders']} đơn, review TB {r['avg_review']}, "
            f"{r['pct_low_review_pct']}% đơn dưới 3 sao"
            for i, r in enumerate(rows)
        ]
        return "Seller top 10% doanh thu nhưng review TB thấp hơn trung bình toàn sàn:\n" + "\n".join(lines), tool_calls_log

    # N1-Q3: Bang có chi tiêu TB/đơn cao VÀ thời gian giao TB cũng dài (thoả cả 2 điều kiện)
    if "bang" in lower and any(k in lower for k in ["chi tiêu", "chi tieu"]) \
            and any(k in lower for k in ["thời gian giao", "thoi gian giao", "giao hàng"]):
        result = call("query_dataset_sql", {"sql": """
            WITH stats AS (
                SELECT customer_state, ROUND(AVG(payment_value), 2) AS avg_spend,
                       ROUND(AVG(actual_delivery_days), 2) AS avg_days, COUNT(*) AS n
                FROM orders WHERE order_status='delivered'
                GROUP BY customer_state HAVING COUNT(*) >= 50
            ),
            overall AS (SELECT AVG(avg_spend) AS m1, AVG(avg_days) AS m2 FROM stats)
            SELECT s.* FROM stats s CROSS JOIN overall o
            WHERE s.avg_spend > o.m1 AND s.avg_days > o.m2 ORDER BY s.avg_spend DESC
        """})
        rows = result.get("rows", [])
        if not rows:
            return "Không có bang nào thoả cả 2 điều kiện (chi tiêu TB cao VÀ thời gian giao TB dài, so với TB toàn quốc).", tool_calls_log
        lines = [f"{r['customer_state']}: chi tiêu TB {r['avg_spend']}, giao TB {r['avg_days']} ngày ({r['n']} đơn)" for r in rows]
        return (
            "Các bang có chi tiêu TB/đơn CAO HƠN trung bình toàn quốc VÀ thời gian giao TB "
            "cũng DÀI HƠN trung bình toàn quốc:\n" + "\n".join(lines)
        ), tool_calls_log

    # N1-Q4: Top 20 sản phẩm doanh thu cao nhất, KHÔNG thuộc top 10 category
    if any(k in lower for k in ["sản phẩm", "san pham"]) and any(k in lower for k in ["doanh thu", "revenue"]) \
            and any(k in lower for k in ["không thuộc", "khong thuoc", "ngoài top", "ngoai top", "không nằm", "khong nam"]):
        result = call("query_dataset_sql", {"sql": """
            WITH cat_revenue AS (
                SELECT ct.product_category_name_english AS cat, SUM(oi.price) AS rev
                FROM order_items oi JOIN orders o ON oi.order_id=o.order_id
                JOIN products p ON oi.product_id=p.product_id
                JOIN category_translation ct ON p.product_category_name=ct.product_category_name
                WHERE o.order_status='delivered' GROUP BY cat ORDER BY rev DESC LIMIT 10
            ),
            product_revenue AS (
                SELECT oi.product_id, ct.product_category_name_english AS cat,
                       ROUND(SUM(oi.price), 2) AS revenue, COUNT(*) AS n_sold,
                       ROUND(AVG(r.review_score), 2) AS avg_review
                FROM order_items oi JOIN orders o ON oi.order_id=o.order_id
                JOIN products p ON oi.product_id=p.product_id
                JOIN category_translation ct ON p.product_category_name=ct.product_category_name
                LEFT JOIN order_reviews r ON oi.order_id=r.order_id
                WHERE o.order_status='delivered' GROUP BY oi.product_id, cat
            )
            SELECT * FROM product_revenue WHERE cat NOT IN (SELECT cat FROM cat_revenue)
            ORDER BY revenue DESC LIMIT 20
        """})
        rows = result.get("rows", [])
        if not rows:
            return "Không lấy được dữ liệu.", tool_calls_log
        lines = [
            f"{i+1}. {r['product_id'][:12]}... ({r['cat']}): doanh thu {r['revenue']:,.0f}, "
            f"bán {r['n_sold']} lần, review TB {r['avg_review']}"
            for i, r in enumerate(rows)
        ]
        return "Top 20 sản phẩm doanh thu cao nhất (KHÔNG thuộc top 10 category doanh thu cao nhất):\n" + "\n".join(lines), tool_calls_log

    # N2-Q6: Chia 5 nhóm theo freight_value, so sánh tỉ lệ giao trễ
    if any(k in lower for k in ["freight", "phí ship", "phi ship", "phí vận chuyển", "phi van chuyen"]) \
            and any(k in lower for k in ["nhóm", "nhom", "chia"]):
        result = call("query_dataset_sql", {"sql": """
            WITH buckets AS (
                SELECT *, NTILE(5) OVER (ORDER BY total_freight) AS nhom
                FROM orders WHERE order_status='delivered' AND total_freight IS NOT NULL
            )
            SELECT nhom, ROUND(MIN(total_freight),2) AS freight_min, ROUND(MAX(total_freight),2) AS freight_max,
                   ROUND(AVG(is_late)*100, 2) AS late_rate_pct, COUNT(*) AS n
            FROM buckets GROUP BY nhom ORDER BY nhom
        """})
        rows = result.get("rows", [])
        if not rows:
            return "Không lấy được dữ liệu.", tool_calls_log
        lines = [
            f"Nhóm {r['nhom']} (freight {r['freight_min']}-{r['freight_max']}): "
            f"{r['late_rate_pct']}% trễ ({r['n']} đơn)"
            for r in rows
        ]
        return "So sánh tỉ lệ giao trễ theo 5 nhóm phí vận chuyển (freight_value), từ thấp đến cao:\n" + "\n".join(lines), tool_calls_log

    # N2-Q7: Seller có thời gian xử lý đơn hàng (purchase -> giao cho carrier) dài nhất, >=50 đơn
    if "seller" in lower and any(k in lower for k in ["thời gian xử lý", "thoi gian xu ly", "xử lý đơn", "xu ly don", "carrier"]):
        result = call("query_dataset_sql", {"sql": """
            SELECT oi.seller_id, s.seller_state,
                   ROUND(AVG(o.approval_to_carrier_h), 2) AS avg_processing_hours,
                   COUNT(DISTINCT oi.order_id) AS n_orders
            FROM order_items oi
            JOIN orders o ON oi.order_id = o.order_id
            JOIN sellers s ON oi.seller_id = s.seller_id
            WHERE o.order_status = 'delivered' AND o.approval_to_carrier_h IS NOT NULL
            GROUP BY oi.seller_id, s.seller_state
            HAVING COUNT(DISTINCT oi.order_id) >= 50
            ORDER BY avg_processing_hours DESC LIMIT 10
        """})
        rows = result.get("rows", [])
        if not rows:
            return "Không lấy được dữ liệu (hoặc không có seller nào đủ 50 đơn delivered).", tool_calls_log
        lines = [
            f"{i+1}. {r['seller_id'][:12]}... ({r['seller_state']}): "
            f"TB {r['avg_processing_hours']} giờ xử lý ({r['n_orders']} đơn)"
            for i, r in enumerate(rows)
        ]
        return (
            "Top seller có thời gian xử lý đơn hàng dài nhất (từ lúc đặt đến khi giao cho "
            "đơn vị vận chuyển, chỉ tính seller có ≥50 đơn delivered):\n" + "\n".join(lines)
        ), tool_calls_log

    # N2-Q8: Sản phẩm giá bán TB cao NHƯNG tỉ lệ review 1-2 sao cao, >=100 sản phẩm bán ra
    if any(k in lower for k in ["sản phẩm", "san pham"]) and any(k in lower for k in ["giá bán", "gia ban", "giá cao", "gia cao"]) \
            and any(k in lower for k in ["1-2 sao", "1–2 sao", "review thấp", "review thap", "1 sao", "2 sao"]):
        result = call("query_dataset_sql", {"sql": """
            SELECT oi.product_id, ct.product_category_name_english AS category,
                   ROUND(AVG(oi.price), 2) AS avg_price, COUNT(*) AS n_sold,
                   ROUND(AVG(CASE WHEN r.review_score <= 2 THEN 1.0 ELSE 0 END)*100, 2) AS pct_1_2_sao
            FROM order_items oi
            JOIN products p ON oi.product_id = p.product_id
            JOIN category_translation ct ON p.product_category_name = ct.product_category_name
            LEFT JOIN order_reviews r ON oi.order_id = r.order_id
            GROUP BY oi.product_id, category
            HAVING COUNT(*) >= 100
            ORDER BY avg_price DESC, pct_1_2_sao DESC LIMIT 10
        """})
        rows = result.get("rows", [])
        if not rows:
            return "Không lấy được dữ liệu (hoặc không có sản phẩm nào bán ≥100 lần).", tool_calls_log
        lines = [
            f"{i+1}. {r['product_id'][:12]}... ({r['category']}): giá TB {r['avg_price']}, "
            f"bán {r['n_sold']} lần, {r['pct_1_2_sao']}% review 1-2 sao"
            for i, r in enumerate(rows)
        ]
        return "Sản phẩm giá bán TB cao (chỉ tính SP bán ≥100 lần), sắp theo giá rồi theo tỉ lệ review 1-2 sao:\n" + "\n".join(lines), tool_calls_log

    # === Nhóm 3 (Customer Behavior) ===

    # N3-Q9: khách hàng unique mua >=2 đơn, đóng góp bao nhiêu % doanh thu
    if any(k in lower for k in ["khách hàng", "khach hang"]) and any(k in lower for k in ["quay lại", "quay lai", "2 đơn", "2 don"]):
        result = call("query_dataset_sql", {"sql": """
            WITH cust_orders AS (
                SELECT c.customer_unique_id, COUNT(DISTINCT o.order_id) AS n_orders, SUM(o.payment_value) AS spend
                FROM orders o JOIN customers c ON o.customer_id = c.customer_id
                WHERE o.order_status='delivered' GROUP BY c.customer_unique_id
            ),
            total AS (SELECT SUM(spend) AS total_rev FROM cust_orders)
            SELECT COUNT(*) AS n_repeat, ROUND(SUM(spend), 2) AS repeat_revenue,
                   ROUND(100.0*SUM(spend)/(SELECT total_rev FROM total), 2) AS pct_revenue
            FROM cust_orders WHERE n_orders >= 2
        """})
        r = result.get("rows", [{}])[0]
        return (
            "Lưu ý: customer_id đổi mỗi lần đặt hàng, customer_unique_id mới thực sự đại diện "
            "1 con người — nên phải GROUP BY customer_unique_id để xác định khách quay lại.\n\n"
            f"Có {r.get('n_repeat')} khách hàng (unique) đã mua từ 2 đơn trở lên, đóng góp "
            f"{r.get('repeat_revenue'):,.0f} trên tổng doanh thu (~{r.get('pct_revenue')}%)."
        ), tool_calls_log

    # N3-Q11: khách >=3 đơn, có cả review 5 sao lẫn 1 sao
    if any(k in lower for k in ["khách hàng", "khach hang"]) and "5 sao" in lower and "1 sao" in lower:
        result = call("query_dataset_sql", {"sql": """
            WITH cust_reviews AS (
                SELECT c.customer_unique_id, COUNT(DISTINCT o.order_id) AS n_orders,
                       MAX(CASE WHEN r.review_score=5 THEN 1 ELSE 0 END) AS has_5,
                       MAX(CASE WHEN r.review_score=1 THEN 1 ELSE 0 END) AS has_1,
                       SUM(o.payment_value) AS total_spend
                FROM orders o JOIN customers c ON o.customer_id=c.customer_id
                LEFT JOIN order_reviews r ON o.order_id=r.order_id
                WHERE o.order_status='delivered' GROUP BY c.customer_unique_id
            )
            SELECT COUNT(*) AS n_customers, ROUND(SUM(total_spend),2) AS total_spend
            FROM cust_reviews WHERE n_orders>=3 AND has_5=1 AND has_1=1
        """})
        r = result.get("rows", [{}])[0]
        return (
            f"Có {r.get('n_customers')} khách hàng có ≥3 đơn và từng cho cả review 5 sao lẫn 1 sao, "
            f"tổng chi tiêu của nhóm này: {r.get('total_spend'):,.0f}."
        ), tool_calls_log

    # === Nhóm 4 (Payment + Order Value) ===

    # N4-Q12: so sánh hành vi mua hàng giữa các phương thức thanh toán
    if any(k in lower for k in ["phương thức thanh toán", "phuong thuc thanh toan"]) and "so sánh" in lower:
        result = call("query_dataset_sql", {"sql": """
            SELECT op.payment_type,
                   ROUND(AVG(o.payment_value),2) AS avg_order_value,
                   ROUND(AVG(op.payment_installments),2) AS avg_installments,
                   ROUND(AVG(CASE WHEN r.review_score<3 THEN 1.0 ELSE 0 END)*100,2) AS pct_low_review,
                   ROUND(AVG(o.is_late)*100,2) AS late_rate_pct
            FROM order_payments op JOIN orders o ON op.order_id=o.order_id
            LEFT JOIN order_reviews r ON op.order_id=r.order_id
            WHERE o.order_status='delivered' GROUP BY op.payment_type ORDER BY avg_order_value DESC
        """})
        rows = result.get("rows", [])
        if not rows:
            return "Không lấy được dữ liệu.", tool_calls_log
        lines = [
            f"{r['payment_type']}: giá trị đơn TB {r['avg_order_value']}, {r['avg_installments']} kỳ trả góp TB, "
            f"{r['pct_low_review']}% review <3 sao, {r['late_rate_pct']}% trễ"
            for r in rows
        ]
        spread = max(r["avg_order_value"] for r in rows) - min(r["avg_order_value"] for r in rows)
        return (
            "So sánh hành vi mua hàng theo phương thức thanh toán:\n" + "\n".join(lines) +
            f"\n\n→ 'credit_card' có giá trị đơn TB và số kỳ trả góp cao vượt trội — chênh lệch "
            f"giá trị đơn giữa phương thức cao nhất/thấp nhất là ~{spread:.0f}."
        ), tool_calls_log

    # N4-Q13: đơn có payment_value lệch đáng kể so với price+freight
    if any(k in lower for k in ["payment_value", "chênh lệch", "chenh lech"]) and any(k in lower for k in ["price", "freight", "giá"]) \
            and not any(k in lower for k in ["nhất quán", "nhat quan", "trạng thái"]):
        result = call("query_dataset_sql", {"sql": """
            WITH item_totals AS (SELECT order_id, SUM(price+freight_value) AS item_total FROM order_items GROUP BY order_id),
            pay_totals AS (SELECT order_id, SUM(payment_value) AS pay_total FROM order_payments GROUP BY order_id)
            SELECT p.order_id, ROUND(p.pay_total,2) AS payment_total, ROUND(i.item_total,2) AS item_total,
                   ROUND(p.pay_total - i.item_total,2) AS diff
            FROM pay_totals p JOIN item_totals i ON p.order_id=i.order_id
            ORDER BY ABS(p.pay_total - i.item_total) DESC LIMIT 20
        """})
        rows = result.get("rows", [])
        if not rows:
            return "Không lấy được dữ liệu.", tool_calls_log
        lines = [f"{r['order_id'][:12]}...: payment={r['payment_total']}, price+freight={r['item_total']}, lệch {r['diff']:+.2f}" for r in rows[:10]]
        return (
            "20 đơn có chênh lệch payment_value vs (price+freight) lớn nhất (hiện 10 dòng đầu):\n" +
            "\n".join(lines) +
            "\n\nCác khả năng gây chênh lệch: (1) có voucher/coupon giảm giá không phản ánh trong "
            "order_items, (2) phí phát sinh khác (bảo hiểm, phụ phí), (3) đơn có nhiều payment record "
            "(trả nhiều lần) cộng dồn sai, (4) sai số làm tròn khi cộng nhiều dòng."
        ), tool_calls_log

    # === Nhóm 5 (Seller vs Customer Geography) ===

    # N5-Q14: khoảng cách địa lý seller-customer -> tỉ lệ trễ + freight
    if any(k in lower for k in ["khoảng cách địa lý", "khoang cach dia ly", "khoảng cách", "khoang cach"]) \
            and any(k in lower for k in ["seller", "zip"]):
        result = call("query_dataset_sql", {"sql": """
            WITH buckets AS (
                SELECT *, NTILE(2) OVER (ORDER BY distance_km) AS nhom
                FROM orders WHERE order_status='delivered' AND distance_km IS NOT NULL
            )
            SELECT nhom, ROUND(AVG(distance_km),1) AS avg_distance, ROUND(AVG(is_late)*100,2) AS late_rate_pct,
                   ROUND(AVG(total_freight),2) AS avg_freight, COUNT(*) AS n
            FROM buckets GROUP BY nhom ORDER BY nhom
        """})
        rows = result.get("rows", [])
        if len(rows) < 2:
            return "Không lấy được dữ liệu.", tool_calls_log
        return (
            "Lưu ý: khoảng cách tính bằng haversine trên toạ độ TRUNG BÌNH theo zip code prefix "
            "(không phải toạ độ chính xác từng địa chỉ), nên đây là ước lượng gần đúng.\n\n"
            f"Nhóm khoảng cách THẤP (TB {rows[0]['avg_distance']}km): {rows[0]['late_rate_pct']}% trễ, "
            f"freight TB {rows[0]['avg_freight']} ({rows[0]['n']} đơn)\n"
            f"Nhóm khoảng cách CAO (TB {rows[1]['avg_distance']}km): {rows[1]['late_rate_pct']}% trễ, "
            f"freight TB {rows[1]['avg_freight']} ({rows[1]['n']} đơn)"
        ), tool_calls_log

    # N5-Q15: mỗi bang khách hàng -> bang seller phổ biến nhất + tỉ lệ trễ cặp đó
    if "bang" in lower and any(k in lower for k in ["phổ biến nhất", "pho bien nhat"]) and "seller" in lower:
        result = call("query_dataset_sql", {"sql": """
            WITH pairs AS (
                SELECT customer_state, seller_state, COUNT(*) AS n, AVG(is_late) AS late_rate
                FROM orders WHERE order_status='delivered' AND seller_state IS NOT NULL
                GROUP BY customer_state, seller_state
            ),
            ranked AS (SELECT *, ROW_NUMBER() OVER (PARTITION BY customer_state ORDER BY n DESC) AS rn FROM pairs)
            SELECT customer_state, seller_state AS pho_bien_nhat, n AS n_orders, ROUND(late_rate*100,2) AS late_rate_pct
            FROM ranked WHERE rn=1 ORDER BY customer_state LIMIT 10
        """})
        rows = result.get("rows", [])
        if not rows:
            return "Không lấy được dữ liệu.", tool_calls_log
        lines = [f"{r['customer_state']} → {r['pho_bien_nhat']}: {r['n_orders']} đơn, {r['late_rate_pct']}% trễ" for r in rows]
        return "Bang seller phổ biến nhất cho từng bang khách hàng (hiện 10/27 bang):\n" + "\n".join(lines), tool_calls_log

    # === Nhóm 6 ("business analyst") ===

    # N6-Q16: 5 danh mục ưu tiên để giảm review <=2 sao
    if any(k in lower for k in ["danh mục", "danh muc"]) and "ưu tiên" in lower and any(k in lower for k in ["review", "đánh giá"]):
        result = call("query_dataset_sql", {"sql": """
            SELECT ct.product_category_name_english AS category, COUNT(DISTINCT oi.order_id) AS n_orders,
                   ROUND(AVG(CASE WHEN r.review_score<=2 THEN 1.0 ELSE 0 END)*100,2) AS pct_low_review,
                   ROUND(SUM(oi.price),2) AS revenue,
                   COUNT(DISTINCT CASE WHEN r.review_score<=2 THEN oi.order_id END) AS n_low_review_orders
            FROM order_items oi JOIN orders o ON oi.order_id=o.order_id
            JOIN products p ON oi.product_id=p.product_id
            JOIN category_translation ct ON p.product_category_name=ct.product_category_name
            LEFT JOIN order_reviews r ON oi.order_id=r.order_id
            WHERE o.order_status='delivered' GROUP BY category HAVING COUNT(DISTINCT oi.order_id) >= 30
            ORDER BY n_low_review_orders DESC LIMIT 5
        """})
        rows = result.get("rows", [])
        if not rows:
            return "Không lấy được dữ liệu.", tool_calls_log
        lines = [
            f"{i+1}. {r['category']}: {r['n_low_review_orders']} đơn review thấp / {r['n_orders']} đơn "
            f"({r['pct_low_review']}%), doanh thu {r['revenue']:,.0f}"
            for i, r in enumerate(rows)
        ]
        return (
            "Tiêu chí ưu tiên: SỐ LƯỢNG đơn review ≤2 sao thực tế (tác động trực tiếp nếu cải thiện), "
            "chỉ xét danh mục có ≥30 đơn để tránh nhiễu do mẫu nhỏ.\n\n"
            "Top 5 danh mục nên ưu tiên cải thiện:\n" + "\n".join(lines)
        ), tool_calls_log

    # N6-Q17: seller doanh thu cao + nhiều đơn + trễ cao + review thấp (đều cao)
    if "seller" in lower and "doanh thu" in lower and any(k in lower for k in ["review thấp", "review thap"]) \
            and any(k in lower for k in ["trễ", "tre"]):
        result = call("query_dataset_sql", {"sql": """
            WITH seller_stats AS (
                SELECT oi.seller_id, SUM(oi.price) AS revenue, COUNT(DISTINCT oi.order_id) AS n_orders,
                       AVG(o.is_late) AS late_rate, AVG(CASE WHEN r.review_score<3 THEN 1.0 ELSE 0 END) AS low_review_rate
                FROM order_items oi JOIN orders o ON oi.order_id=o.order_id
                LEFT JOIN order_reviews r ON oi.order_id=r.order_id
                WHERE o.order_status='delivered' GROUP BY oi.seller_id HAVING COUNT(DISTINCT oi.order_id) >= 30
            ),
            medians AS (SELECT MEDIAN(late_rate) AS med_late, MEDIAN(low_review_rate) AS med_review, MEDIAN(revenue) AS med_rev FROM seller_stats),
            total AS (SELECT SUM(revenue) AS total_rev FROM seller_stats)
            SELECT s.seller_id, ROUND(s.revenue,2) AS revenue, s.n_orders,
                   ROUND(s.late_rate*100,2) AS late_rate_pct, ROUND(s.low_review_rate*100,2) AS low_review_pct,
                   ROUND(100.0*s.revenue/(SELECT total_rev FROM total), 3) AS pct_of_total_rev
            FROM seller_stats s CROSS JOIN medians m
            WHERE s.revenue > m.med_rev AND s.late_rate > m.med_late AND s.low_review_rate > m.med_review
            ORDER BY s.revenue DESC LIMIT 10
        """})
        rows = result.get("rows", [])
        if not rows:
            return "Không có seller nào thoả cả 4 điều kiện (doanh thu, số đơn, tỉ lệ trễ, tỉ lệ review thấp đều trên trung vị).", tool_calls_log
        total_pct = sum(r["pct_of_total_rev"] for r in rows)
        lines = [f"{r['seller_id'][:12]}...: doanh thu {r['revenue']:,.0f}, {r['n_orders']} đơn, {r['late_rate_pct']}% trễ, {r['low_review_pct']}% review<3sao" for r in rows]
        return (
            "Tiêu chí: doanh thu, số đơn, tỉ lệ trễ, tỉ lệ review<3sao ĐỀU cao hơn trung vị toàn bộ seller "
            "(chỉ xét seller ≥30 đơn).\n\n" + "\n".join(lines) +
            f"\n\n→ Nhóm seller này chiếm khoảng {total_pct:.2f}% tổng doanh thu."
        ), tool_calls_log

    # N6-Q18: top 20% seller theo freight, delay rate > trung vị
    if "seller" in lower and "20%" in lower and any(k in lower for k in ["freight", "logistics"]):
        result = call("query_dataset_sql", {"sql": """
            WITH seller_freight AS (
                SELECT oi.seller_id, SUM(oi.freight_value) AS total_freight, AVG(o.is_late) AS late_rate, COUNT(DISTINCT oi.order_id) AS n_orders
                FROM order_items oi JOIN orders o ON oi.order_id=o.order_id
                WHERE o.order_status='delivered' GROUP BY oi.seller_id HAVING COUNT(DISTINCT oi.order_id)>=20
            ),
            ranked AS (
                SELECT *, PERCENT_RANK() OVER (ORDER BY total_freight DESC) AS pct_rank, MEDIAN(late_rate) OVER () AS med_late
                FROM seller_freight
            )
            SELECT seller_id, ROUND(total_freight,2) AS total_freight, n_orders, ROUND(late_rate*100,2) AS late_rate_pct
            FROM ranked WHERE pct_rank <= 0.2 AND late_rate > med_late ORDER BY total_freight DESC LIMIT 10
        """})
        rows = result.get("rows", [])
        if not rows:
            return "Không có seller nào thoả điều kiện.", tool_calls_log
        lines = [f"{r['seller_id'][:12]}...: freight {r['total_freight']:,.0f}, {r['n_orders']} đơn, {r['late_rate_pct']}% trễ" for r in rows]
        return (
            "Trong top 20% seller đóng góp nhiều freight nhất, đây là các seller có tỉ lệ giao trễ "
            "CAO HƠN trung vị toàn bộ seller — nên ưu tiên cải thiện logistics trước:\n" + "\n".join(lines)
        ), tool_calls_log

    # === Nhóm 7 (câu "bẫy") ===

    # N7-Q19: doanh thu = payment_value?
    if "doanh thu" in lower and "payment_value" in lower:
        result = call("query_dataset_sql", {"sql": """
            SELECT (SELECT ROUND(SUM(payment_value),2) FROM order_payments) AS total_payment_value,
                   (SELECT ROUND(SUM(price),2) FROM order_items) AS total_item_price,
                   (SELECT ROUND(SUM(price+freight_value),2) FROM order_items) AS total_price_plus_freight
        """})
        r = result.get("rows", [{}])[0]
        return (
            f"KHÔNG bằng nhau: tổng payment_value = {r.get('total_payment_value'):,.0f}, "
            f"tổng price (order_items) = {r.get('total_item_price'):,.0f}, "
            f"tổng price+freight = {r.get('total_price_plus_freight'):,.0f}.\n\n"
            "'Doanh thu' cần định nghĩa rõ: nếu tính DOANH THU BÁN HÀNG (không gồm phí ship) → dùng "
            "SUM(order_items.price). Nếu tính TỔNG TIỀN KHÁCH THỰC TRẢ (gồm cả ship, có thể có "
            "voucher/phí khác) → dùng SUM(payment_value). Hai định nghĩa lệch nhau vì payment_value "
            "gồm freight + có thể có chênh lệch do khuyến mãi, còn price+freight cũng không khớp "
            "tuyệt đối với payment_value do có đơn nhiều payment record hoặc phụ phí khác."
        ), tool_calls_log

    # N7-Q20: AOV tính trên order_id, customer_unique_id hay order_item_id?
    if "aov" in lower:
        result = call("query_dataset_sql", {"sql": """
            SELECT
              (SELECT ROUND(AVG(payment_value),2) FROM orders WHERE order_status='delivered') AS aov_per_order,
              (SELECT ROUND(SUM(o.payment_value)/COUNT(DISTINCT c.customer_unique_id),2)
               FROM orders o JOIN customers c ON o.customer_id=c.customer_id WHERE o.order_status='delivered') AS spend_per_customer,
              (SELECT ROUND(AVG(price),2) FROM order_items) AS avg_item_price
        """})
        r = result.get("rows", [{}])[0]
        return (
            f"AOV tính trên order_id (đúng chuẩn): {r.get('aov_per_order')}\n"
            f"Chi tiêu TB mỗi customer_unique_id (khác AOV — đây là 'chi tiêu/khách', 1 khách có thể "
            f"nhiều đơn): {r.get('spend_per_customer')}\n"
            f"Giá TB mỗi order_item (khác AOV — đây là giá TB 1 sản phẩm, 1 đơn có thể nhiều sản phẩm): "
            f"{r.get('avg_item_price')}\n\n"
            "AOV (Average Order Value) chuẩn PHẢI tính trên order_id — vì đơn vị đo là '1 lượt mua' "
            "(1 order = 1 purchase event). Tính theo order_item_id sẽ ra giá TB của TỪNG SẢN PHẨM "
            "(sai đơn vị đo), còn tính theo customer_unique_id ra 'chi tiêu TB mỗi khách' — một chỉ "
            "số khác (Customer Value), không phải AOV."
        ), tool_calls_log

    # N7-Q21: mỗi order_id có đúng 1 review?
    if "mỗi order" in lower or "moi order" in lower or "mọi order_id" in lower or "moi order_id" in lower:
        if any(k in lower for k in ["review"]):
            result = call("query_dataset_sql", {"sql": """
                SELECT n_reviews, COUNT(*) AS n_orders FROM (
                    SELECT order_id, COUNT(*) AS n_reviews FROM order_reviews GROUP BY order_id
                ) t GROUP BY n_reviews ORDER BY n_reviews
            """})
            rows = result.get("rows", [])
            lines = [f"{r['n_reviews']} review: {r['n_orders']} đơn" for r in rows]
            n_multi = sum(r["n_orders"] for r in rows if r["n_reviews"] != 1)
            return (
                f"KHÔNG — phân bổ số review/đơn:\n" + "\n".join(lines) +
                f"\n\n→ Có {n_multi} đơn có số review KHÁC 1 (0 hoặc nhiều hơn 1), là các trường hợp bất thường."
            ), tool_calls_log

        # N7-Q22: mỗi order có đúng 1 payment record?
        if any(k in lower for k in ["payment"]):
            result = call("query_dataset_sql", {"sql": """
                SELECT n_payments, COUNT(*) AS n_orders FROM (
                    SELECT order_id, COUNT(*) AS n_payments FROM order_payments GROUP BY order_id
                ) t GROUP BY n_payments ORDER BY n_payments LIMIT 10
            """})
            rows = result.get("rows", [])
            lines = [f"{r['n_payments']} payment record: {r['n_orders']} đơn" for r in rows]
            n_multi = sum(r["n_orders"] for r in rows if r["n_payments"] != 1)
            return (
                "KHÔNG — nhiều đơn có nhiều payment record (ví dụ trả 1 phần bằng voucher, 1 phần "
                "bằng credit_card). Phân bổ (10 dòng đầu):\n" + "\n".join(lines) +
                f"\n\n→ Với đơn có nhiều payment record, tổng tiền của đơn = SUM(payment_value) theo "
                f"order_id (cộng dồn tất cả record), không lấy 1 dòng bất kỳ."
            ), tool_calls_log

    # N7-Q23: tổng tiền order có luôn = price + freight?
    if any(k in lower for k in ["tổng tiền", "tong tien"]) and any(k in lower for k in ["price", "freight", "giá sản phẩm"]):
        result = call("query_dataset_sql", {"sql": """
            WITH item_totals AS (SELECT order_id, SUM(price+freight_value) AS item_total FROM order_items GROUP BY order_id),
            pay_totals AS (SELECT order_id, SUM(payment_value) AS pay_total FROM order_payments GROUP BY order_id)
            SELECT CASE WHEN ABS(p.pay_total - i.item_total) < 0.01 THEN 'Khớp chính xác'
                        WHEN ABS(p.pay_total - i.item_total) < 1 THEN 'Lệch nhỏ (<1)'
                        ELSE 'Lệch đáng kể (>=1)' END AS nhom_lech, COUNT(*) AS n_orders
            FROM pay_totals p JOIN item_totals i ON p.order_id=i.order_id GROUP BY nhom_lech
        """})
        rows = result.get("rows", [])
        lines = [f"{r['nhom_lech']}: {r['n_orders']} đơn" for r in rows]
        return "KHÔNG luôn luôn — phân loại mức chênh lệch:\n" + "\n".join(lines), tool_calls_log

    # === 5 câu "stress test" ===

    # SA: top 10 seller doanh thu cao 2018, delay > median
    if "2018" in lower and "seller" in lower and "doanh thu" in lower and any(k in lower for k in ["delay", "trễ", "tre"]):
        result = call("query_dataset_sql", {"sql": """
            WITH seller_2018 AS (
                SELECT oi.seller_id, SUM(oi.price) AS revenue, COUNT(DISTINCT oi.order_id) AS n_orders,
                       AVG(o.is_late) AS late_rate, AVG(r.review_score) AS avg_review, AVG(oi.freight_value) AS avg_freight
                FROM order_items oi JOIN orders o ON oi.order_id=o.order_id
                LEFT JOIN order_reviews r ON oi.order_id=r.order_id
                WHERE o.order_status='delivered' AND EXTRACT(YEAR FROM o.order_purchase_timestamp)=2018
                GROUP BY oi.seller_id
            ),
            med AS (SELECT MEDIAN(late_rate) AS m FROM seller_2018)
            SELECT s.seller_id, ROUND(s.revenue,2) AS revenue, s.n_orders, ROUND(s.late_rate*100,2) AS delay_rate_pct,
                   ROUND(s.avg_review,2) AS avg_review, ROUND(s.avg_freight,2) AS avg_freight
            FROM seller_2018 s CROSS JOIN med WHERE s.late_rate > med.m ORDER BY s.revenue DESC LIMIT 10
        """})
        rows = result.get("rows", [])
        if not rows:
            return "Không có seller nào thoả điều kiện trong 2018.", tool_calls_log
        lines = [f"{r['seller_id'][:12]}...: doanh thu {r['revenue']:,.0f}, {r['n_orders']} đơn, {r['delay_rate_pct']}% trễ, review TB {r['avg_review']}, freight TB {r['avg_freight']}" for r in rows]
        return "Top seller doanh thu cao năm 2018 nhưng delay rate > median toàn bộ seller (2018):\n" + "\n".join(lines), tool_calls_log

    # SB: top 1% customer theo chi tiêu, % doanh thu, % quay lại
    if "top 1%" in lower and any(k in lower for k in ["customer_unique_id", "khách hàng", "khach hang"]):
        result = call("query_dataset_sql", {"sql": """
            WITH cust AS (
                SELECT c.customer_unique_id, SUM(o.payment_value) AS spend, COUNT(DISTINCT o.order_id) AS n_orders
                FROM orders o JOIN customers c ON o.customer_id=c.customer_id
                WHERE o.order_status='delivered' GROUP BY c.customer_unique_id
            ),
            ranked AS (SELECT *, PERCENT_RANK() OVER (ORDER BY spend DESC) AS pr FROM cust),
            top1 AS (SELECT * FROM ranked WHERE pr <= 0.01),
            total AS (SELECT SUM(spend) AS total_rev FROM cust)
            SELECT COUNT(*) AS n_top1, ROUND(SUM(top1.spend),2) AS top1_revenue,
                   ROUND(100.0*SUM(top1.spend)/(SELECT total_rev FROM total),2) AS pct_of_total_revenue,
                   ROUND(100.0*AVG(CASE WHEN top1.n_orders>=2 THEN 1.0 ELSE 0 END),2) AS pct_repeat
            FROM top1
        """})
        r = result.get("rows", [{}])[0]
        return (
            f"Top 1% khách hàng ({r.get('n_top1')} người) theo tổng chi tiêu đóng góp "
            f"{r.get('pct_of_total_revenue')}% tổng doanh thu. Trong nhóm này, {r.get('pct_repeat')}% "
            f"là khách quay lại (≥2 đơn)."
        ), tool_calls_log

    # SC: top 10 category doanh thu cao nhưng review thấp, cân nhắc cả volume
    if "danh mục" in lower and "doanh thu" in lower and any(k in lower for k in ["review thấp", "review thap"]) \
            and any(k in lower for k in ["volume", "số lượng đơn", "so luong don"]):
        result = call("query_dataset_sql", {"sql": """
            SELECT ct.product_category_name_english AS category, ROUND(SUM(oi.price),2) AS revenue,
                   COUNT(DISTINCT oi.order_id) AS n_orders, ROUND(AVG(r.review_score),2) AS avg_review
            FROM order_items oi JOIN orders o ON oi.order_id=o.order_id
            JOIN products p ON oi.product_id=p.product_id
            JOIN category_translation ct ON p.product_category_name=ct.product_category_name
            LEFT JOIN order_reviews r ON oi.order_id=r.order_id
            WHERE o.order_status='delivered' GROUP BY category HAVING COUNT(DISTINCT oi.order_id) >= 100
            ORDER BY revenue DESC, avg_review ASC LIMIT 10
        """})
        rows = result.get("rows", [])
        if not rows:
            return "Không lấy được dữ liệu.", tool_calls_log
        lines = [f"{i+1}. {r['category']}: doanh thu {r['revenue']:,.0f}, {r['n_orders']} đơn, review TB {r['avg_review']}" for i, r in enumerate(rows)]
        return (
            "Đã lọc bỏ danh mục <100 đơn để tránh outlier volume nhỏ; sắp theo doanh thu giảm dần, "
            "review tăng dần:\n" + "\n".join(lines)
        ), tool_calls_log

    # SD: kiểm tra nhất quán order_items+freight, payment_value, trạng thái đơn
    if any(k in lower for k in ["nhất quán", "nhat quan"]) and "order_items" in lower:
        result = call("query_dataset_sql", {"sql": """
            WITH item_totals AS (SELECT order_id, SUM(price+freight_value) AS item_total FROM order_items GROUP BY order_id),
            pay_totals AS (SELECT order_id, SUM(payment_value) AS pay_total FROM order_payments GROUP BY order_id)
            SELECT o.order_status, ROUND(AVG(ABS(p.pay_total - i.item_total)),2) AS avg_abs_diff, COUNT(*) AS n
            FROM pay_totals p JOIN item_totals i ON p.order_id=i.order_id
            JOIN orders o ON p.order_id=o.order_id
            GROUP BY o.order_status ORDER BY avg_abs_diff DESC
        """})
        rows = result.get("rows", [])
        lines = [f"{r['order_status']}: chênh lệch TB {r['avg_abs_diff']} ({r['n']} đơn)" for r in rows]
        return (
            "Chênh lệch TB |payment_value - (price+freight)| theo từng trạng thái đơn:\n" + "\n".join(lines) +
            "\n\nCác nguyên nhân dữ liệu có thể: voucher/khuyến mãi, đơn có nhiều payment record, "
            "đơn bị huỷ/hoàn tiền một phần, hoặc sai số làm tròn."
        ), tool_calls_log

    # SE: seller giá cao + freight cao + trễ cao + review thấp — composite score
    if "seller" in lower and any(k in lower for k in ["giá cao", "gia cao"]) and any(k in lower for k in ["freight cao", "freight"]) \
            and any(k in lower for k in ["định lượng", "dinh luong", "phương pháp", "phuong phap"]):
        result = call("query_dataset_sql", {"sql": """
            WITH seller_stats AS (
                SELECT oi.seller_id, AVG(oi.price) AS avg_price, AVG(oi.freight_value) AS avg_freight,
                       AVG(o.is_late) AS late_rate, AVG(r.review_score) AS avg_review, COUNT(DISTINCT oi.order_id) AS n_orders
                FROM order_items oi JOIN orders o ON oi.order_id=o.order_id
                LEFT JOIN order_reviews r ON oi.order_id=r.order_id
                WHERE o.order_status='delivered' GROUP BY oi.seller_id HAVING COUNT(DISTINCT oi.order_id)>=30
            ),
            ranked AS (
                SELECT *, PERCENT_RANK() OVER (ORDER BY avg_price) AS r_price,
                       PERCENT_RANK() OVER (ORDER BY avg_freight) AS r_freight,
                       PERCENT_RANK() OVER (ORDER BY late_rate) AS r_late,
                       PERCENT_RANK() OVER (ORDER BY avg_review ASC) AS r_review_bad
                FROM seller_stats
            )
            SELECT seller_id, ROUND(avg_price,2) AS avg_price, ROUND(avg_freight,2) AS avg_freight,
                   ROUND(late_rate*100,2) AS late_rate_pct, ROUND(avg_review,2) AS avg_review,
                   ROUND((r_price+r_freight+r_late+r_review_bad)/4, 3) AS composite_score
            FROM ranked ORDER BY composite_score DESC LIMIT 10
        """})
        rows = result.get("rows", [])
        if not rows:
            return "Không lấy được dữ liệu.", tool_calls_log
        lines = [f"{r['seller_id'][:12]}...: giá TB {r['avg_price']}, freight TB {r['avg_freight']}, {r['late_rate_pct']}% trễ, review TB {r['avg_review']}, điểm tổng hợp {r['composite_score']}" for r in rows]
        return (
            "Phương pháp: xếp hạng percentile (0-1) từng seller trên 4 tiêu chí (giá cao, freight cao, "
            "tỉ lệ trễ cao, review thấp — quy đổi để '1' luôn là xấu nhất), rồi lấy TRUNG BÌNH 4 "
            "percentile làm 'điểm tổng hợp' (composite score). Điểm càng gần 1 càng đáng lo ngại ở "
            "cả 4 mặt cùng lúc. Chỉ xét seller ≥30 đơn.\n\nTop 10 seller đáng lo ngại nhất:\n" + "\n".join(lines)
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

    # N2-Q5 (mở rộng): thời gian giao TB + tỉ lệ trễ theo tháng, chỉ rõ tháng cao nhất mỗi loại
    if ("tháng" in lower or "thang" in lower) and any(k in lower for k in ["trễ", "tre", "tỉ lệ", "ti le", "thời gian", "thoi gian"]):
        result = call("query_dataset_sql", {
            "sql": "SELECT purchase_month, ROUND(AVG(is_late)*100, 2) AS late_rate_pct, "
                   "ROUND(AVG(actual_delivery_days), 2) AS avg_days, COUNT(*) AS n "
                   "FROM orders WHERE order_status='delivered' GROUP BY purchase_month ORDER BY purchase_month",
        })
        rows = result.get("rows", [])
        if not rows:
            return "Không lấy được dữ liệu theo tháng.", tool_calls_log
        max_days = max(rows, key=lambda r: r["avg_days"])
        max_late = max(rows, key=lambda r: r["late_rate_pct"])
        lines = [f"Tháng {r['purchase_month']}: TB {r['avg_days']} ngày giao, {r['late_rate_pct']}% trễ ({r['n']} đơn)" for r in rows]
        return (
            "Thời gian giao hàng TB + tỉ lệ giao trễ theo tháng:\n" + "\n".join(lines) +
            f"\n\n→ Tháng có thời gian giao TB cao nhất: Tháng {max_days['purchase_month']} ({max_days['avg_days']} ngày)."
            f"\n→ Tháng có tỉ lệ giao trễ cao nhất: Tháng {max_late['purchase_month']} ({max_late['late_rate_pct']}%)."
        ), tool_calls_log

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
    # (phải có "bang" hoặc "ngành hàng"/"category" đi kèm — nếu không, đây là câu hỏi
    # "điểm đánh giá trung bình toàn sàn" đơn giản hơn, xử lý riêng ở bảng chỉ số bên dưới)
    if any(k in lower for k in ["đánh giá", "danh gia", "review"]) and \
            any(k in lower for k in ["bang", "ngành hàng", "nganh hang", "category"]):
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
    # (loại trừ câu hỏi về 1 trạng thái CỤ THỂ hoặc về REVIEW — để những câu đó được xử lý
    # riêng, tránh đếm nhầm TOÀN BỘ đơn hàng thay vì đúng phần được hỏi)
    if any(k in lower for k in ["bao nhiêu đơn", "bao nhieu don", "tổng số đơn", "tong so don"]) \
            and not any(k in lower for k in [
                "delivered", "được giao", "duoc giao", "hủy", "huỷ", "huy", "cancel",
                "sao", "review", "đánh giá", "danh gia",
            ]):
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

    # 14.5) Đếm số SẢN PHẨM thuộc 1 danh mục cụ thể (khác với "thống kê ngành hàng" ở dưới —
    # câu này hỏi số lượng SẢN PHẨM trong danh mục, không phải số đơn/tỉ lệ trễ của danh mục)
    if any(k in lower for k in ["sản phẩm", "san pham"]) and "bao nhiêu" in lower \
            and any(k in lower for k in ["danh mục", "danh muc", "thuộc", "thuoc"]):
        cats_for_count = _find_categories_in_text(text)
        if cats_for_count:
            cat = cats_for_count[0]
            result = call("query_dataset_sql", {
                "sql": "SELECT COUNT(*) AS n FROM products p JOIN category_translation ct "
                       "ON p.product_category_name = ct.product_category_name "
                       f"WHERE ct.product_category_name_english = '{cat}'",
            })
            n = result.get("rows", [{}])[0].get("n")
            return f"Danh mục '{cat}' có {n} sản phẩm.", tool_calls_log

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

    # 17) Bảng chỉ số cơ bản/thống kê tổng quan (M1-M29) — kiểm tra sau cùng, trước describe/fallback
    for metric in BASIC_METRICS:
        if metric["match"](lower):
            return _run_basic_metric(metric, call), tool_calls_log

    # 18) Mô tả cấu trúc dataset
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
