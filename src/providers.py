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
}
DEFAULT_MODEL = "gemini-2.5-flash"


class ProviderError(Exception):
    """Lỗi có thể hiển thị trực tiếp cho người dùng (thiếu key, hết quota, model sai...)."""


def list_models() -> list:
    """Trả về danh sách model kèm trạng thái đã cấu hình API key hay chưa (để FE hiện rõ)."""
    result = []
    for model_id, info in AVAILABLE_MODELS.items():
        result.append({
            "id": model_id,
            "label": info["label"],
            "provider": info["provider"],
            "note": info["note"],
            "configured": bool(os.environ.get(info["env_key"])),
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

    if not os.environ.get(info["env_key"]):
        raise ProviderError(f"Chưa cấu hình biến môi trường {info['env_key']} cho model '{model}'.")

    if info["provider"] == "gemini":
        reply, tool_calls = _gemini_turn(model, history, user_message, max_tool_rounds)
    elif info["provider"] == "openai":
        reply, tool_calls = _openai_turn(model, history, user_message, max_tool_rounds)
    elif info["provider"] == "xai":
        reply, tool_calls = _xai_turn(model, history, user_message, max_tool_rounds)
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