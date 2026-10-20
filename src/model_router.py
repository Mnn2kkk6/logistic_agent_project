"""Automatic LLM router for AI Logistics Agent.

The UI selects a dataset, never an LLM. This module classifies the task,
chooses a configured provider/model, and retries with compatible alternatives.
The existing XGBoost business models remain responsible for logistics predictions.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from src import providers

logger = logging.getLogger(__name__)

# Ordering is a preference, not a guarantee of provider availability. Candidates are
# filtered against providers.AVAILABLE_MODELS and configured API keys at runtime.
_FAST_MODEL_ORDER = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "openai/gpt-oss-20b",
    "nvidia/nemotron-3.5-lightning:free",
    "google/gemma-4-31b-it:free",
    "gpt-4o-mini",
    "openrouter/free",
    "openai/gpt-oss-120b",
    "nvidia/nemotron-3-ultra-550b-a55b-20260604:free",
    "poolside/laguna-s-2.1:free",
    "gpt-4o",
    "grok-4-fast",
    "grok-4.6",
    "gemini-3.1-pro-preview",
    "local-rule-based",
]

_REASONING_MODEL_ORDER = [
    "gemini-3.1-pro-preview",
    "nvidia/nemotron-3-ultra-550b-a55b-20260604:free",
    "openai/gpt-oss-120b",
    "grok-4.6",
    "gpt-4o",
    "poolside/laguna-s-2.1:free",
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "openai/gpt-oss-20b",
    "nvidia/nemotron-3.5-lightning:free",
    "google/gemma-4-31b-it:free",
    "gpt-4o-mini",
    "openrouter/free",
    "grok-4-fast",
    "local-rule-based",
]

_REASONING_MARKERS = (
    "so sánh", "so sanh", "đối chiếu", "tương quan", "mối quan hệ",
    "nguyên nhân", "vì sao", "phân tích sâu", "nhiều chiều",
    "theo tháng và", "theo quý và", "xu hướng", "kết hợp",
    "join ", "sql", "forecast", "dự báo xu hướng", "dự báo nhiều",
    "huấn luyện", "train model", "feature importance",
)


def classify_task(message: str) -> str:
    """Phân loại nhẹ bằng quy tắc trước khi chọn model; không gọi thêm LLM."""
    text = (message or "").strip().lower()
    if any(marker in text for marker in _REASONING_MARKERS):
        return "reasoning"
    return "fast"


def _key_is_configured(model_info: dict[str, Any]) -> bool:
    env_key = model_info.get("env_key")
    if env_key is None:
        return True
    if os.environ.get(env_key):
        return True
    # providers._gemini_client supports GOOGLE_API_KEY as an alternative.
    if env_key == "GEMINI_API_KEY" and os.environ.get("GOOGLE_API_KEY"):
        return True
    return False


def _valid_configured_model(model_id: str | None) -> bool:
    if not model_id:
        return False
    info = providers.AVAILABLE_MODELS.get(model_id)
    return bool(info and _key_is_configured(info))


def _ordered_candidates(task: str) -> list[str]:
    base = _REASONING_MODEL_ORDER if task == "reasoning" else _FAST_MODEL_ORDER
    preferred_env = (
        "MODEL_ROUTER_REASONING_MODEL"
        if task == "reasoning"
        else "MODEL_ROUTER_FAST_MODEL"
    )
    preferred = os.environ.get(preferred_env, "").strip()

    ranked: list[str] = []
    if (
        _valid_configured_model(preferred)
        and providers.AVAILABLE_MODELS[preferred].get("provider") != "local"
    ):
        ranked.append(preferred)

    for model_id in base:
        info = providers.AVAILABLE_MODELS.get(model_id, {})
        # For open-ended questions, local has already been checked. Do not let a
        # generic local fallback jump ahead of configured online providers.
        if info.get("provider") == "local":
            continue
        if _valid_configured_model(model_id) and model_id not in ranked:
            ranked.append(model_id)

    if not ranked:
        # If no online provider has a configured key, the local response remains
        # the final graceful fallback (it may show supported question examples).
        if dataset_id == "olist" and "local-rule-based" in providers.AVAILABLE_MODELS:
            return ["local-rule-based"]

    if not ranked:
        raise providers.ProviderError(
            "Không tìm thấy model nào khả dụng. Hãy kiểm tra API key trong file .env."
        )

    # On a failure, prefer a different provider first so fallback is not merely a
    # second model behind the same provider/key. Then try the remaining candidates.
    primary = ranked[0]
    primary_provider = providers.AVAILABLE_MODELS[primary].get("provider")
    cross_provider = [
        model_id for model_id in ranked[1:]
        if providers.AVAILABLE_MODELS[model_id].get("provider") != primary_provider
    ]
    same_provider = [
        model_id for model_id in ranked[1:]
        if providers.AVAILABLE_MODELS[model_id].get("provider") == primary_provider
    ]
    return [primary, *cross_provider, *same_provider]


def chat_turn_auto(
    history: list[dict[str, str]],
    user_message: str,
    dataset_id: str = "olist",
) -> tuple[str, list[dict[str, Any]], list[dict[str, str]], dict[str, Any]]:
    """Ưu tiên local cho mẫu đã code; câu hỏi còn lại mới dùng model online."""
    task = classify_task(user_message)
    local_reply: str | None = None
    local_tool_calls: list[dict[str, Any]] = []
    local_new_history: list[dict[str, str]] | None = None
    local_marker = "Mình chưa nhận diện được mẫu câu này"

    # Run the existing rule-based agent first. If it recognizes a coded pattern,
    # its answer/tool result wins and no paid API call is made.
    if "local-rule-based" in providers.AVAILABLE_MODELS:
        try:
            local_reply, local_tool_calls, local_new_history = providers.chat_turn(
                "local-rule-based", history, user_message, dataset_id=dataset_id
            )
            if local_marker not in local_reply:
                route_info = {
                    "task": "local",
                    "model_id": "local-rule-based",
                    "attempt": 0,
                    "fallback_used": False,
                    "dataset_id": dataset_id,
                }
                logger.info(
                    "Auto routing: matched local rule (dataset=%s)",
                    dataset_id,
                )
                return local_reply, local_tool_calls, local_new_history, route_info
        except Exception as exc:  # local rule failure should not block online providers
            logger.warning("Local rule attempt failed; trying configured providers: %s", exc)
            local_reply = None
            local_tool_calls = []
            local_new_history = None

    candidates = _ordered_candidates(task)
    # Do not repeat local when it was just tested and did not recognize the prompt.
    candidates = [candidate for candidate in candidates if candidate != "local-rule-based"]

    if not candidates:
        if dataset_id == "dataco":
            raise providers.ProviderError("DataCo đang được chọn nhưng chưa có model AI trực tuyến khả dụng. Hãy cấu hình API key có quota trong .env.")
        if local_reply is not None and local_new_history is not None:
            route_info = {
                "task": "local",
                "model_id": "local-rule-based",
                "attempt": 1,
                "fallback_used": True,
                "dataset_id": dataset_id,
            }
            logger.info("No configured online model; using local guidance (dataset=%s)", dataset_id)
            return local_reply, local_tool_calls, local_new_history, route_info
        raise providers.ProviderError(
            "Không có dịch vụ AI trực tuyến nào được cấu hình và bộ câu hỏi sẵn có chưa xử lý được yêu cầu này."
        )

    try:
        max_attempts = int(os.environ.get("MODEL_ROUTER_MAX_ATTEMPTS", "8"))
    except ValueError:
        max_attempts = 8
    max_attempts = min(max(1, max_attempts), 8)
    candidates = candidates[:max_attempts]

    errors: list[str] = []
    for attempt, model_id in enumerate(candidates, start=1):
        try:
            reply, tool_calls, new_history = providers.chat_turn(
                model_id, history, user_message, dataset_id=dataset_id
            )
            route_info = {
                "task": task,
                "model_id": model_id,
                "attempt": attempt,
                "fallback_used": attempt > 1,
                "dataset_id": dataset_id,
            }
            logger.info(
                "Auto model routing: task=%s dataset=%s model=%s attempt=%s",
                task, dataset_id, model_id, attempt,
            )
            return reply, tool_calls, new_history, route_info
        except Exception as exc:  # provider SDK errors vary by vendor
            errors.append(f"{model_id}: {exc}")
            logger.warning(
                "Model routing attempt failed (task=%s, model=%s, attempt=%s): %s",
                task, model_id, attempt, exc,
            )

    # If every configured online provider failed (e.g. quota/rate limits), do not
    # expose provider/model identifiers in the chat UI. Return local guidance instead.
    if local_reply is not None and local_new_history is not None:
        route_info = {
            "task": "local",
            "model_id": "local-rule-based",
            "attempt": len(candidates) + 1,
            "fallback_used": True,
            "dataset_id": dataset_id,
        }
        logger.warning("All online candidates failed; returning local guidance (dataset=%s)", dataset_id)
        return local_reply, local_tool_calls, local_new_history, route_info

    raise providers.ProviderError(
        "Agent hiện chưa thể xử lý câu hỏi. Hãy kiểm tra API key, hạn mức sử dụng "
        "hoặc cấu hình dịch vụ AI trên server rồi thử lại."
    )
