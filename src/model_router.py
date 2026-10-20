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
    if _valid_configured_model(preferred):
        ranked.append(preferred)

    for model_id in base:
        if _valid_configured_model(model_id) and model_id not in ranked:
            ranked.append(model_id)

    if not ranked:
        # Defensive fallback: include any configured non-local model, followed by
        # the repository's offline fallback if present.
        for model_id, info in providers.AVAILABLE_MODELS.items():
            if info.get("provider") != "local" and _key_is_configured(info):
                ranked.append(model_id)
        if "local-rule-based" in providers.AVAILABLE_MODELS:
            ranked.append("local-rule-based")

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
    """Chạy lượt chat bằng model được chọn ngầm; chỉ backend ghi log model đã dùng."""
    task = classify_task(user_message)
    candidates = _ordered_candidates(task)

    try:
        max_attempts = int(os.environ.get("MODEL_ROUTER_MAX_ATTEMPTS", "3"))
    except ValueError:
        max_attempts = 3
    max_attempts = min(max(1, max_attempts), 5)
    candidates = candidates[:max_attempts]

    errors: list[str] = []
    for attempt, model_id in enumerate(candidates, start=1):
        try:
            reply, tool_calls, new_history = providers.chat_turn(
                model_id, history, user_message
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

    last_error = errors[-1] if errors else "Không có model phù hợp."
    raise providers.ProviderError(
        "Agent chưa thể xử lý câu hỏi bằng các model khả dụng. "
        f"Lỗi cuối cùng: {last_error}"
    )
