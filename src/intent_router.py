"""Lightweight intent routing for the single-chat AI Logistics Agent.

The classifier deliberately uses deterministic rules instead of another LLM call.
It routes clear document questions to direct RAG, quantitative questions to the
active dataset tools, mixed questions to the full hybrid agent, and small-talk /
general questions to a no-tool LLM request.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any


_DOCUMENT_MARKERS = (
    "readme",
    "tai lieu",
    "tai lieu du an",
    "theo tai lieu",
    "dua tren tai lieu",
    "documentation",
    "docs",
    "dinh nghia",
    "y nghia cua truong",
    "y nghia truong",
    "mo ta truong",
    "truong du lieu",
    "data dictionary",
    "kien truc",
    "quy trinh",
    "cau truc repo",
    "cau truc du an",
    "huong dan",
    "chuc nang cua du an",
    "schema",
    "ten cot",
    "cac cot",
    "column names",
    "architecture",
    "project structure",
    "field definition",
    "data field",
    "workflow",
    "project documentation",
    "project overview",
    "project architecture",
    "project structure",
    "how does the project work",
    "how does this project work",
    "what does the project do",
    "what does this project do",
    "how does the agent work",
    "ai logistics agent",
    "agent cua du an",
    "du an nay",
    "he thong nay",
)

_DATA_MARKERS = (
    "bao nhieu",
    "tinh ",
    "dem ",
    "tong ",
    "trung binh",
    "top ",
    "xep hang",
    "thong ke",
    "ty le",
    "phan tram",
    "doanh thu",
    "so don",
    "so luong",
    "cao nhat",
    "thap nhat",
    "nhieu nhat",
    "it nhat",
    "du doan",
    "predict",
    "forecast",
    "train model",
    "huan luyen",
    "sql",
    "query",
    "join ",
    "xu huong",
    "theo thang",
    "theo quy",
    "theo nam",
    "tuong quan",
    "correlation",
    "late rate",
    "delivery risk",
    "rui ro giao",
    "calculate",
    "count ",
    "how many",
    "average",
    "total ",
    "percentage",
    "rank ",
    "compare ",
    "by market",
    "by region",
    "by country",
    "by seller",
    "by category",
    "theo thi truong",
    "theo quoc gia",
    "theo khu vuc",
    "theo seller",
    "theo nganh hang",
    "theo phuong thuc van chuyen",
    "nhanh nhat",
    "tot nhat",
    "fastest",
    "slowest",
    "best performing",
)

_HYBRID_MARKERS = (
    "ket hop",
    "doi chieu tai lieu voi",
    "so sanh tai lieu voi",
    "dua tren tai lieu va",
    "dung tai lieu de giai thich",
    "giai thich va tinh",
    "giai thich va thong ke",
    "theo tai lieu va tinh",
)

_FOLLOW_UP_MARKERS = (
    "con ",
    "vay ",
    "the ",
    "noi ro hon",
    "giai thich them",
    "chi tiet hon",
    "cai nay",
    "truong nay",
    "y do",
)


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", (text or "").lower())
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    return text.replace("đ", "d").replace("–", "-").replace("—", "-")


def _has_field_definition(text: str) -> bool:
    """Recognize questions asking what a snake_case dataset field means."""
    has_definition_phrase = any(
        phrase in text
        for phrase in (" la gi", " nghia la gi", " y nghia ", " meaning of ", " mean?", "what is ", "what does ", "definition of ", "define ", "explain ", "giai thich ", "describe ")
    )
    has_field_name = bool(re.search(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b", text))
    return has_definition_phrase and has_field_name


def _classify_one(message: str) -> str:
    text = _normalize(message).strip()
    if not text:
        return "general"

    has_document_intent = any(marker in text for marker in _DOCUMENT_MARKERS)
    has_document_intent = has_document_intent or _has_field_definition(text)
    has_named_project = any(
        entity in text
        for entity in ("dataco", "olist", "ai logistics agent", "qdrant", "fastembed")
    )
    asks_for_definition = any(
        phrase in text
        for phrase in (" la gi", " what is ", "what is", " nghia la gi", "meaning of ")
    )
    has_document_intent = has_document_intent or (has_named_project and asks_for_definition)
    has_data_intent = any(marker in text for marker in _DATA_MARKERS)
    has_explicit_hybrid = any(marker in text for marker in _HYBRID_MARKERS)

    if has_explicit_hybrid or (has_document_intent and has_data_intent):
        return "hybrid"
    if has_document_intent:
        return "rag"
    if has_data_intent:
        return "data_ml"
    return "general"


def filter_tools_for_intent(available_tools: list[dict[str, Any]], intent: str) -> list[dict[str, Any]]:
    """Return only tool schemas appropriate to the current request intent."""
    if intent == "general":
        return []
    if intent == "data_ml":
        return [tool for tool in available_tools if tool.get("name") != "query_knowledge_base"]
    if intent == "rag":
        return [tool for tool in available_tools if tool.get("name") == "query_knowledge_base"]
    return list(available_tools)


def classify_intent(message: str, history: list[dict[str, Any]] | None = None) -> str:
    """Return one of: rag, data_ml, hybrid, general.

    Short follow-up messages reuse the last clear intent so a conversation such as
    "What does late_delivery_risk mean?" -> "Can you explain that more?" stays on
    the document path instead of silently losing its retrieval capability.
    """
    intent = _classify_one(message)
    normalized = _normalize(message).strip()
    words = normalized.split()
    is_short_follow_up = (
        len(words) <= 7
        and any(normalized.startswith(marker) for marker in _FOLLOW_UP_MARKERS)
    )
    if intent == "general" and is_short_follow_up and history:
        for item in reversed(history[-8:]):
            if not isinstance(item, dict) or item.get("role") != "user":
                continue
            previous_intent = _classify_one(str(item.get("content", "")))
            if previous_intent != "general":
                return previous_intent
    return intent
