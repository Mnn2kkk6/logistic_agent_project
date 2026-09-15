"""
Flask API + Web UI cho AI Logistics Agent — hỗ trợ chọn model (Gemini Flash /
Flash-Lite / GPT-4o / GPT-4o mini) để tránh nghẽn quota free tier.

Endpoints:
  GET  /                    Giao diện chat web (mở http://localhost:5000)
  GET  /health
  GET  /models               Danh sách model khả dụng + trạng thái đã cấu hình key hay chưa
  POST /predict               {distance_km, customer_state, seller_state, ...}
  GET  /order/<order_id>
  GET  /stats/state/<state>
  GET  /stats/seller/<seller_id>
  GET  /stats/category/<category>
  GET  /stats/top-risky-states?n=5
  GET  /stats/top-risky-categories?n=5
  POST /chat                  {"message": "...", "model": "gemini-2.5-flash"}  (nhớ lịch sử theo session)
  POST /chat/reset                                  xoá lịch sử hội thoại của session hiện tại

Chạy local:
    $env:GEMINI_API_KEY="AIza..."        (PowerShell)
    python -m src.api
Rồi mở trình duyệt: http://localhost:5000

Chạy bằng Docker: xem README mục "Chạy bằng Docker".
"""
import os
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, session

from src import providers, tools

# template_folder chỉ định rõ ràng vì app chạy như package "src.api" — mặc định Flask sẽ
# tìm "src/templates/" (sai) thay vì "templates/" ở thư mục gốc project (đúng).
_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
app = Flask(__name__, template_folder=str(_TEMPLATE_DIR))
# Secret key dùng để ký cookie session. Đặt FLASK_SECRET_KEY cố định (vd trong .env / Docker)
# để session không bị mất khi container/app khởi động lại; nếu không set thì dùng random mỗi lần.
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(24).hex())

# Lưu lịch sử hội thoại (dạng canonical: list[{"role", "content"}]) theo session_id,
# sống trong bộ nhớ process. Đủ dùng cho chạy local/demo 1 vài người dùng; nếu deploy
# nhiều người dùng thật đồng thời thì nên đổi sang Redis/DB thay vì dict trong RAM.
_CHAT_HISTORY: dict[str, list] = {}


def _get_session_id() -> str:
    sid = session.get("sid")
    if not sid:
        sid = uuid.uuid4().hex
        session["sid"] = sid
    return sid


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def models():
    return jsonify({"models": providers.list_models(), "default": providers.DEFAULT_MODEL})


@app.post("/predict")
def predict():
    payload = request.get_json(force=True) or {}
    return jsonify(tools.predict_new_order(**payload))


@app.get("/order/<order_id>")
def order(order_id):
    return jsonify(tools.get_order_info(order_id))


@app.get("/stats/state/<state>")
def state_stats(state):
    return jsonify(tools.get_state_stats(state))


@app.get("/stats/seller/<seller_id>")
def seller_stats(seller_id):
    return jsonify(tools.get_seller_stats(seller_id))


@app.get("/stats/category/<category>")
def category_stats(category):
    return jsonify(tools.get_category_stats(category))


@app.get("/stats/top-risky-states")
def top_states():
    n = int(request.args.get("n", 5))
    return jsonify(tools.top_risky_states(n))


@app.get("/stats/top-risky-categories")
def top_categories():
    n = int(request.args.get("n", 5))
    return jsonify(tools.top_risky_categories(n))


@app.post("/chat/reset")
def chat_reset():
    _CHAT_HISTORY.pop(_get_session_id(), None)
    return jsonify({"status": "ok"})


@app.post("/chat")
def chat():
    payload = request.get_json(force=True) or {}
    message = (payload.get("message") or "").strip()
    model = payload.get("model") or providers.DEFAULT_MODEL
    if not message:
        return jsonify({"error": "Thiếu nội dung tin nhắn."}), 400
    if model not in providers.AVAILABLE_MODELS:
        return jsonify({"error": f"Model '{model}' không hợp lệ."}), 400

    sid = _get_session_id()
    history = _CHAT_HISTORY.setdefault(sid, [])

    try:
        reply, tool_calls, new_history = providers.chat_turn(model, history, message)
    except providers.ProviderError as e:
        return jsonify({"error": str(e)}), 502

    _CHAT_HISTORY[sid] = new_history
    return jsonify({"reply": reply, "tool_calls": tool_calls, "model": model})


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    port = int(os.environ.get("PORT", 5000))
    # host=0.0.0.0 để container Docker expose ra ngoài được (127.0.0.1 mặc định chỉ nghe
    # trong nội bộ container, máy host sẽ không kết nối được dù đã map port).
    app.run(host="0.0.0.0", port=port, debug=debug)
