"""
Flask API + Web UI cho AI Logistics Agent.

Endpoints:
  GET  /                    Giao diện chat web (mở http://localhost:5000)
  GET  /health
  POST /predict              {distance_km, customer_state, seller_state, ...}
  GET  /order/<order_id>
  GET  /stats/state/<state>
  GET  /stats/seller/<seller_id>
  GET  /stats/category/<category>
  GET  /stats/top-risky-states?n=5
  GET  /stats/top-risky-categories?n=5
  POST /chat                 {"message": "..."}   (cần GEMINI_API_KEY, nhớ lịch sử theo session)
  POST /chat/reset                                 xoá lịch sử hội thoại của session hiện tại

Chạy:
    set GEMINI_API_KEY=AIza...   (PowerShell: $env:GEMINI_API_KEY="AIza...")
    python -m src.api
Rồi mở trình duyệt: http://localhost:5000
"""
import os
import uuid

from flask import Flask, jsonify, render_template, request, session

from src import tools

app = Flask(__name__)
# Secret key dùng để ký cookie session (chỉ chạy local nên dùng random mỗi lần start là đủ).
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(24).hex())

# Lưu lịch sử hội thoại (list[types.Content]) theo session_id, sống trong bộ nhớ process.
# Đủ dùng cho chạy local 1 người dùng; nếu deploy nhiều người dùng thật thì cần đổi sang
# Redis/DB thay vì dict trong RAM.
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
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        return jsonify({"error": "Chưa cấu hình GEMINI_API_KEY trên server."}), 400

    from google.genai import types

    from src.agent import SYSTEM_PROMPT, _build_gemini_tools, _generate_with_retry, _get_genai_client, run_tool

    message = (request.get_json(force=True) or {}).get("message", "").strip()
    if not message:
        return jsonify({"error": "Thiếu nội dung tin nhắn."}), 400

    sid = _get_session_id()
    contents = _CHAT_HISTORY.setdefault(sid, [])
    contents.append(types.Content(role="user", parts=[types.Part(text=message)]))

    client = _get_genai_client()
    config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, tools=_build_gemini_tools())

    final_text = ""
    tool_calls_log = []
    try:
        for _ in range(5):  # tối đa 5 vòng gọi tool để tránh loop vô hạn
            response = _generate_with_retry(client, config, contents)
            candidate = response.candidates[0]
            contents.append(candidate.content)

            function_calls = [p.function_call for p in candidate.content.parts if p.function_call]
            if not function_calls:
                final_text = response.text or ""
                break

            response_parts = []
            for fc in function_calls:
                args = dict(fc.args) if fc.args else {}
                result = run_tool(fc.name, args)
                tool_calls_log.append({"name": fc.name, "args": args})
                response_parts.append(types.Part.from_function_response(name=fc.name, response={"result": result}))
            contents.append(types.Content(role="user", parts=response_parts))
    except Exception as e:  # noqa: BLE001
        # Bỏ lượt user vừa thêm khỏi lịch sử để không làm hỏng ngữ cảnh phiên chat cho lần hỏi tiếp theo.
        contents.pop()
        return jsonify({"error": f"Lỗi khi gọi Gemini API: {e}"}), 502

    return jsonify({"reply": final_text, "tool_calls": tool_calls_log})


if __name__ == "__main__":
    app.run(debug=True, port=5000)