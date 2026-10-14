"""
Flask API + Web UI cho AI Logistics Agent — hỗ trợ chọn model.

Endpoints:
  GET  /                    Giao diện chat web
  GET  /health
  GET  /models               Danh sách model khả dụng + trạng thái đã cấu hình key hay chưa
  POST /predict               {distance_km, customer_state, seller_state, ...}
  GET  /order/<order_id>
  GET  /stats/state/<state>
  GET  /stats/seller/<seller_id>
  GET  /stats/category/<category>
  GET  /stats/top-risky-states?n=5
  GET  /stats/top-risky-categories?n=5
  POST /chat                  {"message": "...", "model": "..."}  (nhớ lịch sử theo session)
  POST /chat/restore          {"history": [{"role":"user|assistant", "content":"..."}]}
                              khôi phục context khi mở lại một cuộc trò chuyện đã lưu ở frontend
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

from flask import Flask, jsonify, render_template, request, send_from_directory, session

from src import providers, tools
from src.marketing_tools import (
    get_marketing_funnel_summary,
    get_marketing_channel_performance,
    get_seller_360,
    get_acquisition_logistics_performance,
)

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


@app.post("/chat/restore")
def chat_restore():
    """
    Khôi phục canonical history từ frontend khi người dùng mở lại một chat đã lưu.
    Chỉ nhận role user/assistant và giới hạn số message để tránh nạp payload bất thường.
    """
    payload = request.get_json(force=True) or {}
    raw_history = payload.get("history")

    if not isinstance(raw_history, list):
        return jsonify({"error": "history phải là một mảng."}), 400

    if len(raw_history) > 100:
        return jsonify({"error": "Lịch sử hội thoại quá dài."}), 400

    canonical_history = []
    for item in raw_history:
        if not isinstance(item, dict):
            return jsonify({"error": "Mỗi phần tử history phải là object."}), 400

        role = item.get("role")
        content = item.get("content")

        if role not in {"user", "assistant"}:
            return jsonify({"error": "role phải là user hoặc assistant."}), 400
        if not isinstance(content, str):
            return jsonify({"error": "content phải là chuỗi."}), 400

        content = content.strip()
        if not content:
            continue

        canonical_history.append({"role": role, "content": content})

    _CHAT_HISTORY[_get_session_id()] = canonical_history
    return jsonify({"status": "ok", "messages": len(canonical_history)})


@app.get("/map")
def map_page():
    # CARTO yêu cầu API key (miễn phí) cho tile raster; không có key thì map.html tự dùng nền OSM.
    # Key sẽ hiện trong HTML gửi cho người xem (không phải bí mật) — để ở biến môi trường cho dễ đổi.
    return render_template("map.html", carto_key=os.environ.get("CARTO_API_KEY", "").strip())


@app.get("/api/map/filters")
def map_filters():
    """Danh sách tháng / ngành hàng / bang cho bộ lọc của bản đồ."""
    return jsonify(tools.get_map_filters())


@app.get("/api/map/flows")
def map_flows():
    """Luồng gộp theo bang. Query: start, end (YYYY-MM), category, min_orders."""
    return jsonify(tools.get_state_flow_map(
        start=request.args.get("start"),
        end=request.args.get("end"),
        category=request.args.get("category"),
        min_orders=request.args.get("min_orders", 5),
    ))


@app.get("/api/map/sample")
def map_sample():
    from_state = request.args.get("from", "")
    to_state = request.args.get("to", "")
    try:
        limit = int(request.args.get("limit", 40))
    except ValueError:
        limit = 40
    return jsonify(tools.get_order_sample_for_flow(
        from_state, to_state, limit,
        start=request.args.get("start"),
        end=request.args.get("end"),
        category=request.args.get("category"),
    ))


@app.get("/api/map/trend")
def map_trend():
    """Chuỗi theo tháng của 1 luồng (cho sparkline)."""
    return jsonify(tools.get_flow_trend(
        request.args.get("from", ""), request.args.get("to", ""),
        start=request.args.get("start"),
        end=request.args.get("end"),
        category=request.args.get("category"),
    ))


@app.post("/api/map/predict-route")
def map_predict_route():
    """Dự đoán 1 đơn mới giữa 2 bang và trả kèm toạ độ để vẽ lên bản đồ."""
    p = request.get_json(force=True, silent=True) or {}
    result = tools.predict_route(
        p.get("from_state", ""), p.get("to_state", ""),
        category=p.get("category") or None,
        total_price=p.get("total_price"),
        total_freight=p.get("total_freight"),
        product_weight_g=p.get("product_weight_g"),
    )
    return jsonify(result), (400 if "error" in result else 200)


@app.get("/api/map/geojson")
def map_geojson():
    """Ranh giới 27 bang Brazil (đã đơn giản hoá) cho lớp choropleth."""
    resp = send_from_directory(tools.BASE_DIR / "data", "brazil_states.geojson", mimetype="application/json")
    resp.headers["Cache-Control"] = "public, max-age=86400"
    return resp


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
    except Exception as e:  # noqa: BLE001
        # Bất kỳ lỗi nào khác đều phải trả JSON, không để Flask trả HTML 500 mặc định.
        app.logger.exception("Lỗi không mong đợi khi xử lý /chat (model=%s)", model)
        return jsonify({"error": f"Đã có lỗi không mong đợi khi xử lý yêu cầu: {e}"}), 500

    _CHAT_HISTORY[sid] = new_history
    return jsonify({"reply": reply, "tool_calls": tool_calls, "model": model})


@app.get("/marketing/funnel")
def marketing_funnel():
    origin = request.args.get("origin")
    return jsonify(get_marketing_funnel_summary(origin))


@app.get("/marketing/channels")
def marketing_channels():
    limit = int(request.args.get("limit", 20))
    return jsonify(get_marketing_channel_performance(limit))


@app.get("/seller/<seller_id>/360")
def seller_360(seller_id):
    return jsonify(get_seller_360(seller_id))


@app.get("/marketing/acquisition-logistics")
def acquisition_logistics():
    limit = int(request.args.get("limit", 20))
    return jsonify(get_acquisition_logistics_performance(limit))


def _warm_map_cache():
    """Tính sẵn dữ liệu bản đồ (filters + luồng mặc định) ở luồng nền để lần mở /map đầu tiên
    không phải chờ DuckDB đọc CSV + chạy model. Tắt bằng MAP_WARMUP=0."""
    try:
        tools.get_map_filters()
        tools.get_state_flow_map()
    except Exception:  # noqa: BLE001 — warm-up chỉ là tối ưu, lỗi thì bỏ qua
        app.logger.exception("Warm-up bản đồ thất bại (bỏ qua)")


if os.environ.get("MAP_WARMUP", "1") == "1":
    import threading
    threading.Thread(target=_warm_map_cache, daemon=True).start()


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    port = int(os.environ.get("PORT", 5000))
    # host=0.0.0.0 để container Docker expose ra ngoài được.
    app.run(host="0.0.0.0", port=port, debug=debug)
