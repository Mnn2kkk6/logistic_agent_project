# syntax=docker/dockerfile:1

# ============ Stage 1: build — cài dependency vào 1 venv riêng ============
FROM python:3.12-slim AS builder

WORKDIR /app

# Cài đặt các gói hệ thống cần để build một số wheel (vd xgboost, pyarrow) nếu không có
# sẵn bản wheel cho kiến trúc máy — apt cache được xoá ngay sau khi cài để giữ layer nhỏ.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy CHỈ requirements trước — Docker cache layer này, nên nếu chỉ sửa code (src/, templates/)
# mà không đổi dependency thì bước "pip install" (chậm nhất) sẽ không phải chạy lại.
COPY requirements-docker.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements-docker.txt

# ============ Stage 2: runtime — image cuối cùng, không có build-essential ============
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    FLASK_DEBUG=0 \
    PORT=5000

WORKDIR /app

# Chỉ copy venv đã cài sẵn từ stage build — không mang theo build-essential, pip cache,...
COPY --from=builder /opt/venv /opt/venv

# Copy code + dataset đã xử lý + model đã train (data/models đổi ít, để layer riêng cuối
# cùng, không làm hỏng cache của layer pip install ở trên khi chỉ sửa code).
COPY src ./src
COPY templates ./templates
COPY data ./data
COPY models ./models

# Chạy bằng user thường, không phải root — thực hành bảo mật cơ bản khi deploy.
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 5000

# Healthcheck để "docker ps" / docker-compose biết container có thực sự phục vụ được không,
# không chỉ là process còn sống.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=3)" || exit 1

CMD ["python", "-m", "src.api"]
