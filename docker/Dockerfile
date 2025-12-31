# docker/Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# deps for lightgbm runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
  && rm -rf /var/lib/apt/lists/*

# копируем метаданные и исходники
COPY pyproject.toml README.md /app/
COPY src /app/src

# ставим пакет
RUN pip install --no-cache-dir -U pip setuptools wheel \
    && pip install --no-cache-dir .

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "bullfinch_forest_ml_demo.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
