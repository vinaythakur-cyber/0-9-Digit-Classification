# Slim CPU image. The model is ~288k parameters and runs a prediction in a few
# milliseconds on a CPU, so there is no reason to ship a CUDA image.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    MODEL_PATH=/app/models/digit_ood_cnn.pt

WORKDIR /app

# Dependencies first, in their own layer: application code changes far more
# often than the requirements, so this keeps rebuilds fast.
COPY requirements.txt .
# The CPU-only wheel index turns a ~2.5 GB CUDA install into ~200 MB.
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

COPY src/ src/
COPY app/ app/
COPY configs/ configs/
COPY models/ models/

# Run as a non-root user. Nothing here needs root, and a container that does
# not need it should not have it.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
