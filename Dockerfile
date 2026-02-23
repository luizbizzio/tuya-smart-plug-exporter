FROM python:3.11-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY tuya_smart_plug_exporter.py .

RUN useradd -r -u 10001 appuser && mkdir -p /config && chown -R appuser:appuser /app /config
USER appuser

EXPOSE 9122

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:9122/-/healthy', timeout=3)"

ENTRYPOINT ["python", "tuya_smart_plug_exporter.py"]
CMD ["--config.file=/config/config.yaml"]
