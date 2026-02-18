FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY tuya_smart_plug_exporter.py .

EXPOSE 9999

VOLUME ["/config"]

ENTRYPOINT ["python", "tuya_smart_plug_exporter.py"]
CMD ["--config.file=/config/config.yaml"]
