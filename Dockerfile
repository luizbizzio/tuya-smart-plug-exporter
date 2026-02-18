FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY tuya_exporter.py .

EXPOSE 9999

CMD ["python", "tuya_smartplug_exporter.py", "--config.file=/config/config.yaml"]
