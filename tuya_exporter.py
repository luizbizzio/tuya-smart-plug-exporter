# SPDX-FileCopyrightText: Copyright (c) 2024-2026 Luiz Bizzio
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry
from wsgiref.simple_server import make_server
import tinytuya
from threading import Thread
import time

# Exporter Configuration
EXPORTER_PORT = 9999

# Device Configuration
DEVICE_CONFIGS = [
    {
        "ip": "DEVICE_LOCAL_IP_1",
        "device_id": "DEVICE_ID_1",
        "local_key": "LOCAL_KEY_1"
    },
    {
        "ip": "DEVICE_LOCAL_IP_2",
        "device_id": "DEVICE_ID_2",
        "local_key": "LOCAL_KEY_2"
    }
]

# Prometheus Metrics
registry = CollectorRegistry()
metrics = {
    "current": Gauge("tuya_consumption_current", "Current in amps.", ["device_id"], registry=registry),
    "power": Gauge("tuya_consumption_power", "Power in watts.", ["device_id"], registry=registry),
    "voltage": Gauge("tuya_consumption_voltage", "Voltage in volts.", ["device_id"], registry=registry),
}

device_metrics = {config["device_id"]: {"current": float("nan"), "power": float("nan"), "voltage": float("nan")} for config in DEVICE_CONFIGS}

def update_device_metrics(device_config):
    """Continuously fetch metrics for a device in the background."""
    device_id = device_config["device_id"]
    while True:
        try:
            device = tinytuya.OutletDevice(device_config["device_id"], device_config["ip"], device_config["local_key"])
            device.set_socketTimeout(3)
            for version in [3.4, 3.3, 3.2, 3.1, 3.0]:
                try:
                    device.set_version(version)
                    device.updatedps(["18", "19", "20"])
                    data = device.status()
                    if "Error" not in data:
                        device_metrics[device_id] = {
                            "current": float(data["dps"].get("18", 0)) / 1000.0,
                            "power": float(data["dps"].get("19", 0)) / 10.0,
                            "voltage": float(data["dps"].get("20", 0)) / 10.0,
                        }
                        print(f"Updated metrics for device {device_id} using version {version}")
                        break
                except Exception:
                    continue
            else:
                raise Exception(f"Failed to connect to device {device_id}")
        except Exception as e:
            print(f"Error updating device {device_id}: {e}")
            device_metrics[device_id] = {"current": float("nan"), "power": float("nan"), "voltage": float("nan")}
        time.sleep(10)

def start_background_updater():
    """Start background threads to update metrics for all devices."""
    for config in DEVICE_CONFIGS:
        Thread(target=update_device_metrics, args=(config,), daemon=True).start()

def metrics_app(environ, start_response):
    """WSGI application for Prometheus metrics."""
    if environ["PATH_INFO"] == "/metrics":
        for device_id, metrics_data in device_metrics.items():
            metrics["current"].labels(device_id=device_id).set(metrics_data["current"])
            metrics["power"].labels(device_id=device_id).set(metrics_data["power"])
            metrics["voltage"].labels(device_id=device_id).set(metrics_data["voltage"])
        data = generate_latest(registry)
        start_response("200 OK", [("Content-type", CONTENT_TYPE_LATEST)])
        return [data]
    start_response("404 Not Found", [("Content-type", "text/plain")])
    return [b"Not Found"]

if __name__ == "__main__":
    print(f"Starting server on http://localhost:{EXPORTER_PORT}/metrics")
    start_background_updater()
    try:
        make_server("", EXPORTER_PORT, metrics_app).serve_forever()
    except KeyboardInterrupt:
        print("Shutting down server.")

