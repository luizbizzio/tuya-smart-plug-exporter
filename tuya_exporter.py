import time
from typing import List
import click
import tinytuya
from prometheus_client import start_http_server
from prometheus_client.core import GaugeMetricFamily, REGISTRY

# Port to run the Prometheus exporter on
EXPORTER_PORT = 9999

# Replace the placeholder values with your device's local IP address, device ID, and local key
DEVICE_CONFIGS = [
    DeviceConfig(ip="DEVICE_LOCAL_IP", device_id="DEVICE_ID", local_key="DEVICE_LOCAL_KEY")
]

class DeviceConfig:
    def __init__(self, ip: str, device_id: str, local_key: str):
        self.ip = ip
        self.device_id = device_id
        self.local_key = local_key

class Collector:
    def __init__(self, configs: List[DeviceConfig]):
        self.configs = configs

    def collect(self):
        gauges = {
            "tuya_consumption_current": GaugeMetricFamily("tuya_consumption_current", "Current in amps."),
            "tuya_consumption_power": GaugeMetricFamily("tuya_consumption_power", "Power in watts."),
            "tuya_consumption_voltage": GaugeMetricFamily("tuya_consumption_voltage", "Voltage in volts.")
        }
        for config in self.configs:
            device = tinytuya.OutletDevice(config.device_id, config.ip, config.local_key)
            device.set_version(3.4) # If you encounter connection issues, try changing the version to 3.0, 3.1, 3.2, or 3.3
            device.set_socketTimeout(2)
            device.updatedps([18, 19, 20]) 
            data = device.status()
            if "Error" not in data:
                gauges["tuya_consumption_current"].add_metric([], float(data.get("dps", {}).get("18", 0)) / 1000.0)
                gauges["tuya_consumption_power"].add_metric([], float(data.get("dps", {}).get("19", 0)) / 10.0)
                gauges["tuya_consumption_voltage"].add_metric([], float(data.get("dps", {}).get("20", 0)) / 10.0)

        yield from gauges.values()

@click.command()
@click.option("--port", help="Port to run the Prometheus exporter on.", default=EXPORTER_PORT)
def main(port):
    REGISTRY.register(Collector(DEVICE_CONFIGS))
    start_http_server(port)
    print(f"Serving metrics on port {port}")
    while True:
        time.sleep(0.1)

if __name__ == "__main__":
    main()
