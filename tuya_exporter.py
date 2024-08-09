import time
from typing import List
import click
import tinytuya
from prometheus_client import start_http_server
from prometheus_client.core import GaugeMetricFamily, REGISTRY

# Device Configuration
DEVICE_IP = "DEVICE_LOCAL_IP"
DEVICE_ID = "DEVICE_ID"
LOCAL_KEY = "LOCAL_KEY"

# Exporter Port
EXPORTER_PORT = 9999

MEASURES = {
    "current": 18,
    "power": 19,
    "voltage": 20
}

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
            connected = False

            for version in [3.4, 3.3, 3.2, 3.1, 3.0]:
                device.set_version(version)
                device.set_socketTimeout(2)

                device.updatedps([MEASURES["current"], MEASURES["power"], MEASURES["voltage"]])
                data = device.status()
                if "Error" not in data:
                    gauges["tuya_consumption_current"].add_metric([], float(data.get("dps", {}).get(str(MEASURES["current"]), 0)) / 1000.0)
                    gauges["tuya_consumption_power"].add_metric([], float(data.get("dps", {}).get(str(MEASURES["power"]), 0)) / 10.0)
                    gauges["tuya_consumption_voltage"].add_metric([], float(data.get("dps", {}).get(str(MEASURES["voltage"]), 0)) / 10.0)
                    connected = True
                    break

            if not connected:
                print(f"Failed to connect to device with:\nIP {config.ip}\nID {config.device_id}\nafter trying all versions.")
                return

        yield from gauges.values()

def load_config() -> List[DeviceConfig]:
    return [
        DeviceConfig(ip=DEVICE_IP, device_id=DEVICE_ID, local_key=LOCAL_KEY)
    ]

@click.command()
@click.option("--port", help="Port to run the Prometheus exporter on.", default=EXPORTER_PORT)
def main(port):
    configs = load_config()
    REGISTRY.register(Collector(configs))
    start_http_server(port)
    print(f"\nServing metrics on http://localhost:{port}/metrics")
    while True:
        time.sleep(0.1)

if __name__ == "__main__":
    main()
