# Contributing to Tuya Smart Plug Exporter

Install the dependencies with `python -m pip install -r requirements.txt` in a virtual environment. Start with [`config.example.yaml`](config.example.yaml) and the [README](README.md). The default Prometheus endpoint is `http://localhost:9122/metrics`.

## Review a change

- Use synthetic TinyTuya responses to check DPS discovery and scaling. Cover a response missing one of voltage, current, power, or relay state and a device that stops responding.
- Compare the `/metrics` scrape for units, labels, and values. Keep device identifiers, IP addresses, and local keys out of examples and fixtures.
- For container or configuration changes, verify that [`compose.yaml`](compose.yaml) can load a test configuration.

Include the device model or protocol version if relevant, a sanitized example of the DPS layout, and the metric values you expect.
