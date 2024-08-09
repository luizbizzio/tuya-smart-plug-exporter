# Tuya Smart Plug Exporter

This repository contains a simple Python script designed to monitor Tuya smart plugs and export their metrics to Prometheus. It allows for easy tracking of power consumption data, including current, power, and voltage.

## Overview 📊

The Tuya Smart Plug Exporter script offers a straightforward solution for exporting Tuya smart plug metrics to Prometheus. It collects crucial data such as current, power, and voltage from your smart plugs and makes it available for Prometheus monitoring with minimal setup.

## Features 🌟

- **Metrics Collection:** Retrieves current, power, and voltage data from Tuya smart plugs.
  
- **Prometheus Integration:** Provides metrics in a format compatible with Prometheus scraping.
  
- **Simple Configuration:** Easy to set up with minimal configuration required.
  
- **Low CPU Usage:** Designed to be efficient and have a minimal impact on system resources.

## Configuration ⚙️

1. **Device Configuration:**

   Update the `DEVICE_CONFIGS` list with your device details:

   ```python
    # Device Configuration
    DEVICE_IP = "DEVICE_LOCAL_IP"
    DEVICE_ID = "DEVICE_ID"
    LOCAL_KEY = "LOCAL_KEY"
   ```

- `DEVICE_IP`: Local IP address of your Tuya smart plug.
- `DEVICE_ID`: Unique identifier for your Tuya device.
- `LOCAL_KEY`: Authentication key for your Tuya device.

  > If you are unsure how to obtain these details, refer to [this tutorial](https://www.youtube.com/watch?v=Q1ZShFJDvE0)

2. **Exporter Port (Optional):**

   Set the port for the Prometheus exporter:

   ```python
   EXPORTER_PORT = 9999
   ```

## Usage 🚀

1. **Install Dependencies:**

   Install the required Python packages:

   ```sh
   pip install click tinytuya prometheus_client
   ```

2. **Run the Script:**

   Launch the script to start the Prometheus exporter:

   ```sh
   python tuya_exporter.py
   ```

3. **Access Metrics:**

   After running the script, you can access the metrics at:
    
     ```init
     http://localhost:9999
     ```

## Notes 📝

- **Automatic Version Detection:** The script automatically tries different versions to ensure connectivity. No manual adjustment is needed for the version.

- **Configuration:** Ensure that device IPs, IDs, and local keys are correctly configured. You can verify these settings in the configuration section of the script.

## License
This project is licensed under the MIT License.





