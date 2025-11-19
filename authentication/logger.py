import requests
import time
import json 

USER_ID = 0000000  
API_KEY = "XXX"  

def send_log(event: str, level: str = None, extra: dict = None):
    """
    Send a log message to Grafana Cloud Loki with additional fields.

    Parameters:
        event (str): The main log message.
        level (str, optional): The log level, e.g., 'debug', 'info', 'error'.
        extra (dict, optional): A dictionary containing additional fields that will be included in the log data.
    """

    if not event:
        print("The 'event' field is required.")
        return
    
    extra_string = json.dumps(extra) if extra else "{}"

    # Default log structure
    log_data = {
        "streams": [
            {
                "stream": {"event": event},
                "values": [
                    [
                        str(int(time.time()) * 1000000000), 
                        extra_string
                    ]
                ]
            }
        ]
    }

    # Add level to stream if provided
    if level:
        log_data["streams"][0]["stream"]["level"] = level

    # Send the log data to Grafana Loki
    try:
        response = requests.post(
            url="https://REMOVED/loki/api/v1/push",
            auth=(USER_ID, API_KEY),
            json=log_data,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()  # Will raise an exception for HTTP errors
        print(f"Log successfully sent: {event} {extra_string}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to send log: {e}")

# Separate functions for different log levels

def log_debug(event: str, extra: dict = None):
    """Send a debug log."""
    send_log(event, level="debug", extra=extra)

def log_info(event: str, extra: dict = None):
    """Send an info log."""
    send_log(event, level="info", extra=extra)

def log_error(event: str, extra: dict = None):
    """Send an error log."""
    send_log(event, level="error", extra=extra)
