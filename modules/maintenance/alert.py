# maintenance/alert.py

import os
import json
import datetime
import httpx
from dotenv import load_dotenv

load_dotenv()

WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL", "").strip()
ALERT_LOG_FILE = "alert_fallback.log"

def log_alert(data: dict):
    """
    Log alert locally and (optionally) send webhook notification.
    """
    timestamp = datetime.datetime.utcnow().isoformat() + "Z"
    entry = {
        "timestamp": timestamp,
        "event": data.get("type", "unknown"),
        "model": data.get("model"),
        "provider": data.get("provider"),
        "user_prompt": data.get("user_prompt"),
        "raw_output_preview": data.get("raw_output", "")[:300],
    }

    # Save to log file
    with open(ALERT_LOG_FILE, "a") as log:
        log.write(json.dumps(entry) + "\n")

    # Optionally send to webhook
    if WEBHOOK_URL:
        try:
            httpx.post(WEBHOOK_URL, json=entry, timeout=5)
            print("📡 Alert webhook sent.")
        except Exception as e:
            print(f"⚠️ Failed to send webhook: {e}")
    else:
        print("📎 Alert logged locally only (webhook disabled).")
