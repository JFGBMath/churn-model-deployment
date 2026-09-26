import json
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = Path("logs/predictions.log")


def log_prediction(input_data: dict, prediction: str, probability: float):

    """
    Appends one prediction event to the log file as a JSON line
    """
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": input_data,
        "prediction": prediction,
        "probability": probability,
    }

    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")