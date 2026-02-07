import pandas as pd
import os
from datetime import datetime

# logs saving locations
LOG_FILE = "logs/prediction_logs.csv"

def log_prediction(input_data: dict, prediction: int, probability: float, latency: float):
    """
    Saves a single prediction event to a CSV file.
    """
    # 1. Ensure the logs folder exists
    if not os.path.exists("logs"):
        os.makedirs("logs")

    # 2. Prepare the data row
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        **input_data,  # Unpack input features (CreditScore, Age, etc.)
        "prediction": prediction,
        "probability": round(probability, 4),
        "latency_ms": round(latency * 1000, 2), # Convert seconds to milliseconds
        "model_version": "v1.0" # Ideally fetch this dynamically
    }

    # 3. Create DataFrame (1 row)
    df = pd.DataFrame([log_entry])

    # 4. Append to CSV
    # If file exists, append without header. If not, write with header.
    file_exists = os.path.isfile(LOG_FILE)
    df.to_csv(LOG_FILE, mode='a', header=not file_exists, index=False)