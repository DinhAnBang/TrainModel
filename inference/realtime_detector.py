import sys
import time
from pathlib import Path

import pandas as pd

from predict import predict_csv


def realtime_detect(experiment_name, csv_path, delay=0.5, limit=20):
    df = pd.read_csv(csv_path, low_memory=False)
    if limit is not None:
        df = df.head(limit)
    temp_path = Path(f"_realtime_temp_{experiment_name}.csv")
    try:
        for idx in range(len(df)):
            df.iloc[[idx]].to_csv(temp_path, index=False)
            result = predict_csv(experiment_name, str(temp_path))
            pred = result.iloc[0]["prediction"]
            probability = result.iloc[0].get("prediction_probability")
            text = f"[{idx}] Prediction: {pred}"
            if pd.notna(probability):
                text += f" | Confidence: {float(probability):.4f}"
            print(text)
            time.sleep(delay)
    finally:
        temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python inference/realtime_detector.py <experiment_name> <csv_path>")
        raise SystemExit(1)
    realtime_detect(sys.argv[1], sys.argv[2])
