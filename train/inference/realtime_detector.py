import time
import sys
import pandas as pd

from predict_ml import load_best_ml_model, preprocess_new_data


def realtime_detect(dataset_name, csv_path, delay=0.5, limit=20):
    model, scaler, encoder, preprocess_info, model_path = load_best_ml_model(dataset_name)

    print("Realtime detector started")
    print("Dataset:", dataset_name)
    print("Model:", model_path)
    print("=" * 60)

    df = pd.read_csv(csv_path)

    if limit is not None:
        df = df.head(limit)

    for idx, row in df.iterrows():
        one_row = pd.DataFrame([row])

        X = preprocess_new_data(one_row, scaler, encoder, preprocess_info)

        pred = model.predict(X)[0]

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X)[0][1]
        else:
            prob = None

        status = "ATTACK" if pred == 1 else "NORMAL"

        print(f"[{idx}] Prediction: {status}", end="")

        if prob is not None:
            print(f" | Attack probability: {prob:.4f}")
        else:
            print()

        time.sleep(delay)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("python inference/realtime_detector.py <dataset_name> <csv_path>")
        sys.exit(1)

    dataset_name = sys.argv[1]
    csv_path = sys.argv[2]

    realtime_detect(dataset_name, csv_path)