"""Tương thích lệnh cũ. Pipeline mới tự nhận diện loại best model."""
import sys
from predict import predict_csv

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python inference/predict_ml.py <experiment_name> <csv_path>")
        raise SystemExit(1)
    predict_csv(sys.argv[1], sys.argv[2])
