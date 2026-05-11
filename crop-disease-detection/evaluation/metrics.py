from pathlib import Path
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute evaluation metrics from CSV.")
    parser.add_argument("--csv", type=Path, required=True, help="CSV with y_true and y_pred columns")
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)
    if "y_true" not in df.columns or "y_pred" not in df.columns:
        raise RuntimeError("CSV must include columns: y_true, y_pred")

    y_true = df["y_true"]
    y_pred = df["y_pred"]

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)


if __name__ == "__main__":
    main()
