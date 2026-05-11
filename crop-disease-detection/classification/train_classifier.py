from pathlib import Path
import argparse
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def default_features_csv() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    return project_root / "dataset" / "labels" / "features.csv"


def default_model_path() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    return project_root / "runs" / "classify" / "svm_model.joblib"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train crop disease classifier.")
    parser.add_argument("--features", type=Path, default=default_features_csv(), help="Feature CSV path")
    parser.add_argument("--model-out", type=Path, default=default_model_path(), help="Output model path")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if not args.features.exists():
        raise FileNotFoundError(f"Feature CSV not found: {args.features}")

    df = pd.read_csv(args.features)
    if "label" not in df.columns:
        raise RuntimeError("Feature CSV must include a 'label' column.")

    feature_cols = [c for c in df.columns if c.startswith("f")]
    if not feature_cols:
        raise RuntimeError("No feature columns found. Expected columns like f1, f2, ...")

    X = df[feature_cols]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(kernel="rbf", C=10.0, gamma="scale", probability=True)),
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.4f}")
    print("Classification report:")
    print(classification_report(y_test, preds, zero_division=0))

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "feature_cols": feature_cols}, args.model_out)
    print(f"Model saved to: {args.model_out}")


if __name__ == "__main__":
    main()
