import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Coloque o CSV do Kaggle em {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


def find_best_threshold(y_true, proba) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    # precision/recall têm um elemento a mais que thresholds
    return float(thresholds[f1[:-1].argmax()])


def main():
    df = load_data()
    print(f"Transações: {len(df)} | Fraudes: {df['Class'].sum()} "
          f"({df['Class'].mean():.3%})")

    X = df.drop(columns="Class")
    y = df["Class"]

    # 60% treino, 20% validação (escolher modelo e threshold), 20% teste
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.25, random_state=42, stratify=y_tmp
    )

    models = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=42
        ),
    }

    val_scores, fitted = {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        proba_val = model.predict_proba(X_val)[:, 1]
        val_scores[name] = average_precision_score(y_val, proba_val)
        fitted[name] = model
        print(f"{name}: PR-AUC (validação) = {val_scores[name]:.4f}")

    best_name = max(val_scores, key=val_scores.get)
    best_model = fitted[best_name]
    print(f"\nMelhor modelo: {best_name}")

    # Threshold escolhido na validação, nunca no teste
    threshold = find_best_threshold(
        y_val, best_model.predict_proba(X_val)[:, 1]
    )

    proba_test = best_model.predict_proba(X_test)[:, 1]
    preds = (proba_test >= threshold).astype(int)

    metrics = {
        "best_model": best_name,
        "threshold": round(threshold, 4),
        "pr_auc_test": round(average_precision_score(y_test, proba_test), 4),
        "precision_test": round(precision_score(y_test, preds), 4),
        "recall_test": round(recall_score(y_test, preds), 4),
        "f1_test": round(f1_score(y_test, preds), 4),
        "pr_auc_validation": {k: round(v, 4) for k, v in val_scores.items()},
    }
    print(json.dumps(metrics, indent=2))

    joblib.dump(
        {"model": best_model, "threshold": threshold},
        BASE_DIR / "model.joblib",
    )
    with open(BASE_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    PrecisionRecallDisplay.from_predictions(
        y_test, proba_test, name=best_name
    )
    plt.title("Curva Precision-Recall (teste)")
    plt.savefig(BASE_DIR / "pr_curve.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()