"""Train/test split evaluation for standalone FSE package."""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from fuzzy_system import FSEPrediction, TomatoFuzzySystem  # noqa: E402

DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "semantic_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. Copy semantic_dataset.csv vào thư mục data/processed"
        )
    df = pd.read_csv(DATASET_PATH)
    if "ground_truth" not in df.columns:
        raise ValueError("Dataset thiếu cột ground_truth.")
    print(f"Loaded {len(df)} samples from {DATASET_PATH}")
    return df


def evaluate_split_set(
    df: pd.DataFrame,
    fuzzy_system: TomatoFuzzySystem,
    set_name: str,
    enable_thresholds: bool,
    enable_fallback: bool,
) -> tuple[dict, pd.DataFrame]:
    print("=" * 60)
    print(f"Evaluating {set_name.upper()} set ({len(df)} samples)")
    print("=" * 60)

    predictions: list[int] = []
    confidences: list[float] = []
    rule_strengths: list[dict[str, float]] = []

    for idx, row in df.iterrows():
        pred: FSEPrediction = fuzzy_system.predict(
            moisture=row["Moisture"],
            ph=row["pH"],
            nitrogen=row["N"],
            temperature=row["Temperature"],
            humidity=row["Humidity"],
            ndi_label=row.get("NDI_Label"),
            pdi_label=row.get("PDI_Label"),
            enable_thresholds=enable_thresholds,
            enable_fallback=enable_fallback,
        )
        predictions.append(pred.class_id)
        confidences.append(pred.confidence)
        rule_strengths.append(pred.rule_strengths)
        if (idx + 1) % 2000 == 0:
            print(f"Processed {idx + 1}/{len(df)} samples...")

    df = df.copy()
    df["fse_prediction"] = predictions
    df["fse_confidence"] = confidences
    df["fse_rule_strengths"] = rule_strengths

    y_true = df["ground_truth"].to_numpy()
    y_pred = df["fse_prediction"].to_numpy()
    valid_mask = (y_true >= 0) & (y_pred >= 0) & (y_pred <= 7)
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]

    if len(y_true_valid) == 0:
        raise ValueError(f"Không có mẫu hợp lệ trong {set_name} set.")

    class_names = [
        "optimal",
        "nutrient_deficiency",
        "fungal_risk",
        "water_deficit_acidic",
        "water_deficit_alkaline",
        "acidic_soil",
        "alkaline_soil",
        "heat_stress",
    ]

    results = {
        "set_name": set_name,
        "accuracy": float(accuracy_score(y_true_valid, y_pred_valid)),
        "f1_macro": float(f1_score(y_true_valid, y_pred_valid, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(y_true_valid, y_pred_valid, average="weighted", zero_division=0)
        ),
        "precision_macro": float(
            precision_score(y_true_valid, y_pred_valid, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true_valid, y_pred_valid, average="macro", zero_division=0)
        ),
        "valid_predictions": int(len(y_true_valid)),
        "total_samples": int(len(df)),
        "prediction_success_rate": float(len(y_true_valid) / len(df)),
        "avg_confidence": float(np.mean(confidences)),
    }

    cm = confusion_matrix(y_true_valid, y_pred_valid, labels=range(len(class_names)))
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f"FSE Confusion Matrix - {set_name.upper()} Set")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_path = OUTPUT_DIR / f"confusion_matrix_{set_name}.png"
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

    report = classification_report(
        y_true_valid,
        y_pred_valid,
        target_names=class_names,
        labels=range(len(class_names)),
        zero_division=0,
        output_dict=True,
    )
    report_df = pd.DataFrame(report).transpose().round(4)
    report_path = OUTPUT_DIR / f"classification_report_{set_name}.csv"
    report_df.to_csv(report_path)
    print(f"Classification report saved to {report_path}")

    return results, df


def main() -> None:
    print("=" * 80)
    print("Standalone Train/Test split (80/20) for FSE")
    print("=" * 80)

    df = load_dataset()
    df_valid = df[df["ground_truth"] >= 0].copy()
    print(f"Valid samples: {len(df_valid)}/{len(df)}")

    X = df_valid.drop(columns=["ground_truth"], errors="ignore")
    y = df_valid["ground_truth"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    df_train = X_train.copy()
    df_train["ground_truth"] = y_train.values
    df_test = X_test.copy()
    df_test["ground_truth"] = y_test.values

    print(f"Train samples: {len(df_train)}")
    print(f"Test samples : {len(df_test)}")

    enable_thresholds = os.getenv("FSE_ENABLE_THRESHOLDS", "1") != "0"
    enable_fallback = os.getenv("FSE_ENABLE_FALLBACK", "1") != "0"
    fuzzy_system = TomatoFuzzySystem()

    train_results, df_train_pred = evaluate_split_set(
        df_train,
        fuzzy_system,
        "train",
        enable_thresholds,
        enable_fallback,
    )

    test_results, df_test_pred = evaluate_split_set(
        df_test,
        fuzzy_system,
        "test",
        enable_thresholds,
        enable_fallback,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "experiment": "standalone_train_test_split",
        "timestamp": timestamp,
        "config": {
            "test_size": 0.2,
            "random_state": 42,
            "stratify": True,
            "enable_thresholds": enable_thresholds,
            "enable_fallback": enable_fallback,
        },
        "train_results": train_results,
        "test_results": test_results,
    }

    summary_path = OUTPUT_DIR / f"train_test_results_{timestamp}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")

    df_train_pred.to_csv(OUTPUT_DIR / f"train_predictions_{timestamp}.csv", index=False)
    df_test_pred.to_csv(OUTPUT_DIR / f"test_predictions_{timestamp}.csv", index=False)
    print(f"Predictions saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
