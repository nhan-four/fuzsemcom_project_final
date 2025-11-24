"""Standalone evaluation script for FSE without external deps."""
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

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

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from fuzzy_system import FSEPrediction, TomatoFuzzySystem  # noqa: E402

DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "semantic_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR.mkdir(exist_ok=True)


@dataclass
class FSEEvaluator:
    fuzzy_system: TomatoFuzzySystem
    enable_thresholds: bool = True
    enable_fallback: bool = True

    def __post_init__(self) -> None:
        self.class_names = [
            "optimal",
            "nutrient_deficiency",
            "fungal_risk",
            "water_deficit_acidic",
            "water_deficit_alkaline",
            "acidic_soil",
            "alkaline_soil",
            "heat_stress",
        ]

    def evaluate(self, df: pd.DataFrame) -> Tuple[Dict[str, float], pd.DataFrame]:
        print("Evaluating Fuzzy Semantic Encoder Performance...")
        print("=" * 70)

        predictions: list[int] = []
        confidences: list[float] = []
        rule_strengths: list[Dict[str, float]] = []
        prediction_times: list[float] = []

        for idx, row in df.iterrows():
            start = time.time()
            pred: FSEPrediction = self.fuzzy_system.predict(
                moisture=row["Moisture"],
                ph=row["pH"],
                nitrogen=row["N"],
                temperature=row["Temperature"],
                humidity=row["Humidity"],
                ndi_label=row.get("NDI_Label"),
                pdi_label=row.get("PDI_Label"),
                enable_thresholds=self.enable_thresholds,
                enable_fallback=self.enable_fallback,
            )
            predictions.append(pred.class_id)
            confidences.append(pred.confidence)
            rule_strengths.append(pred.rule_strengths)
            prediction_times.append(time.time() - start)

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
            raise ValueError("Không có mẫu hợp lệ sau khi lọc ground_truth/prediction.")

        results = {
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
            "avg_prediction_time": float(np.mean(prediction_times)),
            "total_prediction_time": float(np.sum(prediction_times)),
            "avg_confidence": float(np.mean(confidences)),
        }

        self._print_results(results)
        self._plot_confusion_matrix(y_true_valid, y_pred_valid)
        self._save_classification_report(y_true_valid, y_pred_valid)

        return results, df

    def _print_results(self, results: Dict[str, float]) -> None:
        print("\nPERFORMANCE SUMMARY")
        print("-" * 70)
        print(f"Accuracy           : {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print(f"F1-Score (Macro)  : {results['f1_macro']:.4f}")
        print(f"F1-Score (Weighted): {results['f1_weighted']:.4f}")
        print(f"Precision (Macro) : {results['precision_macro']:.4f}")
        print(f"Recall (Macro)    : {results['recall_macro']:.4f}")
        print(f"Average Confidence: {results['avg_confidence']:.4f}")
        print(
            f"Valid Predictions  : {results['valid_predictions']} / {results['total_samples']}"
        )

    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        cm = confusion_matrix(y_true, y_pred, labels=range(len(self.class_names)))
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
        )
        plt.title("FSE Confusion Matrix")
        plt.xlabel("Predicted Class")
        plt.ylabel("True Class")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        path = OUTPUT_DIR / "confusion_matrix_full.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Confusion matrix saved to {path}")

    def _save_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        report = classification_report(
            y_true,
            y_pred,
            target_names=self.class_names,
            labels=range(len(self.class_names)),
            zero_division=0,
            output_dict=True,
        )
        report_df = pd.DataFrame(report).transpose().round(4)
        path = OUTPUT_DIR / "classification_report_full.csv"
        report_df.to_csv(path)
        print(f"Classification report saved to {path}")


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


def main() -> None:
    df = load_dataset()
    enable_thresholds = os.getenv("FSE_ENABLE_THRESHOLDS", "1") != "0"
    enable_fallback = os.getenv("FSE_ENABLE_FALLBACK", "1") != "0"

    evaluator = FSEEvaluator(
        TomatoFuzzySystem(),
        enable_thresholds=enable_thresholds,
        enable_fallback=enable_fallback,
    )

    start = datetime.now()
    results, df_with_predictions = evaluator.evaluate(df)

    results_path = OUTPUT_DIR / "full_evaluation_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    predictions_path = OUTPUT_DIR / "predictions_full.csv"
    df_with_predictions.to_csv(predictions_path, index=False)
    print(f"Predictions saved to {predictions_path}")

    duration = datetime.now() - start
    print(f"\nTotal runtime: {duration.total_seconds():.2f} seconds")


if __name__ == "__main__":
    import os

    main()
