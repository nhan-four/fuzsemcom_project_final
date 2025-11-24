"""Generate semantic_dataset.csv from raw agriculture data for standalone FSE."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
RAW_FILE = DATA_DIR / "raw" / "Agriculture_dataset_with_metadata.csv"
PROCESSED_FILE = DATA_DIR / "processed" / "processed_data.csv"
GROUND_TRUTH_FILE = DATA_DIR / "processed" / "ground_truth_labels.csv"
SEMANTIC_DATASET_FILE = DATA_DIR / "processed" / "semantic_dataset.csv"
REQUIRED_COLUMNS = ["Moisture", "pH", "N", "Temperature", "Humidity"]

sys.path.insert(0, str(SRC_ROOT))
from ground_truth import GroundTruthGenerator, save_ground_truth  # noqa: E402


def preprocess_raw_dataset() -> pd.DataFrame:
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file dữ liệu gốc: {RAW_FILE}. Vui lòng đặt đúng raw dataset."
        )
    df_raw = pd.read_csv(RAW_FILE)
    df_clean = df_raw.dropna(subset=REQUIRED_COLUMNS)
    PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(PROCESSED_FILE, index=False)
    print(f"Saved {len(df_clean)} samples to {PROCESSED_FILE}.")
    return df_clean


def attach_ground_truth(df: pd.DataFrame) -> pd.DataFrame:
    generator = GroundTruthGenerator()
    labels = generator.generate(df)
    df_labeled = save_ground_truth(df, labels, str(GROUND_TRUTH_FILE))
    df_labeled.to_csv(SEMANTIC_DATASET_FILE, index=False)
    print(
        "Semantic dataset saved to {file} with {n} labeled samples.".format(
            file=SEMANTIC_DATASET_FILE, n=len(df_labeled)
        )
    )
    dropped = len(df) - len(df_labeled)
    if dropped:
        print(f"Dropped {dropped} samples with ground_truth = -1.")
    return df_labeled


def main() -> None:
    print("=== Step 1/2: Preprocess raw dataset ===")
    df_clean = preprocess_raw_dataset()

    print("=== Step 2/2: Generate ground-truth labels & semantic dataset ===")
    attach_ground_truth(df_clean)
    print("Hoàn tất: semantic_dataset.csv đã sẵn sàng cho các script evaluate.")


if __name__ == "__main__":
    main()
