from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from ground_truth import GroundTruthGenerator, save_ground_truth  # noqa: E402

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "processed_data.csv"
GROUND_TRUTH_FILE = PROJECT_ROOT / "data" / "processed" / "ground_truth_labels.csv"
SEMANTIC_DATASET_FILE = PROJECT_ROOT / "data" / "processed" / "semantic_dataset.csv"


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {INPUT_FILE}. Vui lòng chạy scripts/02_preprocess_data.py trước."
        )
    df = pd.read_csv(INPUT_FILE)
    generator = GroundTruthGenerator()
    labels = generator.generate(df)
    df_labeled = save_ground_truth(df, labels, GROUND_TRUTH_FILE)
    df_labeled.to_csv(SEMANTIC_DATASET_FILE, index=False)
    print(f"Labeled samples: {len(df_labeled)} saved to {GROUND_TRUTH_FILE}.")
    print(f"Semantic dataset saved to {SEMANTIC_DATASET_FILE}.")
    dropped = len(df) - len(df_labeled)
    if dropped:
        print(f"Dropped {dropped} samples with ground_truth = -1.")


if __name__ == "__main__":
    main()

