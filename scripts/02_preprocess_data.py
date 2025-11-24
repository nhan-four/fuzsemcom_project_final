from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "Agriculture_dataset_with_metadata.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "processed_data.csv"
REQUIRED_COLUMNS = ["Moisture", "pH", "N", "Temperature", "Humidity"]


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Không tìm thấy file {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    df_clean = df.dropna(subset=REQUIRED_COLUMNS)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(df_clean)} samples to {OUTPUT_FILE}.")


if __name__ == "__main__":
    main()

