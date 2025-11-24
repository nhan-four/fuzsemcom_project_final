from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = PROJECT_ROOT / "data" / "raw" / "Agriculture_dataset_with_metadata.csv"


def main() -> None:
    if not RAW_DATA.exists():
        raise FileNotFoundError(f"Không tìm thấy file {RAW_DATA}")
    df = pd.read_csv(RAW_DATA)
    print("Columns:", df.columns.tolist())
    preview_columns = [
        "Moisture",
        "pH",
        "N",
        "Temperature",
        "Humidity",
        "NDI_Label",
        "PDI_Label",
    ]
    existing = [col for col in preview_columns if col in df.columns]
    print(df[existing].head())


if __name__ == "__main__":
    main()

