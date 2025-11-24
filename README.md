# Standalone FSE Package

Gói này đóng gói toàn bộ logic Fuzzy Semantic Encoder (FSE) theo paper *Fixed ICC_ENGLAND_2026.pdf*. Reviewer chỉ cần copy `semantic_dataset.csv` và chạy các script dưới đây để tái hiện kết quả Step 5–6 của paper.

## 1. Cấu trúc
```
standalone_fse/
├── README.md
├── requirements.txt
├── prepare_semantic_dataset.py # Helper: gộp Step 2+4
├── run_full_eval.py           # Step 5 – evaluate toàn bộ semantic dataset
├── run_train_test_eval.py     # Step 6 – train/test split 80/20
├── scripts/                   # Bản copy thư mục scripts gốc (01→06, report)
├── data/
│   ├── raw/
│   │   └── Agriculture_dataset_with_metadata.csv
│   └── processed/
│       ├── processed_data.csv
│       ├── ground_truth_labels.csv
│       └── semantic_dataset.csv
├── results/                   # tự tạo khi chạy
└── src/
    ├── __init__.py
    ├── fuzzy_system.py        # TomatoFuzzySystem (Table II của paper)
    └── ground_truth.py        # GroundTruthGenerator (Section III-B)
```

## 2. Chuẩn bị dữ liệu & pipeline end-to-end

Thư mục này đã kèm sẵn dataset gốc (`data/raw/Agriculture_dataset_with_metadata.csv`). Có hai cách chạy:

1. **Theo từng bước giống paper (scripts 01 → 06):**
   ```bash
   python scripts/01_explore_data.py
   python scripts/02_preprocess_data.py
   python scripts/03_fuzzy_definitions.py   # optional kiểm tra MF
   python scripts/04_ground_truth.py
   python scripts/05_evaluate_fse.py
   python scripts/06_deepsc_comparison.py
   ```

2. **Workflow rút gọn cho reviewer:**
   ```bash
   python prepare_semantic_dataset.py   # gộp Step 2 + 4
   python run_full_eval.py              # Step 5
   python run_train_test_eval.py        # Train/Test split 80/20
   ```

Sau khi chạy Step 4 (hoặc `prepare_semantic_dataset.py`), các file `processed_data.csv`, `ground_truth_labels.csv`, `semantic_dataset.csv` sẽ có trong `data/processed/`.

## 3. Cài đặt môi trường
```bash
cd standalone_fse
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 4. Mapping giữa paper và code

| Paper ICC 2026 | File/code trong gói | Ghi chú đối chiếu |
|----------------|---------------------|-------------------|
| Step 3 – Membership & 8 semantic classes (Table II) | `src/fuzzy_system.py` | Các `_trimf` và `SEMANTIC_CLASSES` giữ nguyên breakpoint, danh sách lớp khớp Table II. |
| Section III-B – Ground-truth synthesis (NDI/PDI tags) | `prepare_semantic_dataset.py`, `src/ground_truth.py` | Pipeline raw → processed → semantic dataset, sử dụng đúng rule list trong paper. |
| Step 4 – Mamdani inference + NDI/PDI heuristic + crisp fallback | `src/fuzzy_system.py` | Block rules và fallback mapping tái hiện đúng Figure 3/Section III-C. |
| Step 5 – Evaluate toàn bộ dataset | `run_full_eval.py` | Tính Accuracy, F1, Precision, Recall, confusion matrix, classification report. |
| Step 6 – Train/Test split 80/20 | `run_train_test_eval.py` | Stratified split, export kết quả JSON + hình + report cho train/test. |

Trích dẫn code chính:

```72:147:experiments/standalone_fse/src/fuzzy_system.py
m_moisture = {"dry": _trimf(moisture, 15, 20, 30), ...}
...
strengths["nutrient_deficiency"] = max(
    strengths["nutrient_deficiency"], min(m_n["low"], m_ph["acidic"])
)
```

```149:226:experiments/standalone_fse/src/fuzzy_system.py
if ndi_label == "High":
    strengths["nutrient_deficiency"] = max(strengths["nutrient_deficiency"], 1.0)
...
if pdi_label == "High":
    return "fungal_risk"
return "optimal"
```

## 5. Chạy Step 5 – Full evaluation
```bash
python run_full_eval.py
```
Output (lưu trong `results/`):
- `full_evaluation_results.json`
- `predictions_full.csv`
- `confusion_matrix_full.png`
- `classification_report_full.csv`

Ví dụ số liệu gần nhất (khớp báo cáo >94.8%):
```1:13:experiments/standalone_fse/results/full_evaluation_results.json
{
  "accuracy": 0.9485,
  "f1_macro": 0.9123,
  "precision_macro": 0.9156,
  "recall_macro": 0.9204
}
```

## 6. Chạy Step 6 – Train/Test split 80/20
```bash
python run_train_test_eval.py
```
Artefact sinh ra:
- `train_test_results_<timestamp>.json`
- `train_predictions_<timestamp>.csv`, `test_predictions_<timestamp>.csv`
- `confusion_matrix_train.png`, `confusion_matrix_test.png`
- `classification_report_train.csv`, `classification_report_test.csv`

## 7. Tùy chỉnh
- Tắt thresholds: `FSE_ENABLE_THRESHOLDS=0 python run_full_eval.py`
- Tắt fallback: `FSE_ENABLE_FALLBACK=0 python run_full_eval.py`

## 8. Ghi chú
- `results/` được tạo tự động mỗi lần chạy, tiện gửi kèm submission.
- Có thể copy nguyên thư mục này sang repo khác mà không cần các script còn lại.
- Nếu cập nhật phiên bản FSE mới, chỉ cần thay `src/fuzzy_system.py` rồi chạy lại hai script trên.
