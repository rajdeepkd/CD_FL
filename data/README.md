# Data Directory

Place your per-client CSVs in `data/raw_data/`.

Each CSV = one client. Example:
- `data/raw_data/client1.csv`
- `data/raw_data/client2.csv`

## Expected CSV Format

- One row per sample.
- Columns = features + label.
- The label column should ideally be binary (0/1, benign/malicious, etc.).

### Label detection

If you do not specify `data.label_column` in `configs/default.yaml`,
the code will try to auto-detect using a case-insensitive match from:
- `label`, `y`, `target`, `class`

If none of those are found, you must set `data.label_column` explicitly.

### Feature detection

The pipeline will:
- Guess which columns are categorical vs numeric.
- Handle missing values.
- Encode categorical columns.
- Scale numeric columns.

Resulting processed data for each client will be saved into:
`data/processed/<client_name>.csv`

We also log class distributions (pre/post SMOTE) into:
`results/summaries/<client_name>_class_distribution.txt`

## Privacy / Safety

Do not put sensitive personal data here unless you have the right to process it.
The framework is intended for defensive cybersecurity / anomaly detection research.
