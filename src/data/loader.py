import os
from typing import Dict, Tuple, List, Optional, Any
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE


# You can tune these after you see memory usage.
# Smaller = less memory, more collisions.
HASH_BUCKETS_PER_CAT = {
    "protocol": 32,
    "service": 64,
    "conn_state": 16,
    "local_orig": 4,
    "local_resp": 4,
    "history": 64,
}
DEFAULT_HASH_BUCKETS = 32  # fallback if a column isn't listed



def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _desired_client_mapping() -> List[Tuple[str, str]]:
    """
    Explicit, fixed client->file mapping.
    Order matters and is hard-coded to guarantee stable client IDs.

    client1 -> CIC_IoMT_2024.csv
    client2 -> CIC_IoT_2023.csv
    client3 -> Edge_IIoT.csv
    client4 -> IoT_23.csv
    client5 -> MedBIoT.csv
    """
    return [
        ("client1", "CIC_IoMT_2024.csv"),
        ("client2", "CIC_IoT_2023.csv"),
        ("client3", "Edge_IIoT.csv"),
        ("client4", "IoT_23.csv"),
        ("client5", "MedBIoT.csv"),
    ]


def _load_all_clients_raw(raw_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load CSVs from raw_dir according to the fixed mapping above.
    This guarantees:
      - client1 is always CIC_IoMT_2024.csv,
      - client2 is always CIC_IoT_2023.csv,
      etc.

    If any file is missing, we raise an error with a helpful message.
    """
    _ensure_dir(raw_dir)

    mapping = _desired_client_mapping()
    clients: Dict[str, pd.DataFrame] = {}

    for client_id, filename in mapping:
        path = os.path.join(raw_dir, filename)
        if not os.path.isfile(path):
            raise RuntimeError(
                f"Expected '{filename}' for {client_id} in {raw_dir}, but it was not found.\n"
                f"Please place the file at: {path}"
            )
        df = pd.read_csv(path)
        clients[client_id] = df

    return clients


def _apply_label_mapping(df: pd.DataFrame, label_col: str) -> pd.Series:
    # BenignTraffic -> 0, else -> 1
    return (
        df[label_col]
        .apply(lambda x: 0 if str(x) == "BenignTraffic" else 1)
        .astype(int)
    )


def _count_sublabels(df: pd.DataFrame) -> Dict[str, int]:
    if "Sub_Label" not in df.columns:
        return {}
    counts = Counter(df["Sub_Label"].astype(str).tolist())
    return dict(counts)


def _drop_columns(df: pd.DataFrame, cols_to_drop: List[str], label_column: str) -> pd.DataFrame:
    """
    Drop everything in cols_to_drop EXCEPT the label column.
    We keep the label column even if it's in cols_to_drop by accident.
    """
    drop_actual = [c for c in cols_to_drop if c in df.columns and c != label_column]
    return df.drop(columns=drop_actual, errors="ignore")


def _train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified split if both classes exist.
    Otherwise do a simple tail-split.
    """
    if len(np.unique(y)) == 1:
        n = len(y)
        n_test = int(max(1, round(test_size * n)))
        idx_test = np.arange(n - n_test, n)
        idx_train = np.arange(0, n - n_test)
    else:
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        idx_train, idx_test = next(splitter.split(X, y))
    return (
        X.iloc[idx_train].reset_index(drop=True),
        X.iloc[idx_test].reset_index(drop=True),
        y.iloc[idx_train].reset_index(drop=True),
        y.iloc[idx_test].reset_index(drop=True),
    )

import hashlib

def _hash_to_bucket(val: str, num_buckets: int) -> int:
    """
    Deterministic string -> bucket index in [0, num_buckets).
    Uses md5 so result is stable across processes/OS.
    """
    h = hashlib.md5(val.encode("utf-8")).hexdigest()
    # take first 8 hex chars -> int -> mod
    bucket = int(h[:8], 16) % num_buckets
    return bucket


def _clean_categorical_column(series: pd.Series) -> pd.Series:
    """
    Normalize categorical values:
    - NaN, empty string, "-", " " => "NA_CAT"
    - Otherwise cast to string.
    """
    if series is None:
        return pd.Series([], dtype=str)

    s = series.astype(str).str.strip()
    s = s.replace(to_replace=["", "-", "nan", "None", "NA", "N/A"], value="NA_CAT")
    # After .replace(), anything left that's literally '' (empty) or '-' will already map.
    s = s.fillna("NA_CAT")
    return s

def _clean_numeric_columns(
    df: pd.DataFrame,
    numeric_cols: List[str],
) -> pd.DataFrame:
    """
    Coerce numeric columns to float.
    Replace invalid entries like "", "-", "NA", etc. with NaN (via to_numeric),
    but DO NOT impute here. We'll impute later using train median so we can
    reuse the same imputer for test.
    """
    df = df.copy()
    for col in numeric_cols:
        if col not in df.columns:
            # if column totally missing in this client, create it now as all NaN
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _build_global_categorical_vocab(
    clients_raw: Dict[str, pd.DataFrame],
    cat_cols: List[str],
) -> Dict[str, List[str]]:
    """
    Build a global vocab for each categorical column, AFTER cleaning.
    "-" or "" etc. will map to "NA_CAT", so they won't explode vocab.
    """
    vocab: Dict[str, set] = {c: set() for c in cat_cols}
    for df in clients_raw.values():
        for c in cat_cols:
            if c in df.columns:
                cleaned = _clean_categorical_column(df[c])
                vals = cleaned.astype(str).fillna("NA_CAT").unique().tolist()
                vocab[c].update(vals)
            else:
                # column missing entirely in this dataset
                vocab[c].add("NA_CAT")
    # ensure deterministic order
    return {c: sorted(list(vocab[c])) for c in cat_cols}

# def _one_hot_with_vocab_fast(
#     df: pd.DataFrame,
#     cat_cols: List[str],
#     vocab: Dict[str, List[str]],
# ) -> pd.DataFrame:
#     """
#     Efficient one-hot using cleaned categorical values.
#     For each c in cat_cols:
#       - normalize values with _clean_categorical_column (NA, blank, "-" -> "NA_CAT")
#       - map to integer codes
#       - create dense one-hot block
#     """
#     n_rows = len(df)
#     blocks = []
#     block_colnames = []

#     for c in cat_cols:
#         if c in df.columns:
#             col_vals = _clean_categorical_column(df[c]).to_numpy()
#         else:
#             col_vals = np.array(["NA_CAT"] * n_rows, dtype=str)

#         vocab_list = vocab[c]
#         idx_map = {cat_val: idx for idx, cat_val in enumerate(vocab_list)}

#         # codes[i] = index of the vocab entry OR -1 if unseen
#         codes = np.full(shape=(n_rows,), fill_value=-1, dtype=np.int32)
#         for i, val in enumerate(col_vals):
#             codes[i] = idx_map.get(val, -1)

#         block = np.zeros((n_rows, len(vocab_list)), dtype=np.float32)
#         valid_mask = codes >= 0
#         block[valid_mask, codes[valid_mask]] = 1.0

#         colnames = [f"{c}__{v}" for v in vocab_list]
#         block_df = pd.DataFrame(block, columns=colnames, index=df.index)

#         blocks.append(block_df)
#         block_colnames.extend(colnames)

#     if blocks:
#         cat_df = pd.concat(blocks, axis=1)
#     else:
#         cat_df = pd.DataFrame(index=df.index)

#     return cat_df

def _hashed_one_hot_block(
    df: pd.DataFrame,
    cat_cols: List[str],
) -> pd.DataFrame:
    """
    For each categorical column c:
      - clean values (blank/'-' -> 'NA_CAT')
      - hash each string to a bucket in [0, HASH_BUCKETS_PER_CAT[c])
      - produce one-hot over those buckets
    Returns a DataFrame with columns in deterministic order:
      [ f"{c}__bucket_{0}", f"{c}__bucket_{1}", ..., ]
    so all clients share the same exact dimensions.
    """
    n_rows = len(df)
    blocks = []

    for c in cat_cols:
        num_buckets = HASH_BUCKETS_PER_CAT.get(c, DEFAULT_HASH_BUCKETS)

        # clean values
        if c in df.columns:
            col_vals = _clean_categorical_column(df[c]).to_numpy()
        else:
            col_vals = np.array(["NA_CAT"] * n_rows, dtype=str)

        # map each row to an integer bucket
        codes = np.empty(shape=(n_rows,), dtype=np.int32)
        for i, v in enumerate(col_vals):
            codes[i] = _hash_to_bucket(v, num_buckets)

        # create dense one-hot [n_rows, num_buckets]
        block = np.zeros((n_rows, num_buckets), dtype=np.float32)
        block[np.arange(n_rows), codes] = 1.0  # fully vectorized

        colnames = [f"{c}__bucket_{b}" for b in range(num_buckets)]
        block_df = pd.DataFrame(block, columns=colnames, index=df.index)

        blocks.append(block_df)

    if blocks:
        cat_df = pd.concat(blocks, axis=1)
    else:
        cat_df = pd.DataFrame(index=df.index)

    return cat_df


def _one_hot_with_vocab(
    df: pd.DataFrame,
    cat_cols: List[str],
    vocab: Dict[str, List[str]],
) -> pd.DataFrame:
    """
    For each categorical col 'c':
    - generate columns c__<category> for every category in vocab[c]
    - fill them with 1.0 or 0.0 for each row
    If df[c] is missing, treat all rows as "NA_CAT".
    """
    out_df = df.copy()
    for c in cat_cols:
        if c in out_df.columns:
            col_vals = out_df[c].astype(str).fillna("NA_CAT")
        else:
            col_vals = pd.Series(["NA_CAT"] * len(out_df), index=out_df.index)

        for v in vocab[c]:
            new_col = f"{c}__{v}"
            out_df[new_col] = (col_vals == v).astype(float)

        # drop original category col
        if c in out_df.columns:
            out_df = out_df.drop(columns=[c])

    return out_df


def _scale_numeric(
    df: pd.DataFrame,
    num_cols: List[str],
    scaler_type: str,
    scaler_fit: Optional[Any] = None,
) -> Tuple[pd.DataFrame, Any]:
    """
    Fit scaler on train, reuse on test.
    scaler_type: "standard" or "minmax"
    """
    out_df = df.copy()
    if not num_cols:
        return out_df, None

    if scaler_fit is None:
        scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
        scaler_fit = scaler.fit(out_df[num_cols])
    out_df[num_cols] = scaler_fit.transform(out_df[num_cols])
    return out_df, scaler_fit


def _smote_if_needed(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    use_smote: bool,
    threshold: float,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, int], Dict[str, int]]:
    """
    Apply SMOTE to the TRAIN split if:
    - multiple classes exist AND
    - minority ratio < threshold
    We return before/after class counts for logging.
    """
    before = Counter(y_train.tolist())

    if len(before) > 1:
        total = sum(before.values())
        minority = min(before.values())
        ratio = minority / total
    else:
        ratio = 1.0

    if use_smote and ratio < threshold and len(before) > 1:
        sm = SMOTE()
        Xb, yb = sm.fit_resample(X_train, y_train)
        after = Counter(yb.tolist())
        return Xb, yb, dict(before), dict(after)
    else:
        return X_train, y_train, dict(before), dict(before)


def _log_client_stats(
    client_id: str,
    out_dir: str,
    sublabel_counts: Dict[str, int],
    train_counts_before: Dict[str, int],
    train_counts_after: Dict[str, int],
    test_counts: Dict[str, int],
) -> None:
    """
    Initial log for each client, before appending final test metrics.
    We include:
    - Sub_Label distribution (raw)
    - Benign/malicious counts in train before/after SMOTE
    - Benign/malicious counts in test (no SMOTE)
    """
    path = os.path.join(out_dir, f"{client_id}_class_distribution.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("Client data summary\n")
        f.write("===================\n\n")
        f.write(f"Client: {client_id}\n\n")

        # Sub_Label counts
        f.write("Sub_Label counts (raw before drop):\n")
        if sublabel_counts:
            for k, v in sorted(sublabel_counts.items(), key=lambda x: (-x[1], x[0])):
                f.write(f"  {k}: {v}\n")
        else:
            f.write("  (no Sub_Label column present)\n")
        f.write("\n")

        # Train/Test class balance info
        def fmt_counts(tag, cnts):
            tot = sum(cnts.values()) if cnts else 0
            parts = []
            for c in sorted(cnts.keys()):
                parts.append(f"class {c}: {cnts[c]} ({cnts[c]/tot:.3f} of split)")
            return f"{tag}: total={tot} | " + ", ".join(parts) + "\n"

        f.write(fmt_counts("TRAIN BEFORE SMOTE", train_counts_before))
        f.write(fmt_counts("TRAIN AFTER  SMOTE", train_counts_after))
        f.write(fmt_counts("TEST (NO SMOTE)   ", test_counts))

        f.write("\n(Strategy metrics will be appended after training.)\n")


def prepare_all_clients(
    raw_dir: str,
    processed_dir: str,
    label_column: str,
    drop_columns: List[str],
    numeric_features: List[str],
    categorical_features: List[str],
    test_size: float,
    random_state: int,
    scaler_type: str,
    use_smote: bool,
    smote_minority_ratio_threshold: float,
    summaries_dir: str,
) -> Dict[str, Dict[str, Any]]:
    """
    Main preprocessing entry point, now with deterministic client IDs.
    Steps:
    1. Load raw CSVs in the specified order and assign client1..client5.
    2. Drop unused columns.
    3. Map Label -> {0,1}.
    4. Build global vocab for categorical columns.
    5. Per client:
       - train/test split (stratified if possible)
       - one-hot using global vocab
       - scale numeric using train scaler
       - SMOTE on train only
       - log class stats + Sub_Label stats
       - save <client>_train.csv and <client>_test.csv in processed_dir
    6. Return numpy arrays for training and testing.
    """

    _ensure_dir(processed_dir)
    _ensure_dir(summaries_dir)

    # 1. load raw w/ fixed ID mapping
    raw_clients = _load_all_clients_raw(raw_dir)

    # 2. cleanup and binary label mapping
    cleaned_clients: Dict[str, Dict[str, Any]] = {}
    for cid, df_raw in raw_clients.items():
        # count Sub_Label before we drop it
        sl_counts = _count_sublabels(df_raw)

        # drop columns we don't want (but always preserve the label column)
        df = df_raw.copy()
        df = _drop_columns(df, drop_columns, label_column=label_column)

        # map label to binary 0/1
        df[label_column] = _apply_label_mapping(df, label_column)

        cleaned_clients[cid] = {
            "df": df,
            "sublabel_counts": sl_counts,
        }

    # 3. global categorical vocab across ALL cleaned clients
    global_vocab = _build_global_categorical_vocab(
        {cid: v["df"] for cid, v in cleaned_clients.items()},
        categorical_features,
    )

    out_data: Dict[str, Dict[str, Any]] = {}

    # 4. per client processing
    for cid, pack in cleaned_clients.items():
        df = pack["df"]

        # split into features/label
        X_full = df.drop(columns=[label_column]).reset_index(drop=True)
        y_full = df[label_column].reset_index(drop=True)

        # make sure all expected columns exist even if missing in this DF
        # numeric defaults to 0.0, categorical defaults to "NA_CAT"
        for nc in numeric_features:
            if nc not in X_full.columns:
                X_full[nc] = 0.0
        for cc in categorical_features:
            if cc not in X_full.columns:
                X_full[cc] = "NA_CAT"

        # bool-like categorical columns should be stringified
        if "local_orig" in X_full.columns:
            X_full["local_orig"] = X_full["local_orig"].astype(str)
        if "local_resp" in X_full.columns:
            X_full["local_resp"] = X_full["local_resp"].astype(str)

        # stratified train/test split
        X_tr, X_te, y_tr, y_te = _train_test_split(
            X_full, y_full, test_size=test_size, random_state=random_state
        )

        # helper to encode each split using global vocab
        def encode_block(block_df: pd.DataFrame) -> pd.DataFrame:
            """
            Build feature matrix for this split BEFORE imputation/scaling.
            Steps:
            - Ensure all expected numeric & categorical columns exist.
            - Clean numeric columns (coerce to float, NaN for bad values).
            - Clean categorical columns (map blanks/'-' -> 'NA_CAT').
            - Hash categorical columns into fixed-size one-hot buckets.
            Final column order:
              [numeric_features ...] + [hashed categorical buckets ...]
            """
            work = block_df.copy()

            # ensure all required columns exist
            for nc in numeric_features:
                if nc not in work.columns:
                    work[nc] = np.nan
            for cc in categorical_features:
                if cc not in work.columns:
                    work[cc] = "NA_CAT"

            # numeric cleaning (to float with NaN where invalid)
            work = _clean_numeric_columns(work, numeric_features)

            # force bool-likes into string before hashing
            if "local_orig" in work.columns:
                work["local_orig"] = work["local_orig"].astype(str)
            if "local_resp" in work.columns:
                work["local_resp"] = work["local_resp"].astype(str)

            # numeric block first, keep column order stable
            numeric_part = work[numeric_features].copy()  # may still include NaN here

            # hashed categorical one-hot block (fixed width per column)
            cat_part = _hashed_one_hot_block(
                work,
                categorical_features,
            )

            encoded = pd.concat(
                [numeric_part.reset_index(drop=True),
                 cat_part.reset_index(drop=True)],
                axis=1
            )
            return encoded
        
        X_tr_enc = encode_block(X_tr)
        X_te_enc = encode_block(X_te)

        train_medians = {}
        for nc in numeric_features:
            median_val = np.nanmedian(X_tr_enc[nc].to_numpy(dtype=float))
            if np.isnan(median_val):
                median_val = 0.0
            train_medians[nc] = median_val
            X_tr_enc[nc] = X_tr_enc[nc].fillna(median_val)
            X_te_enc[nc] = X_te_enc[nc].fillna(median_val)

        # Scale numeric cols using train fit
        X_tr_enc, scaler_fit = _scale_numeric(
            X_tr_enc, numeric_features, scaler_type, scaler_fit=None
        )
        X_te_enc, _ = _scale_numeric(
            X_te_enc, numeric_features, scaler_type, scaler_fit=scaler_fit
        )
        
        # SMOTE on train only
        X_tr_bal, y_tr_bal, before_cnt, after_cnt = _smote_if_needed(
            X_tr_enc, y_tr, use_smote, smote_minority_ratio_threshold
        )

        test_cnt = Counter(y_te.tolist())

        # log per-client stats (sub-labels, benign/malicious counts, etc.)
        _log_client_stats(
            client_id=cid,
            out_dir=summaries_dir,
            sublabel_counts=pack["sublabel_counts"],
            train_counts_before=before_cnt,
            train_counts_after=after_cnt,
            test_counts=dict(test_cnt),
        )

        # write processed CSVs for reproducibility
        train_out = X_tr_bal.copy()
        train_out[label_column] = y_tr_bal.to_numpy()
        test_out = X_te_enc.copy()
        test_out[label_column] = y_te.to_numpy()

        _ensure_dir(processed_dir)
        train_out.to_csv(os.path.join(processed_dir, f"{cid}_train.csv"), index=False)
        test_out.to_csv(os.path.join(processed_dir, f"{cid}_test.csv"), index=False)

        # stash numpy arrays
        out_data[cid] = {
            "X_train": train_out.drop(columns=[label_column]).to_numpy(dtype=np.float32),
            "y_train": train_out[label_column].to_numpy(dtype=np.int64),
            "X_test": test_out.drop(columns=[label_column]).to_numpy(dtype=np.float32),
            "y_test": test_out[label_column].to_numpy(dtype=np.int64),
        }

    return out_data
