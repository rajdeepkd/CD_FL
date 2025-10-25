from typing import List, Tuple, Dict, Any
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, LabelEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter


def split_features_label(
    df: pd.DataFrame, label_col: str
) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[label_col])
    y = df[label_col]
    return X, y


def detect_feature_types(
    df: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
    label_col: str,
    onehot_max_cardinality: int,
) -> Tuple[List[str], List[str]]:
    cols = [c for c in df.columns if c != label_col]

    if numeric_features:
        num_cols = [c for c in numeric_features if c in cols]
    else:
        num_cols = [
            c for c in cols if pd.api.types.is_numeric_dtype(df[c])
        ]

    if categorical_features:
        cat_cols = [c for c in categorical_features if c in cols]
    else:
        cat_cols = [
            c
            for c in cols
            if c not in num_cols
        ]

    return num_cols, cat_cols


def encode_features(
    X: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
    impute_strategy: str,
    scaler: str,
    onehot_max_cardinality: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    X = X.copy()

    num_imputer = SimpleImputer(strategy=impute_strategy)
    if num_cols:
        X[num_cols] = num_imputer.fit_transform(X[num_cols])

    if scaler == "robust":
        scaler_obj = RobustScaler()
    else:
        scaler_obj = StandardScaler()

    if num_cols:
        X[num_cols] = scaler_obj.fit_transform(X[num_cols])

    encoders: Dict[str, Any] = {}
    cat_frames = []
    for c in cat_cols:
        col_vals = X[c].astype(str).fillna("NA_CAT")
        unique_vals = col_vals.unique()
        if len(unique_vals) <= onehot_max_cardinality:
            ohe = OneHotEncoder(
                handle_unknown="ignore", sparse_output=False, dtype=float
            )
            arr = ohe.fit_transform(col_vals.to_numpy().reshape(-1, 1))
            new_cols = [f"{c}__{cat}" for cat in ohe.categories_[0]]
            cat_frames.append(pd.DataFrame(arr, columns=new_cols, index=X.index))
            encoders[c] = ("onehot", ohe, new_cols)
        else:
            le = LabelEncoder()
            X[c] = le.fit_transform(col_vals)
            encoders[c] = ("label", le, [c])

    if cat_frames:
        cat_df = pd.concat(cat_frames, axis=1)
    else:
        cat_df = pd.DataFrame(index=X.index)

    drop_cols = []
    for c, (mode, _, newcols) in encoders.items():
        if mode == "onehot":
            drop_cols.append(c)
    X = X.drop(columns=drop_cols, errors="ignore")
    X = pd.concat([X, cat_df], axis=1)

    artifacts = {
        "num_imputer": num_imputer,
        "scaler": scaler_obj,
        "encoders": encoders,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
    }

    return X, artifacts


def maybe_apply_smote(
    X: pd.DataFrame,
    y: pd.Series,
    use_smote: bool,
    smote_minority_ratio_threshold: float,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, int], Dict[str, int]]:
    before_counts = Counter(y.tolist())

    if len(before_counts) > 1:
        total = sum(before_counts.values())
        minority = min(before_counts.values())
        minority_ratio = minority / total
    else:
        minority_ratio = 1.0

    if use_smote and minority_ratio < smote_minority_ratio_threshold and len(before_counts) > 1:
        sm = SMOTE()
        X_res, y_res = sm.fit_resample(X, y)
        after_counts = Counter(y_res.tolist())
        return X_res, y_res, before_counts, after_counts
    else:
        return X, y, before_counts, before_counts


def preprocess_client_dataframe(
    df: pd.DataFrame,
    label_col: str,
    numeric_features: List[str],
    categorical_features: List[str],
    impute_strategy: str,
    scaler: str,
    onehot_max_cardinality: int,
    use_smote: bool,
    smote_minority_ratio_threshold: float,
) -> Tuple[pd.DataFrame, str]:
    X, y = split_features_label(df, label_col)

    num_cols, cat_cols = detect_feature_types(
        df=df,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        label_col=label_col,
        onehot_max_cardinality=onehot_max_cardinality,
    )

    X_enc, _artifacts = encode_features(
        X=X,
        num_cols=num_cols,
        cat_cols=cat_cols,
        impute_strategy=impute_strategy,
        scaler=scaler,
        onehot_max_cardinality=onehot_max_cardinality,
    )

    y_enc = y.copy()
    if not pd.api.types.is_numeric_dtype(y_enc):
        unique_vals = sorted(y_enc.unique().tolist())
        mapping = {v: i for i, v in enumerate(unique_vals)}
        y_enc = y_enc.map(mapping).astype(int)
    else:
        y_enc = y_enc.astype(int)

    X_bal, y_bal, before_counts, after_counts = maybe_apply_smote(
        X_enc, y_enc, use_smote, smote_minority_ratio_threshold
    )

    processed_df = X_bal.copy()
    processed_df[label_col] = y_bal

    def counts_to_str(tag: str, cnts) -> str:
        total = sum(cnts.values())
        parts = []
        for k in sorted(cnts.keys()):
            parts.append(f"class {k}: {cnts[k]} ({cnts[k]/total:.3f})")
        return f"{tag}: total={total} | " + ", ".join(parts)

    report_lines = [
        "Class Distribution Report",
        counts_to_str("BEFORE", before_counts),
        counts_to_str("AFTER ", after_counts),
    ]
    report = "\n".join(report_lines)

    return processed_df, report
