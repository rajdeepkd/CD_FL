import pandas as pd
from src.data.preprocess import preprocess_client_dataframe

def test_preprocess_client_dataframe_basic():
    df = pd.DataFrame({
        "feat_num1": [1.0, 2.0, 3.0, 4.0],
        "feat_cat": ["a", "a", "b", "b"],
        "label": [0, 0, 1, 1],
    })

    processed_df, report = preprocess_client_dataframe(
        df=df,
        label_col="label",
        numeric_features=[],
        categorical_features=[],
        impute_strategy="median",
        scaler="standard",
        onehot_max_cardinality=4,
        use_smote=False,
        smote_minority_ratio_threshold=0.3,
    )

    assert "label" in processed_df.columns

    cat_cols = [c for c in processed_df.columns if c.startswith("feat_cat__")]
    assert len(cat_cols) > 0 or "feat_cat" in processed_df.columns

    assert "Class Distribution Report" in report
