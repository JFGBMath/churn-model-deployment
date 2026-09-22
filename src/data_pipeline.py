import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_data(path: str) -> pd.DataFrame:
    """Loads the raw Telco Customer Churn file (CSV or Excel)."""
    if path.endswith(".xlsx") or path.endswith(".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies cleaning steps for the IBM extended Telco dataset:
    - drops identifier/geography columns not useful as predictive features
    - drops leakage columns (Churn Label, Churn Score, CLTV, Churn Reason)
      that either duplicate the target or leak post-outcome information
    - converts Total Charges from string to numeric, handling blanks
    - keeps Churn Value as the target (already encoded 0/1)
    """
    drop_cols = [
        "CustomerID", "Count", "Country", "State", "City", "Zip Code",
        "Lat Long", "Latitude", "Longitude",
        "Churn Label", "Churn Score", "CLTV", "Churn Reason",
    ]
    df = df.drop(columns=drop_cols)

    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
    df["Total Charges"] = df["Total Charges"].fillna(0)

    df = df.rename(columns={"Churn Value": "Churn"})

    return df


def get_feature_columns(df: pd.DataFrame, target_col: str = "Churn"):
    """
    Splits columns into numeric and categorical feature lists,
    based on dtype, excluding the target column.
    """
    feature_df = df.drop(columns=[target_col])
    numeric_cols = feature_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = feature_df.select_dtypes(include=["object"]).columns.tolist()
    return numeric_cols, categorical_cols


def build_preprocessing_pipeline(numeric_cols, categorical_cols) -> ColumnTransformer:
    """
    Builds a reusable preprocessing pipeline: scales numeric features,
    one-hot encodes categoricals. This gets saved together with the
    trained model, so the API can accept raw feature values directly
    without needing to replicate preprocessing logic separately.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )
    return preprocessor