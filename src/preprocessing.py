from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL_COLUMNS = ["Gender", "Ethnicity", "EducationLevel"]


def build_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    categorical = [c for c in CATEGORICAL_COLUMNS if c in feature_columns]
    numeric = [c for c in feature_columns if c not in categorical]

    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric_pipeline, numeric), ("categorical", categorical_pipeline, categorical)],
        verbose_feature_names_out=False,
    )

