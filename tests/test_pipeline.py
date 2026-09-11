import pandas as pd

from src.data import split_features_target
from src.preprocessing import build_preprocessor
from src.evaluate import expected_calibration_error


def test_identifier_and_confidential_columns_are_removed():
    frame = pd.DataFrame(
        {"PatientID": [1, 2], "Age": [70, 80], "Diagnosis": [0, 1], "DoctorInCharge": ["X", "X"]}
    )
    X, y = split_features_target(frame)
    assert list(X.columns) == ["Age"]
    assert y.tolist() == [0, 1]


def test_preprocessor_accepts_known_feature_types():
    X = pd.DataFrame({"Age": [70, None], "Gender": [0, 1], "Ethnicity": [0, 2]})
    transformed = build_preprocessor(list(X.columns)).fit_transform(X)
    assert transformed.shape[0] == 2
    assert transformed.shape[1] >= 3


def test_expected_calibration_error_is_zero_for_perfect_binary_predictions():
    y = pd.Series([0, 0, 1, 1]).to_numpy()
    probabilities = pd.Series([0.0, 0.0, 1.0, 1.0]).to_numpy()
    assert expected_calibration_error(y, probabilities) == 0.0
