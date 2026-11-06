import numpy as np
import pytest

from src.student.predict import PREDICTION_COLUMNS, check_prediction_frame, prediction_frame


def frame(n=5):
    logits = np.linspace(-30, 30, n * 6).reshape(n, 6)
    return prediction_frame([f"id{i}" for i in range(n)], logits, "student-test")


def test_prediction_file_has_contract_columns_and_types():
    pred = frame()
    assert list(pred.columns) == PREDICTION_COLUMNS
    assert all(pred[c].dtype == "float32" for c in PREDICTION_COLUMNS[1:-1])
    assert str(pred["id"].dtype) == "string"
    check_prediction_frame(pred)


def test_extreme_logits_stay_in_range():
    pred = frame()
    p = pred[PREDICTION_COLUMNS[1:-1]].to_numpy()
    assert np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1


@pytest.mark.parametrize("corrupt, message", [
    (lambda f: f.assign(extra=1), "columns"),
    (lambda f: f.assign(id="same"), "duplicate"),
    (lambda f: f.assign(p_threat=np.float32("nan")), "outside"),
    (lambda f: f.assign(model_version=["a", "b", "a", "b", "a"]), "model_version"),
])
def test_bad_prediction_files_are_rejected(corrupt, message):
    with pytest.raises(ValueError, match=message):
        check_prediction_frame(corrupt(frame()))


def test_length_stats_counts_rows_over_each_candidate():
    from src.student.token_lengths import length_stats

    stats = length_stats([10, 100, 200, 300], candidates=(128, 256))
    assert stats["truncated_share"] == {"128": 0.5, "256": 0.25}
    assert stats["rows"] == 4 and stats["max"] == 300
