import numpy as np

from uao_neumonia.domain.models import PredictionResult


def test_create_prediction_result_with_valid_data():
    heatmap = np.zeros((512, 512, 3), dtype=np.uint8)
    result = PredictionResult(label="bacteriana", probability=85.5, heatmap=heatmap)
    assert result.label == "bacteriana"
    assert result.probability == 85.5
    assert np.array_equal(result.heatmap, heatmap)


def test_label_is_string():
    heatmap = np.zeros((512, 512, 3), dtype=np.uint8)
    result = PredictionResult(label="normal", probability=90.0, heatmap=heatmap)
    assert isinstance(result.label, str)


def test_probability_is_float():
    heatmap = np.zeros((512, 512, 3), dtype=np.uint8)
    result = PredictionResult(label="viral", probability=75.3, heatmap=heatmap)
    assert isinstance(result.probability, float)


def test_heatmap_is_numpy_array():
    heatmap = np.ones((512, 512, 3), dtype=np.uint8) * 128
    result = PredictionResult(label="bacteriana", probability=50.0, heatmap=heatmap)
    assert isinstance(result.heatmap, np.ndarray)
    assert result.heatmap.shape == (512, 512, 3)


def test_prediction_result_equality():
    heatmap = np.zeros((512, 512, 3), dtype=np.uint8)
    r1 = PredictionResult(label="normal", probability=95.0, heatmap=heatmap)
    r2 = PredictionResult(label="normal", probability=95.0, heatmap=heatmap)
    assert r1 == r2


def test_prediction_result_different_labels_not_equal():
    h1 = np.zeros((512, 512, 3), dtype=np.uint8)
    h2 = np.zeros((512, 512, 3), dtype=np.uint8)
    r1 = PredictionResult(label="bacteriana", probability=80.0, heatmap=h1)
    r2 = PredictionResult(label="viral", probability=80.0, heatmap=h2)
    assert r1 != r2


def test_prediction_result_accepts_zero_probability():
    heatmap = np.zeros((512, 512, 3), dtype=np.uint8)
    result = PredictionResult(label="normal", probability=0.0, heatmap=heatmap)
    assert result.probability == 0.0


def test_prediction_result_accepts_one_hundred_percent():
    heatmap = np.zeros((512, 512, 3), dtype=np.uint8)
    result = PredictionResult(label="bacteriana", probability=100.0, heatmap=heatmap)
    assert result.probability == 100.0
