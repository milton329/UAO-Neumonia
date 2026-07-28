from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from uao_neumonia.application.prediction_service import predict
from uao_neumonia.domain.models import PredictionResult


def _fake_model(predictions_row):
    model = MagicMock()
    model.predict.return_value = np.array([predictions_row])
    return model


@patch("uao_neumonia.application.prediction_service.generate_grad_cam")
@patch("uao_neumonia.application.prediction_service.load_model")
@patch("uao_neumonia.application.prediction_service.preprocess")
def test_predict_returns_a_prediction_result(
    mock_preprocess, mock_model_fun, mock_grad_cam
):
    mock_preprocess.return_value = np.zeros((1, 512, 512, 1))
    mock_model_fun.return_value = _fake_model([0.9, 0.05, 0.05])
    mock_grad_cam.return_value = np.zeros((512, 512, 3))

    result = predict(np.zeros((256, 256, 3)))

    assert isinstance(result, PredictionResult)
    assert result.label == "bacteriana"
    assert result.probability == pytest.approx(90.0)
    assert result.heatmap.shape == (512, 512, 3)


@patch("uao_neumonia.application.prediction_service.generate_grad_cam")
@patch("uao_neumonia.application.prediction_service.load_model")
@patch("uao_neumonia.application.prediction_service.preprocess")
def test_predict_labels_class_0_as_bacteriana(
    mock_preprocess, mock_model_fun, mock_grad_cam
):
    mock_preprocess.return_value = np.zeros((1, 512, 512, 1))
    mock_model_fun.return_value = _fake_model([0.9, 0.05, 0.05])
    mock_grad_cam.return_value = np.zeros((512, 512, 3))

    result = predict(np.zeros((256, 256, 3)))

    assert result.label == "bacteriana"


@patch("uao_neumonia.application.prediction_service.generate_grad_cam")
@patch("uao_neumonia.application.prediction_service.load_model")
@patch("uao_neumonia.application.prediction_service.preprocess")
def test_predict_labels_class_1_as_normal(
    mock_preprocess, mock_model_fun, mock_grad_cam
):
    mock_preprocess.return_value = np.zeros((1, 512, 512, 1))
    mock_model_fun.return_value = _fake_model([0.05, 0.9, 0.05])
    mock_grad_cam.return_value = np.zeros((512, 512, 3))

    result = predict(np.zeros((256, 256, 3)))

    assert result.label == "normal"


@patch("uao_neumonia.application.prediction_service.generate_grad_cam")
@patch("uao_neumonia.application.prediction_service.load_model")
@patch("uao_neumonia.application.prediction_service.preprocess")
def test_predict_labels_class_2_as_viral(
    mock_preprocess, mock_model_fun, mock_grad_cam
):
    mock_preprocess.return_value = np.zeros((1, 512, 512, 1))
    mock_model_fun.return_value = _fake_model([0.05, 0.05, 0.9])
    mock_grad_cam.return_value = np.zeros((512, 512, 3))

    result = predict(np.zeros((256, 256, 3)))

    assert result.label == "viral"


@patch("uao_neumonia.application.prediction_service.generate_grad_cam")
@patch("uao_neumonia.application.prediction_service.load_model")
@patch("uao_neumonia.application.prediction_service.preprocess")
def test_predict_probability_is_expressed_as_percentage(
    mock_preprocess, mock_model_fun, mock_grad_cam
):
    mock_preprocess.return_value = np.zeros((1, 512, 512, 1))
    mock_model_fun.return_value = _fake_model([0.8, 0.1, 0.1])
    mock_grad_cam.return_value = np.zeros((512, 512, 3))

    result = predict(np.zeros((256, 256, 3)))

    assert result.probability == pytest.approx(80.0)


@patch("uao_neumonia.application.prediction_service.generate_grad_cam")
@patch("uao_neumonia.application.prediction_service.load_model")
@patch("uao_neumonia.application.prediction_service.preprocess")
def test_predict_passes_input_array_to_preprocess(
    mock_preprocess, mock_model_fun, mock_grad_cam
):
    mock_preprocess.return_value = np.zeros((1, 512, 512, 1))
    mock_model_fun.return_value = _fake_model([0.9, 0.05, 0.05])
    mock_grad_cam.return_value = np.zeros((512, 512, 3))
    original_array = np.ones((256, 256, 3))

    predict(original_array)

    called_array = mock_preprocess.call_args[0][0]
    assert np.array_equal(called_array, original_array)


@patch("uao_neumonia.application.prediction_service.generate_grad_cam")
@patch("uao_neumonia.application.prediction_service.load_model")
@patch("uao_neumonia.application.prediction_service.preprocess")
def test_predict_passes_original_array_to_grad_cam_not_the_preprocessed_batch(
    mock_preprocess, mock_model_fun, mock_grad_cam
):
    mock_preprocess.return_value = np.zeros((1, 512, 512, 1))
    mock_model_fun.return_value = _fake_model([0.9, 0.05, 0.05])
    mock_grad_cam.return_value = np.zeros((512, 512, 3))
    original_array = np.ones((256, 256, 3))

    predict(original_array)

    called_array = mock_grad_cam.call_args[0][0]
    assert np.array_equal(called_array, original_array)


@patch("uao_neumonia.application.prediction_service.generate_grad_cam")
@patch("uao_neumonia.application.prediction_service.load_model")
@patch("uao_neumonia.application.prediction_service.preprocess")
def test_predict_returns_empty_label_for_unmapped_class(
    mock_preprocess, mock_model_fun, mock_grad_cam
):
    mock_preprocess.return_value = np.zeros((1, 512, 512, 1))
    mock_model_fun.return_value = _fake_model([0.1, 0.1, 0.1, 0.7])
    mock_grad_cam.return_value = np.zeros((512, 512, 3))

    result = predict(np.zeros((256, 256, 3)))

    assert result.label == ""


@patch("uao_neumonia.application.prediction_service.generate_grad_cam")
@patch("uao_neumonia.application.prediction_service.load_model")
@patch("uao_neumonia.application.prediction_service.preprocess")
def test_predict_with_100_percent_confidence(
    mock_preprocess, mock_model_fun, mock_grad_cam
):
    mock_preprocess.return_value = np.zeros((1, 512, 512, 1))
    mock_model_fun.return_value = _fake_model([1.0, 0.0, 0.0])
    mock_grad_cam.return_value = np.zeros((512, 512, 3))

    result = predict(np.zeros((256, 256, 3)))

    assert result.probability == pytest.approx(100.0)


@patch("uao_neumonia.application.prediction_service.generate_grad_cam")
@patch("uao_neumonia.application.prediction_service.load_model")
@patch("uao_neumonia.application.prediction_service.preprocess")
def test_predict_model_called_once(
    mock_preprocess, mock_model_fun, mock_grad_cam
):
    mock_preprocess.return_value = np.zeros((1, 512, 512, 1))
    mock_model = _fake_model([0.9, 0.05, 0.05])
    mock_model_fun.return_value = mock_model
    mock_grad_cam.return_value = np.zeros((512, 512, 3))

    predict(np.zeros((256, 256, 3)))

    mock_model.predict.assert_called_once()


@patch("uao_neumonia.application.prediction_service.generate_grad_cam")
@patch("uao_neumonia.application.prediction_service.load_model")
@patch("uao_neumonia.application.prediction_service.preprocess")
def test_predict_preprocess_called_once(
    mock_preprocess, mock_model_fun, mock_grad_cam
):
    mock_preprocess.return_value = np.zeros((1, 512, 512, 1))
    mock_model_fun.return_value = _fake_model([0.9, 0.05, 0.05])
    mock_grad_cam.return_value = np.zeros((512, 512, 3))

    predict(np.zeros((256, 256, 3)))

    mock_preprocess.assert_called_once()


@patch("uao_neumonia.application.prediction_service.generate_grad_cam")
@patch("uao_neumonia.application.prediction_service.load_model")
@patch("uao_neumonia.application.prediction_service.preprocess")
def test_predict_grad_cam_called_once(
    mock_preprocess, mock_model_fun, mock_grad_cam
):
    mock_preprocess.return_value = np.zeros((1, 512, 512, 1))
    mock_model_fun.return_value = _fake_model([0.9, 0.05, 0.05])
    mock_grad_cam.return_value = np.zeros((512, 512, 3))

    predict(np.zeros((256, 256, 3)))

    mock_grad_cam.assert_called_once()
