#!/usr/bin/env python
"""Pruebas unitarias de integrator.predict.

model_fun, preprocess y grad_cam se mockean para aislar la lógica de
orquestación de integrator.predict de TensorFlow/OpenCV reales.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.integrator import predict


def _fake_model(predictions_row):
    model = MagicMock()
    model.predict.return_value = np.array([predictions_row])
    return model


@patch("app.integrator.grad_cam")
@patch("app.integrator.model_fun")
@patch("app.integrator.preprocess")
def test_predict_returns_a_three_item_tuple(
    mock_preprocess, mock_model_fun, mock_grad_cam
):
    mock_preprocess.return_value = np.zeros((1, 512, 512, 1))
    mock_model_fun.return_value = _fake_model([0.9, 0.05, 0.05])
    mock_grad_cam.return_value = np.zeros((512, 512, 3))

    result = predict(np.zeros((256, 256, 3)))

    assert isinstance(result, tuple)
    assert len(result) == 3


@patch("app.integrator.grad_cam")
@patch("app.integrator.model_fun")
@patch("app.integrator.preprocess")
def test_predict_labels_class_0_as_bacteriana(
    mock_preprocess, mock_model_fun, mock_grad_cam
):
    mock_preprocess.return_value = np.zeros((1, 512, 512, 1))
    mock_model_fun.return_value = _fake_model([0.9, 0.05, 0.05])
    mock_grad_cam.return_value = np.zeros((512, 512, 3))

    label, _, _ = predict(np.zeros((256, 256, 3)))

    assert label == "bacteriana"


@patch("app.integrator.grad_cam")
@patch("app.integrator.model_fun")
@patch("app.integrator.preprocess")
def test_predict_labels_class_1_as_normal(
    mock_preprocess, mock_model_fun, mock_grad_cam
):
    mock_preprocess.return_value = np.zeros((1, 512, 512, 1))
    mock_model_fun.return_value = _fake_model([0.05, 0.9, 0.05])
    mock_grad_cam.return_value = np.zeros((512, 512, 3))

    label, _, _ = predict(np.zeros((256, 256, 3)))

    assert label == "normal"


@patch("app.integrator.grad_cam")
@patch("app.integrator.model_fun")
@patch("app.integrator.preprocess")
def test_predict_labels_class_2_as_viral(
    mock_preprocess, mock_model_fun, mock_grad_cam
):
    mock_preprocess.return_value = np.zeros((1, 512, 512, 1))
    mock_model_fun.return_value = _fake_model([0.05, 0.05, 0.9])
    mock_grad_cam.return_value = np.zeros((512, 512, 3))

    label, _, _ = predict(np.zeros((256, 256, 3)))

    assert label == "viral"


@patch("app.integrator.grad_cam")
@patch("app.integrator.model_fun")
@patch("app.integrator.preprocess")
def test_predict_probability_is_expressed_as_percentage(
    mock_preprocess, mock_model_fun, mock_grad_cam
):
    mock_preprocess.return_value = np.zeros((1, 512, 512, 1))
    mock_model_fun.return_value = _fake_model([0.8, 0.1, 0.1])
    mock_grad_cam.return_value = np.zeros((512, 512, 3))

    _, proba, _ = predict(np.zeros((256, 256, 3)))

    assert proba == pytest.approx(80.0)


@patch("app.integrator.grad_cam")
@patch("app.integrator.model_fun")
@patch("app.integrator.preprocess")
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


@patch("app.integrator.grad_cam")
@patch("app.integrator.model_fun")
@patch("app.integrator.preprocess")
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


@patch("app.integrator.grad_cam")
@patch("app.integrator.model_fun")
@patch("app.integrator.preprocess")
def test_predict_returns_empty_label_for_unmapped_class(
    mock_preprocess, mock_model_fun, mock_grad_cam
):
    mock_preprocess.return_value = np.zeros((1, 512, 512, 1))
    # 4 clases -> argmax puede caer en el índice 3, que no está en LABELS
    mock_model_fun.return_value = _fake_model([0.1, 0.1, 0.1, 0.7])
    mock_grad_cam.return_value = np.zeros((512, 512, 3))

    label, _, _ = predict(np.zeros((256, 256, 3)))

    assert label == ""
