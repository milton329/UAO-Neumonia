#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pruebas unitarias de grad_cam.grad_cam.

Usan un modelo Keras diminuto (fixture tiny_cnn_model) en vez del modelo
real de 113MB, para que las pruebas corran rápido y no dependan de un
archivo externo.
"""

from unittest.mock import patch

import cv2
import numpy as np

from grad_cam import grad_cam


def test_returns_expected_output_shape(tiny_cnn_model, sample_bgr_image):
    with patch("grad_cam.model_fun", return_value=tiny_cnn_model):
        result = grad_cam(sample_bgr_image)
    assert result.shape == (512, 512, 3)


def test_returns_uint8_dtype(tiny_cnn_model, sample_bgr_image):
    with patch("grad_cam.model_fun", return_value=tiny_cnn_model):
        result = grad_cam(sample_bgr_image)
    assert result.dtype == np.uint8


def test_values_within_valid_pixel_range(tiny_cnn_model, sample_bgr_image):
    with patch("grad_cam.model_fun", return_value=tiny_cnn_model):
        result = grad_cam(sample_bgr_image)
    assert result.min() >= 0
    assert result.max() <= 255


def test_calls_model_fun_once(tiny_cnn_model, sample_bgr_image):
    with patch("grad_cam.model_fun", return_value=tiny_cnn_model) as mock_model_fun:
        grad_cam(sample_bgr_image)
    mock_model_fun.assert_called_once()


def test_works_with_smaller_input_image(tiny_cnn_model):
    small_image = np.full((128, 128, 3), 60, dtype=np.uint8)
    with patch("grad_cam.model_fun", return_value=tiny_cnn_model):
        result = grad_cam(small_image)
    assert result.shape == (512, 512, 3)


def test_looks_up_the_named_conv_layer(tiny_cnn_model, sample_bgr_image):
    original_get_layer = tiny_cnn_model.get_layer
    calls = []

    def spy_get_layer(name):
        calls.append(name)
        return original_get_layer(name)

    tiny_cnn_model.get_layer = spy_get_layer

    with patch("grad_cam.model_fun", return_value=tiny_cnn_model):
        grad_cam(sample_bgr_image)

    assert "conv10_thisone" in calls


def test_is_deterministic_for_the_same_input_and_model(tiny_cnn_model, sample_bgr_image):
    with patch("grad_cam.model_fun", return_value=tiny_cnn_model):
        first = grad_cam(sample_bgr_image)
        second = grad_cam(sample_bgr_image)
    assert np.array_equal(first, second)


def test_heatmap_actually_blends_with_original_image(tiny_cnn_model, sample_bgr_image):
    plain_resized_rgb = cv2.resize(sample_bgr_image, (512, 512))[:, :, ::-1]
    with patch("grad_cam.model_fun", return_value=tiny_cnn_model):
        result = grad_cam(sample_bgr_image)
    assert not np.array_equal(result, plain_resized_rgb)
