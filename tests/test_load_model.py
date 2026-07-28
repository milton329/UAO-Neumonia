#!/usr/bin/env python
"""Pruebas unitarias de load_model.model_fun."""

from unittest.mock import patch

from app import load_model
from app.load_model import model_fun


@patch("app.load_model.tf.keras.models.load_model")
def test_model_fun_calls_load_model_with_model_path(mock_load_model):
    model_fun()
    args, _ = mock_load_model.call_args
    assert args[0] == load_model.MODEL_PATH


@patch("app.load_model.tf.keras.models.load_model")
def test_model_fun_calls_load_model_with_compile_false(mock_load_model):
    model_fun()
    _, kwargs = mock_load_model.call_args
    assert kwargs.get("compile") is False


@patch("app.load_model.tf.keras.models.load_model")
def test_model_fun_returns_whatever_load_model_returns(mock_load_model):
    sentinel_model = object()
    mock_load_model.return_value = sentinel_model
    assert model_fun() is sentinel_model


def test_model_path_points_to_conv_mlp_84():
    assert load_model.MODEL_PATH.name == "conv_MLP_84.h5"
    assert load_model.MODEL_PATH.parent.name == "models"
