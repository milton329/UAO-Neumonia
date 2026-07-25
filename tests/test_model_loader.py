from unittest.mock import patch

from uao_neumonia.infrastructure import model_loader as ml


@patch("uao_neumonia.infrastructure.model_loader.tf.keras.models.load_model")
def test_load_model_calls_load_model_with_model_path(mock_load_model):
    ml.load_model()
    args, _ = mock_load_model.call_args
    assert args[0] == ml.MODEL_PATH


@patch("uao_neumonia.infrastructure.model_loader.tf.keras.models.load_model")
def test_load_model_calls_load_model_with_compile_false(mock_load_model):
    ml.load_model()
    _, kwargs = mock_load_model.call_args
    assert kwargs.get("compile") is False


@patch("uao_neumonia.infrastructure.model_loader.tf.keras.models.load_model")
def test_load_model_returns_whatever_load_model_returns(mock_load_model):
    sentinel_model = object()
    mock_load_model.return_value = sentinel_model
    assert ml.load_model() is sentinel_model


def test_model_path_points_to_conv_mlp_84():
    assert ml.MODEL_PATH.name == "conv_MLP_84.h5"
    assert ml.MODEL_PATH.parent.name == "models"
