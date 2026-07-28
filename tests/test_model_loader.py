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


@patch("uao_neumonia.infrastructure.model_loader.tf.keras.models.load_model")
def test_load_model_returns_model_instance(mock_load_model):
    mock_load_model.return_value = "fake_model"
    result = ml.load_model()
    assert result == "fake_model"


@patch("uao_neumonia.infrastructure.model_loader.tf.keras.models.load_model")
def test_load_model_called_only_once_due_to_cache(mock_load_model):
    ml.load_model()
    ml.load_model()
    mock_load_model.assert_called_once()


@patch("uao_neumonia.infrastructure.model_loader.tf.keras.models.load_model")
def test_cache_returns_same_instance(mock_load_model):
    model = object()
    mock_load_model.return_value = model
    first = ml.load_model()
    second = ml.load_model()
    assert first is second


def test_model_path_is_absolute():
    assert ml.MODEL_PATH.is_absolute()
