import cv2
import numpy as np
import pytest

from uao_neumonia.infrastructure.image_processor import preprocess


def _bgr_image(height, width, value=128):
    return np.full((height, width, 3), value, dtype=np.uint8)


def test_output_shape_is_batch_512_512_1():
    result = preprocess(_bgr_image(300, 400))
    assert result.shape == (1, 512, 512, 1)


def test_output_dtype_is_floating_point():
    result = preprocess(_bgr_image(512, 512))
    assert np.issubdtype(result.dtype, np.floating)


def test_values_are_normalized_between_0_and_1():
    result = preprocess(_bgr_image(512, 512, value=200))
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_batch_dimension_is_one():
    result = preprocess(_bgr_image(128, 128))
    assert result.shape[0] == 1


def test_channel_dimension_is_one_grayscale():
    result = preprocess(_bgr_image(256, 256))
    assert result.shape[-1] == 1


def test_resize_from_smaller_image():
    result = preprocess(_bgr_image(64, 64))
    assert result.shape[1:3] == (512, 512)


def test_resize_from_larger_image():
    result = preprocess(_bgr_image(1024, 800))
    assert result.shape[1:3] == (512, 512)


def test_black_image_does_not_crash_and_stays_near_zero():
    result = preprocess(_bgr_image(512, 512, value=0))
    assert result.shape == (1, 512, 512, 1)
    assert result.max() < 0.05


def test_white_image_does_not_crash():
    result = preprocess(_bgr_image(512, 512, value=255))
    assert result.shape == (1, 512, 512, 1)


def test_distinct_regions_are_actually_converted_to_grayscale():
    array = np.zeros((512, 512, 3), dtype=np.uint8)
    array[:, :256] = (10, 200, 90)
    array[:, 256:] = (250, 30, 60)

    result = preprocess(array)

    assert result.std() > 0


def test_non_square_input_produces_square_output():
    result = preprocess(_bgr_image(400, 600))
    assert result.shape[1] == 512
    assert result.shape[2] == 512


def test_input_with_float_dtype():
    array = np.full((256, 256, 3), 0.5, dtype=np.float32)
    result = preprocess((array * 255).astype(np.uint8))
    assert result.shape == (1, 512, 512, 1)
    assert np.issubdtype(result.dtype, np.floating)


def test_large_input_image():
    result = preprocess(_bgr_image(2000, 1500, value=120))
    assert result.shape == (1, 512, 512, 1)


def test_very_small_input():
    result = preprocess(_bgr_image(16, 16, value=100))
    assert result.shape == (1, 512, 512, 1)


def test_rgba_input_uses_only_first_three_channels():
    rgba = np.full((100, 100, 4), 128, dtype=np.uint8)
    result = preprocess(rgba)
    assert result.shape == (1, 512, 512, 1)


def test_grayscale_input_raises_error():
    gray = np.full((100, 100), 128, dtype=np.uint8)
    with pytest.raises(cv2.error):
        preprocess(gray)
