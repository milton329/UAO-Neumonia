from unittest.mock import MagicMock, patch

import numpy as np
import PIL.Image
import pytest

from uao_neumonia.infrastructure.image_reader import read_dicom, read_standard_image


def _fake_bgr_image(value=100):
    return np.full((256, 256, 3), value, dtype=np.uint8)


def _fake_dicom_dataset(pixel_value=150, bit_depth=np.uint16):
    dataset = MagicMock()
    dataset.pixel_array = np.full((256, 256), pixel_value, dtype=bit_depth)
    return dataset


@patch("uao_neumonia.infrastructure.image_reader.cv2.imread")
def test_read_standard_image_returns_tuple(mock_imread):
    mock_imread.return_value = _fake_bgr_image()
    result = read_standard_image("fake.jpg")
    assert isinstance(result, tuple)
    assert len(result) == 2


@patch("uao_neumonia.infrastructure.image_reader.cv2.imread")
def test_read_standard_image_array_dtype_is_uint8(mock_imread):
    mock_imread.return_value = _fake_bgr_image()
    array, _ = read_standard_image("fake.jpg")
    assert array.dtype == np.uint8


@patch("uao_neumonia.infrastructure.image_reader.cv2.imread")
def test_read_standard_image_img2show_is_pil_image(mock_imread):
    mock_imread.return_value = _fake_bgr_image()
    _, img2show = read_standard_image("fake.jpg")
    assert isinstance(img2show, PIL.Image.Image)


@patch("uao_neumonia.infrastructure.image_reader.cv2.imread")
def test_read_standard_image_normalizes_max_value_to_255(mock_imread):
    mock_imread.return_value = _fake_bgr_image(value=100)
    array, _ = read_standard_image("fake.jpg")
    assert array.max() == 255


@patch("uao_neumonia.infrastructure.image_reader.cv2.imread")
def test_read_standard_image_preserves_shape(mock_imread):
    mock_imread.return_value = _fake_bgr_image()
    array, _ = read_standard_image("fake.jpg")
    assert array.shape == (256, 256, 3)


@patch("uao_neumonia.infrastructure.image_reader.cv2.imread")
def test_read_standard_image_calls_imread_with_given_path(mock_imread):
    mock_imread.return_value = _fake_bgr_image()
    read_standard_image("radiografia.jpg")
    mock_imread.assert_called_once_with("radiografia.jpg")


@patch("uao_neumonia.infrastructure.image_reader.cv2.imread")
def test_read_standard_image_raises_when_file_could_not_be_read(mock_imread):
    mock_imread.return_value = None
    with pytest.raises((AttributeError, TypeError)):
        read_standard_image("no_existe.jpg")


@patch("uao_neumonia.infrastructure.image_reader.cv2.imread")
def test_read_standard_image_with_black_image(mock_imread):
    mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    array, _ = read_standard_image("black.jpg")
    assert array.max() == 0
    assert array.min() == 0


@patch("uao_neumonia.infrastructure.image_reader.cv2.imread")
def test_read_standard_image_with_white_image(mock_imread):
    mock_imread.return_value = np.full((50, 50, 3), 255, dtype=np.uint8)
    array, _ = read_standard_image("white.jpg")
    assert array.max() == 255


@patch("uao_neumonia.infrastructure.image_reader.cv2.imread")
def test_read_standard_image_handles_png_extension(mock_imread):
    mock_imread.return_value = _fake_bgr_image()
    read_standard_image("image.png")
    mock_imread.assert_called_once_with("image.png")


@patch("uao_neumonia.infrastructure.image_reader.dicom.dcmread")
def test_read_dicom_returns_tuple(mock_dcmread):
    mock_dcmread.return_value = _fake_dicom_dataset()
    result = read_dicom("fake.dcm")
    assert isinstance(result, tuple)
    assert len(result) == 2


@patch("uao_neumonia.infrastructure.image_reader.dicom.dcmread")
def test_read_dicom_converts_to_rgb_three_channels(mock_dcmread):
    mock_dcmread.return_value = _fake_dicom_dataset()
    img_rgb, _ = read_dicom("fake.dcm")
    assert img_rgb.shape == (256, 256, 3)


@patch("uao_neumonia.infrastructure.image_reader.dicom.dcmread")
def test_read_dicom_img2show_is_pil_image(mock_dcmread):
    mock_dcmread.return_value = _fake_dicom_dataset()
    _, img2show = read_dicom("fake.dcm")
    assert isinstance(img2show, PIL.Image.Image)


@patch("uao_neumonia.infrastructure.image_reader.dicom.dcmread")
def test_read_dicom_normalizes_pixel_values_to_255(mock_dcmread):
    mock_dcmread.return_value = _fake_dicom_dataset()
    img_rgb, _ = read_dicom("fake.dcm")
    assert img_rgb.max() == 255


@patch("uao_neumonia.infrastructure.image_reader.dicom.dcmread")
def test_read_dicom_calls_dcmread_with_given_path(mock_dcmread):
    mock_dcmread.return_value = _fake_dicom_dataset()
    read_dicom("paciente123.dcm")
    mock_dcmread.assert_called_once_with("paciente123.dcm")


@patch("uao_neumonia.infrastructure.image_reader.dicom.dcmread")
def test_read_dicom_with_8bit_pixel_array(mock_dcmread):
    mock_dcmread.return_value = _fake_dicom_dataset(pixel_value=200, bit_depth=np.uint8)
    img_rgb, _ = read_dicom("8bit.dcm")
    assert img_rgb.shape == (256, 256, 3)


@patch("uao_neumonia.infrastructure.image_reader.dicom.dcmread")
def test_read_dicom_with_all_zero_pixels(mock_dcmread):
    mock_dcmread.return_value = _fake_dicom_dataset(pixel_value=0)
    img_rgb, _ = read_dicom("zero.dcm")
    assert img_rgb.max() == 0
