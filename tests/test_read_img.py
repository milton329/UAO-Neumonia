#!/usr/bin/env python
"""Pruebas unitarias de read_img.read_dicom_file y read_img.read_jpg_file."""

from unittest.mock import MagicMock, patch

import numpy as np
import PIL.Image
import pytest

from app.read_img import read_dicom_file, read_jpg_file

# ---------------------------------------------------------------------------
# read_jpg_file
# ---------------------------------------------------------------------------


def _fake_bgr_image(value=100):
    return np.full((256, 256, 3), value, dtype=np.uint8)


@patch("app.read_img.cv2.imread")
def test_read_jpg_file_returns_tuple(mock_imread):
    mock_imread.return_value = _fake_bgr_image()
    result = read_jpg_file("fake.jpg")
    assert isinstance(result, tuple)
    assert len(result) == 2


@patch("app.read_img.cv2.imread")
def test_read_jpg_file_array_dtype_is_uint8(mock_imread):
    mock_imread.return_value = _fake_bgr_image()
    array, _ = read_jpg_file("fake.jpg")
    assert array.dtype == np.uint8


@patch("app.read_img.cv2.imread")
def test_read_jpg_file_img2show_is_pil_image(mock_imread):
    mock_imread.return_value = _fake_bgr_image()
    _, img2show = read_jpg_file("fake.jpg")
    assert isinstance(img2show, PIL.Image.Image)


@patch("app.read_img.cv2.imread")
def test_read_jpg_file_normalizes_max_value_to_255(mock_imread):
    mock_imread.return_value = _fake_bgr_image(value=100)
    array, _ = read_jpg_file("fake.jpg")
    assert array.max() == 255


@patch("app.read_img.cv2.imread")
def test_read_jpg_file_preserves_shape(mock_imread):
    mock_imread.return_value = _fake_bgr_image()
    array, _ = read_jpg_file("fake.jpg")
    assert array.shape == (256, 256, 3)


@patch("app.read_img.cv2.imread")
def test_read_jpg_file_calls_imread_with_given_path(mock_imread):
    mock_imread.return_value = _fake_bgr_image()
    read_jpg_file("radiografia.jpg")
    mock_imread.assert_called_once_with("radiografia.jpg")


@patch("app.read_img.cv2.imread")
def test_read_jpg_file_raises_when_file_could_not_be_read(mock_imread):
    mock_imread.return_value = None
    with pytest.raises((AttributeError, TypeError)):
        read_jpg_file("no_existe.jpg")


# ---------------------------------------------------------------------------
# read_dicom_file
# ---------------------------------------------------------------------------


def _fake_dicom_dataset():
    dataset = MagicMock()
    dataset.pixel_array = np.full((256, 256), 150, dtype=np.uint16)
    return dataset


@patch("app.read_img.dicom.dcmread")
def test_read_dicom_file_returns_tuple(mock_dcmread):
    mock_dcmread.return_value = _fake_dicom_dataset()
    result = read_dicom_file("fake.dcm")
    assert isinstance(result, tuple)
    assert len(result) == 2


@patch("app.read_img.dicom.dcmread")
def test_read_dicom_file_converts_to_rgb_three_channels(mock_dcmread):
    mock_dcmread.return_value = _fake_dicom_dataset()
    img_rgb, _ = read_dicom_file("fake.dcm")
    assert img_rgb.shape == (256, 256, 3)


@patch("app.read_img.dicom.dcmread")
def test_read_dicom_file_img2show_is_pil_image(mock_dcmread):
    mock_dcmread.return_value = _fake_dicom_dataset()
    _, img2show = read_dicom_file("fake.dcm")
    assert isinstance(img2show, PIL.Image.Image)


@patch("app.read_img.dicom.dcmread")
def test_read_dicom_file_normalizes_pixel_values_to_255(mock_dcmread):
    mock_dcmread.return_value = _fake_dicom_dataset()
    img_rgb, _ = read_dicom_file("fake.dcm")
    assert img_rgb.max() == 255


@patch("app.read_img.dicom.dcmread")
def test_read_dicom_file_calls_dcmread_with_given_path(mock_dcmread):
    mock_dcmread.return_value = _fake_dicom_dataset()
    read_dicom_file("paciente123.dcm")
    mock_dcmread.assert_called_once_with("paciente123.dcm")
