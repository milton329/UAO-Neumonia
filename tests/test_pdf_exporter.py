import tempfile
from pathlib import Path

import numpy as np
import pytest

from uao_neumonia.infrastructure.pdf_exporter import export_pdf


@pytest.fixture
def sample_images():
    original = np.full((512, 512, 3), 100, dtype=np.uint8)
    heatmap = np.full((512, 512, 3), 200, dtype=np.uint8)
    return original, heatmap


def test_export_pdf_returns_string_path(sample_images):
    original, heatmap = sample_images
    with tempfile.TemporaryDirectory() as tmpdir:
        result = export_pdf("123", original, heatmap, "normal", 95.5, output_dir=tmpdir)
        assert isinstance(result, str)
        assert result.endswith(".pdf")


def test_pdf_file_is_created(sample_images):
    original, heatmap = sample_images
    with tempfile.TemporaryDirectory() as tmpdir:
        result = export_pdf("123", original, heatmap, "normal", 95.5, output_dir=tmpdir)
        assert Path(result).exists()
        assert Path(result).stat().st_size > 0


def test_export_pdf_uses_custom_output_dir(sample_images):
    original, heatmap = sample_images
    with tempfile.TemporaryDirectory() as tmpdir:
        result = export_pdf("456", original, heatmap, "viral", 80.0, output_dir=tmpdir)
        assert Path(result).parent == Path(tmpdir)


def test_pdf_filename_includes_patient_id(sample_images):
    original, heatmap = sample_images
    with tempfile.TemporaryDirectory() as tmpdir:
        result = export_pdf("P-001", original, heatmap, "bacteriana", 99.0, output_dir=tmpdir)
        assert "P-001" in Path(result).name


def test_export_pdf_works_with_default_output_dir(sample_images):
    original, heatmap = sample_images
    result = export_pdf("default_test", original, heatmap, "normal", 50.0)
    path = Path(result)
    try:
        assert path.exists()
    finally:
        path.unlink(missing_ok=True)


def test_handles_patient_id_with_spaces(sample_images):
    original, heatmap = sample_images
    with tempfile.TemporaryDirectory() as tmpdir:
        result = export_pdf("id with spaces", original, heatmap, "normal", 70.0, output_dir=tmpdir)
        assert Path(result).exists()


def test_handles_empty_label(sample_images):
    original, heatmap = sample_images
    with tempfile.TemporaryDirectory() as tmpdir:
        result = export_pdf("empty_label", original, heatmap, "", 0.0, output_dir=tmpdir)
        assert Path(result).exists()


def test_probability_formatting_zero(sample_images):
    original, heatmap = sample_images
    with tempfile.TemporaryDirectory() as tmpdir:
        result = export_pdf("zero", original, heatmap, "normal", 0.0, output_dir=tmpdir)
        assert Path(result).exists()


def test_probability_formatting_one_hundred(sample_images):
    original, heatmap = sample_images
    with tempfile.TemporaryDirectory() as tmpdir:
        result = export_pdf("hundred", original, heatmap, "normal", 100.0, output_dir=tmpdir)
        assert Path(result).exists()


def test_handles_different_image_shapes():
    original = np.full((256, 256, 3), 50, dtype=np.uint8)
    heatmap = np.full((256, 256, 3), 150, dtype=np.uint8)
    with tempfile.TemporaryDirectory() as tmpdir:
        result = export_pdf("small", original, heatmap, "viral", 60.0, output_dir=tmpdir)
        assert Path(result).exists()
