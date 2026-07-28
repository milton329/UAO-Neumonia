from pathlib import Path

from uao_neumonia import config


def test_model_path_points_to_correct_file():
    assert config.MODEL_PATH.name == "conv_MLP_84.h5"
    assert config.MODEL_PATH.parent.name == "models"


def test_project_root_is_absolute_path():
    assert isinstance(config.PROJECT_ROOT, Path)
    assert config.PROJECT_ROOT.is_absolute()


def test_labels_contains_three_classes():
    assert len(config.LABELS) == 3
    assert config.LABELS[0] == "bacteriana"
    assert config.LABELS[1] == "normal"
    assert config.LABELS[2] == "viral"


def test_img_size_is_tuple_of_ints():
    assert isinstance(config.IMG_SIZE, tuple)
    assert len(config.IMG_SIZE) == 2
    assert all(isinstance(d, int) for d in config.IMG_SIZE)
    assert config.IMG_SIZE == (512, 512)


def test_clahe_constants_have_correct_types():
    assert isinstance(config.CLAHE_CLIP_LIMIT, float)
    assert isinstance(config.CLAHE_TILE_GRID_SIZE, tuple)
    assert len(config.CLAHE_TILE_GRID_SIZE) == 2


def test_window_constants_are_strings():
    assert isinstance(config.WINDOW_SIZE, str)
    assert isinstance(config.WINDOW_TITLE, str)
    assert isinstance(config.WINDOW_SUBTITLE, str)
    assert len(config.WINDOW_SIZE) > 0


def test_csv_filename_is_historial():
    assert config.CSV_FILENAME == "historial.csv"


def test_conv_layer_name_is_set():
    assert isinstance(config.CONV_LAYER_NAME, str)
    assert len(config.CONV_LAYER_NAME) > 0


def test_heatmap_opacity_is_between_0_and_1():
    assert 0 < config.HEATMAP_OPACITY < 1


def test_pdf_dir_is_path():
    assert isinstance(config.PDF_DIR, Path)
