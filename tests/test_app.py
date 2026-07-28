from unittest.mock import MagicMock, patch

import numpy as np

from uao_neumonia.presentation.app import App


@patch("uao_neumonia.presentation.app.Tk")
@patch("uao_neumonia.presentation.app.font.Font")
@patch("uao_neumonia.presentation.app.ttk.Label")
@patch("uao_neumonia.presentation.app.ttk.Entry")
@patch("uao_neumonia.presentation.app.Text")
@patch("uao_neumonia.presentation.app.ttk.Button")
@patch("uao_neumonia.presentation.app.StringVar")
def test_app_initialization(mock_sv, mock_btn, mock_txt, mock_entry,
                             mock_label, mock_font, mock_tk):
    app = App()
    assert app.root is not None
    mock_tk.assert_called_once()


@patch("uao_neumonia.presentation.app.Tk")
@patch("uao_neumonia.presentation.app.font.Font")
@patch("uao_neumonia.presentation.app.ttk.Label")
@patch("uao_neumonia.presentation.app.ttk.Entry")
@patch("uao_neumonia.presentation.app.Text")
@patch("uao_neumonia.presentation.app.ttk.Button")
@patch("uao_neumonia.presentation.app.StringVar")
def test_cedula_paciente_returns_sin_cedula_when_empty(
    mock_sv, mock_btn, mock_txt, mock_entry, mock_label, mock_font, mock_tk
):
    mock_stringvar = MagicMock()
    mock_stringvar.get.return_value = ""
    mock_sv.return_value = mock_stringvar
    app = App()
    app.text1 = MagicMock()
    app.text1.get.return_value = ""
    assert app._cedula_paciente() == "sin_cedula"


@patch("uao_neumonia.presentation.app.Tk")
@patch("uao_neumonia.presentation.app.font.Font")
@patch("uao_neumonia.presentation.app.ttk.Label")
@patch("uao_neumonia.presentation.app.ttk.Entry")
@patch("uao_neumonia.presentation.app.Text")
@patch("uao_neumonia.presentation.app.ttk.Button")
@patch("uao_neumonia.presentation.app.StringVar")
def test_cedula_paciente_returns_id_when_set(
    mock_sv, mock_btn, mock_txt, mock_entry, mock_label, mock_font, mock_tk
):
    mock_stringvar = MagicMock()
    mock_stringvar.get.return_value = "12345"
    mock_sv.return_value = mock_stringvar
    app = App()
    app.text1 = MagicMock()
    app.text1.get.return_value = "12345"
    assert app._cedula_paciente() == "12345"


@patch("uao_neumonia.presentation.app.Tk")
@patch("uao_neumonia.presentation.app.font.Font")
@patch("uao_neumonia.presentation.app.ttk.Label")
@patch("uao_neumonia.presentation.app.ttk.Entry")
@patch("uao_neumonia.presentation.app.Text")
@patch("uao_neumonia.presentation.app.ttk.Button")
@patch("uao_neumonia.presentation.app.StringVar")
@patch("uao_neumonia.presentation.app.save_record")
@patch("uao_neumonia.presentation.app.showinfo")
def test_save_results_csv_calls_save_record(
    mock_showinfo, mock_save, mock_sv, mock_btn, mock_txt,
    mock_entry, mock_label, mock_font, mock_tk
):
    app = App()
    app.text1 = MagicMock()
    app.text1.get.return_value = "999"
    app.label = "bacteriana"
    app.proba = 88.5
    app.save_results_csv()
    mock_save.assert_called_once_with("999", "bacteriana", 88.5)


@patch("uao_neumonia.presentation.app.Tk")
@patch("uao_neumonia.presentation.app.font.Font")
@patch("uao_neumonia.presentation.app.ttk.Label")
@patch("uao_neumonia.presentation.app.ttk.Entry")
@patch("uao_neumonia.presentation.app.Text")
@patch("uao_neumonia.presentation.app.ttk.Button")
@patch("uao_neumonia.presentation.app.StringVar")
@patch("uao_neumonia.presentation.app.export_pdf")
@patch("uao_neumonia.presentation.app.showinfo")
def test_create_pdf_calls_export_pdf(
    mock_showinfo, mock_export, mock_sv, mock_btn, mock_txt,
    mock_entry, mock_label, mock_font, mock_tk
):
    app = App()
    app.text1 = MagicMock()
    app.text1.get.return_value = "555"
    app.array = np.zeros((512, 512, 3), dtype=np.uint8)
    app.heatmap = np.ones((512, 512, 3), dtype=np.uint8)
    app.label = "viral"
    app.proba = 70.0
    app.create_pdf()
    mock_export.assert_called_once_with(
        patient_id="555",
        original_image=app.array,
        heatmap_image=app.heatmap,
        label="viral",
        probability=70.0,
    )
