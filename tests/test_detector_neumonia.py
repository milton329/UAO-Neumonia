"""Pruebas unitarias para detector_neumonia.py (interfaz gráfica).

Estrategia: la clase App arma widgets reales de Tkinter en __init__ y
termina en self.root.mainloop() (llamada bloqueante). Para probar la
LÓGICA de cada método sin abrir una ventana real, construimos la
instancia saltándonos __init__ (App.__new__(App)) y asignamos los
atributos que cada método necesita como Mock().
"""

import os
from unittest.mock import MagicMock, Mock, patch

import pytest

from app.detector_neumonia import App


@pytest.fixture
def app():
    """Crea una instancia de App sin ejecutar __init__ (sin abrir ventana),
    lista para que cada test le agregue los atributos que necesite.

    Incluye por defecto root/progress/button1, que usa _set_processing()
    (llamado desde run_model) en cualquier flujo que llegue a predecir.
    root.after se ejecuta de inmediato (no hay hilo de eventos real en los
    tests), simulando el "volver al hilo principal" de la app real.
    """
    instance = App.__new__(App)
    instance.root = Mock()
    instance.root.after.side_effect = lambda _delay, func, *a: func(*a)
    instance.progress = Mock()
    instance.button1 = {}
    return instance


@pytest.fixture
def sync_thread():
    """Reemplaza threading.Thread para que el 'hilo' de predicción corra de
    forma síncrona en el mismo hilo del test, en vez de en paralelo."""
    with patch("app.detector_neumonia.threading.Thread") as mock_thread_cls:
        def fake_thread(target=None, daemon=None):
            fake = Mock()
            fake.start.side_effect = target
            return fake

        mock_thread_cls.side_effect = fake_thread
        yield mock_thread_cls


# ---------------------------------------------------------------------
# _cedula_paciente
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw_input, expected",
    [
        ("", "sin_cedula"),
        ("   ", "sin_cedula"),
        ("\t\n", "sin_cedula"),
        ("123456789", "123456789"),
        ("  123456789  ", "123456789"),
        ("1002003004", "1002003004"),
        ("12 345", "12 345"),  # espacios internos NO se eliminan, solo los de los bordes
    ],
)
def test_cedula_paciente(app, raw_input, expected):
    app.text1 = Mock()
    app.text1.get.return_value = raw_input

    assert app._cedula_paciente() == expected


# ---------------------------------------------------------------------
# load_img_file
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "filepath, expects_dicom_reader",
    [
        ("C:/imagenes/paciente1.dcm", True),
        ("C:/imagenes/paciente1.jpg", False),
        ("C:/imagenes/paciente1.jpeg", False),
        ("C:/imagenes/paciente1.png", False),
        ("C:/imagenes/paciente1.DCM", False),  # el chequeo es sensible a mayúsculas
    ],
)
@patch("app.detector_neumonia.ImageTk.PhotoImage")
@patch("app.detector_neumonia.read_jpg_file")
@patch("app.detector_neumonia.read_dicom_file")
@patch("app.detector_neumonia.filedialog.askopenfilename")
def test_load_img_file_chooses_reader_based_on_extension(
    mock_dialog, mock_read_dicom, mock_read_jpg, mock_photo, app,
    filepath, expects_dicom_reader,
):
    mock_dialog.return_value = filepath
    fake_array, fake_img2show = MagicMock(), MagicMock()
    mock_read_dicom.return_value = (fake_array, fake_img2show)
    mock_read_jpg.return_value = (fake_array, fake_img2show)
    app.text_img1 = Mock()
    app.button1 = {}

    app.load_img_file()

    if expects_dicom_reader:
        mock_read_dicom.assert_called_once_with(filepath)
        mock_read_jpg.assert_not_called()
    else:
        mock_read_jpg.assert_called_once_with(filepath)
        mock_read_dicom.assert_not_called()
    assert app.button1["state"] == "normal"


@patch("app.detector_neumonia.read_jpg_file")
@patch("app.detector_neumonia.read_dicom_file")
@patch("app.detector_neumonia.filedialog.askopenfilename")
def test_load_img_file_does_nothing_when_user_cancels_dialog(
    mock_dialog, mock_read_dicom, mock_read_jpg, app
):
    mock_dialog.return_value = ""
    app.text_img1 = Mock()
    app.button1 = {"state": "disabled"}

    app.load_img_file()

    mock_read_dicom.assert_not_called()
    mock_read_jpg.assert_not_called()
    assert app.button1["state"] == "disabled"


@patch("app.detector_neumonia.ImageTk.PhotoImage")
@patch("app.detector_neumonia.read_jpg_file")
@patch("app.detector_neumonia.filedialog.askopenfilename")
def test_load_img_file_resizes_image_to_250x250(mock_dialog, mock_read_jpg, mock_photo, app):
    mock_dialog.return_value = "C:/imagenes/paciente1.jpg"
    fake_array = MagicMock()
    fake_img2show = MagicMock()
    mock_read_jpg.return_value = (fake_array, fake_img2show)
    app.text_img1 = Mock()
    app.button1 = {}

    app.load_img_file()

    fake_img2show.resize.assert_called_once()
    assert fake_img2show.resize.call_args[0][0] == (250, 250)


# ---------------------------------------------------------------------
# run_model
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "label, proba, expected_text",
    [
        ("Bacteriana", 87.654, "87.65%"),
        ("Viral", 45.321, "45.32%"),
        ("Normal", 12.0, "12.00%"),
        ("Bacteriana", 0.0, "0.00%"),
        ("Viral", 100.0, "100.00%"),
        ("Normal", 99.999, "100.00%"),
    ],
)
@patch("app.detector_neumonia.ImageTk.PhotoImage")
@patch("app.detector_neumonia.Image.fromarray")
@patch("app.detector_neumonia.predict")
def test_run_model_updates_label_and_probability(
    mock_predict, mock_fromarray, mock_photo, app, sync_thread,
    label, proba, expected_text,
):
    mock_predict.return_value = (label, proba, MagicMock())
    mock_fromarray.return_value.resize.return_value = MagicMock()
    app.array = MagicMock()
    app.text1 = Mock()
    app.text1.get.return_value = "123456789"
    app.text_img2 = Mock()
    app.text2 = Mock()
    app.text3 = Mock()

    app.run_model()

    assert app.text2.insert.call_args[0][1] == label
    assert app.text3.insert.call_args[0][1] == expected_text


@patch("app.detector_neumonia.ImageTk.PhotoImage")
@patch("app.detector_neumonia.Image.fromarray")
@patch("app.detector_neumonia.predict")
def test_run_model_passes_loaded_array_to_predict(
    mock_predict, mock_fromarray, mock_photo, app, sync_thread
):
    mock_predict.return_value = ("Normal", 50.0, MagicMock())
    app.array = MagicMock()
    app.text1 = Mock()
    app.text1.get.return_value = "123456789"
    app.text_img2 = Mock()
    app.text2 = Mock()
    app.text3 = Mock()

    app.run_model()

    mock_predict.assert_called_once_with(app.array)


@patch("app.detector_neumonia.ImageTk.PhotoImage")
@patch("app.detector_neumonia.Image.fromarray")
@patch("app.detector_neumonia.predict")
def test_run_model_builds_heatmap_image_from_prediction(
    mock_predict, mock_fromarray, mock_photo, app, sync_thread
):
    fake_heatmap = MagicMock()
    mock_predict.return_value = ("Normal", 50.0, fake_heatmap)
    app.array = MagicMock()
    app.text1 = Mock()
    app.text1.get.return_value = "123456789"
    app.text_img2 = Mock()
    app.text2 = Mock()
    app.text3 = Mock()

    app.run_model()

    mock_fromarray.assert_called_once_with(fake_heatmap)


@pytest.mark.parametrize(
    "cedula_input",
    [
        "",              # vacía
        "   ",           # solo espacios
        "12345",         # 5 dígitos, por debajo del mínimo
        "12345678901",   # 11 dígitos, por encima del máximo (bloqueado igual por el keystroke, pero se valida aquí también)
        "12a456",        # no numérica
        "123 456",       # con espacio interno
    ],
)
@patch("app.detector_neumonia.showwarning")
@patch("app.detector_neumonia.predict")
def test_run_model_blocks_prediction_when_cedula_is_invalid(
    mock_predict, mock_showwarning, app, cedula_input
):
    app.array = MagicMock()
    app.text1 = Mock()
    app.text1.get.return_value = cedula_input

    app.run_model()

    mock_predict.assert_not_called()
    mock_showwarning.assert_called_once()
    app.text1.focus_set.assert_called_once()


@pytest.mark.parametrize("cedula_input", ["123456", "1234567890", "987654321"])
@patch("app.detector_neumonia.ImageTk.PhotoImage")
@patch("app.detector_neumonia.Image.fromarray")
@patch("app.detector_neumonia.showwarning")
@patch("app.detector_neumonia.predict")
def test_run_model_predicts_when_cedula_has_valid_format(
    mock_predict, mock_showwarning, mock_fromarray, mock_photo, app, sync_thread,
    cedula_input,
):
    mock_predict.return_value = ("Normal", 50.0, MagicMock())
    app.array = MagicMock()
    app.text1 = Mock()
    app.text1.get.return_value = cedula_input
    app.text_img2 = Mock()
    app.text2 = Mock()
    app.text3 = Mock()

    app.run_model()

    mock_predict.assert_called_once_with(app.array)
    mock_showwarning.assert_not_called()


@patch("app.detector_neumonia.predict")
def test_run_model_shows_progress_bar_while_predicting(mock_predict, app):
    """No usa sync_thread a propósito: queremos observar el estado de la
    interfaz MIENTRAS el hilo de fondo (nunca lanzado de verdad aquí,
    solo simulado) está corriendo, antes de que termine."""
    with patch("app.detector_neumonia.threading.Thread") as mock_thread_cls:
        mock_thread_cls.return_value = Mock()  # start() no hace nada
        app.array = MagicMock()
        app.text1 = Mock()
        app.text1.get.return_value = "123456789"

        app.run_model()

        assert app.button1["state"] == "disabled"
        app.progress.start.assert_called_once()


@patch("app.detector_neumonia.ImageTk.PhotoImage")
@patch("app.detector_neumonia.Image.fromarray")
@patch("app.detector_neumonia.predict")
def test_run_model_hides_progress_bar_after_predicting(
    mock_predict, mock_fromarray, mock_photo, app, sync_thread
):
    mock_predict.return_value = ("Normal", 50.0, MagicMock())
    app.array = MagicMock()
    app.text1 = Mock()
    app.text1.get.return_value = "123456789"
    app.text_img2 = Mock()
    app.text2 = Mock()
    app.text3 = Mock()

    app.run_model()

    app.progress.stop.assert_called_once()
    assert app.button1["state"] == "normal"


@patch("app.detector_neumonia.showerror")
@patch("app.detector_neumonia.predict")
def test_run_model_shows_error_and_restores_button_when_predict_fails(
    mock_predict, mock_showerror, app, sync_thread
):
    mock_predict.side_effect = RuntimeError("el modelo no pudo cargar")
    app.array = MagicMock()
    app.text1 = Mock()
    app.text1.get.return_value = "123456789"

    app.run_model()

    mock_showerror.assert_called_once()
    assert app.button1["state"] == "normal"
    app.progress.stop.assert_called_once()


class TestValidateCedulaKeystroke:
    """Pruebas del validador que restringe el campo de cédula mientras se
    escribe (solo dígitos, máximo 10 caracteres)."""

    @pytest.mark.parametrize(
        "proposed_value, expected",
        [
            ("", True),
            ("1", True),
            ("123456", True),
            ("1234567890", True),
            ("12345678901", False),  # 11 dígitos, supera el máximo
            ("12a", False),
            ("12 34", False),
        ],
    )
    def test_validate_cedula_keystroke(self, proposed_value, expected):
        assert App._validate_cedula_keystroke(proposed_value) is expected


# ---------------------------------------------------------------------
# save_results_csv
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "cedula, label, proba, expected_row",
    [
        ("123456789", "Normal", 12.0, "123456789-Normal-12.00%"),
        ("987654321", "Bacteriana", 55.5, "987654321-Bacteriana-55.50%"),
        ("", "Viral", 33.333, "-Viral-33.33%"),  # ojo: aquí NO se usa _cedula_paciente
        ("111", "Normal", 100.0, "111-Normal-100.00%"),
    ],
)
@patch("app.detector_neumonia.showinfo")
def test_save_results_csv_writes_expected_row(
    mock_showinfo, app, tmp_path, monkeypatch, cedula, label, proba, expected_row
):
    monkeypatch.chdir(tmp_path)
    app.text1 = Mock()
    app.text1.get.return_value = cedula
    app.label = label
    app.proba = proba

    app.save_results_csv()

    contenido = (tmp_path / "reportes" / "historial.csv").read_text()
    assert expected_row in contenido


@patch("app.detector_neumonia.showinfo")
def test_save_results_csv_shows_success_message(mock_showinfo, app, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app.text1 = Mock()
    app.text1.get.return_value = "123456789"
    app.label = "Normal"
    app.proba = 10.0

    app.save_results_csv()

    mock_showinfo.assert_called_once()


@patch("app.detector_neumonia.showinfo")
def test_save_results_csv_appends_multiple_entries(mock_showinfo, app, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app.text1 = Mock()
    app.text1.get.return_value = "111"
    app.label = "Normal"
    app.proba = 10.0
    app.save_results_csv()

    app.text1.get.return_value = "222"
    app.label = "Viral"
    app.proba = 20.0
    app.save_results_csv()

    contenido = (tmp_path / "reportes" / "historial.csv").read_text()
    assert "111-Normal-10.00%" in contenido
    assert "222-Viral-20.00%" in contenido


# ---------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------

@patch("app.detector_neumonia.showinfo")
@patch("app.detector_neumonia.askokcancel")
def test_delete_clears_all_fields_when_user_confirms(mock_confirm, mock_showinfo, app):
    mock_confirm.return_value = True
    app.text1 = Mock()
    app.text2 = Mock()
    app.text3 = Mock()
    app.text_img1 = Mock()
    app.text_img2 = Mock()

    app.delete()

    app.text1.delete.assert_called_once_with(0, "end")
    app.text2.delete.assert_called_once_with(1.0, "end")
    app.text3.delete.assert_called_once_with(1.0, "end")
    app.text_img1.delete.assert_called_once_with("1.0", "end")
    app.text_img2.delete.assert_called_once_with("1.0", "end")


@patch("app.detector_neumonia.askokcancel")
def test_delete_does_not_clear_fields_when_user_cancels(mock_confirm, app):
    mock_confirm.return_value = False
    app.text1 = Mock()
    app.text2 = Mock()
    app.text3 = Mock()
    app.text_img1 = Mock()
    app.text_img2 = Mock()

    app.delete()

    app.text1.delete.assert_not_called()
    app.text2.delete.assert_not_called()
    app.text3.delete.assert_not_called()


@patch("app.detector_neumonia.showinfo")
@patch("app.detector_neumonia.askokcancel")
def test_delete_shows_success_message_when_confirmed(mock_confirm, mock_showinfo, app):
    mock_confirm.return_value = True
    app.text1 = Mock()
    app.text2 = Mock()
    app.text3 = Mock()
    app.text_img1 = Mock()
    app.text_img2 = Mock()

    app.delete()

    mock_showinfo.assert_called_once()


@patch("app.detector_neumonia.showinfo")
@patch("app.detector_neumonia.askokcancel")
def test_delete_does_not_show_message_when_cancelled(mock_confirm, mock_showinfo, app):
    mock_confirm.return_value = False
    app.text1 = Mock()
    app.text2 = Mock()
    app.text3 = Mock()
    app.text_img1 = Mock()
    app.text_img2 = Mock()

    app.delete()

    mock_showinfo.assert_not_called()


@patch("app.detector_neumonia.askokcancel")
def test_delete_asks_confirmation_before_clearing(mock_confirm, app):
    mock_confirm.return_value = True
    app.text1 = Mock()
    app.text2 = Mock()
    app.text3 = Mock()
    app.text_img1 = Mock()
    app.text_img2 = Mock()

    app.delete()

    mock_confirm.assert_called_once()


# ---------------------------------------------------------------------
# create_pdf
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "cedula_input, expected_filename_id",
    [
        ("987654321", "987654321"),
        ("", "sin_cedula"),
        ("  555  ", "555"),
    ],
)
@patch("app.detector_neumonia.showinfo")
@patch("app.detector_neumonia.capture_window")
def test_create_pdf_names_file_with_patient_id(
    mock_capture_window, mock_showinfo, app, tmp_path, monkeypatch,
    cedula_input, expected_filename_id,
):
    monkeypatch.chdir(tmp_path)
    app.text1 = Mock()
    app.text1.get.return_value = cedula_input
    app.root = Mock()
    mock_img = mock_capture_window.return_value
    mock_img.convert.return_value = mock_img
    expected_jpg = os.path.join("reportes", f"Reporte_{expected_filename_id}.jpg")
    expected_pdf = os.path.join("reportes", f"Reporte_{expected_filename_id}.pdf")

    app.create_pdf()

    mock_img.save.assert_any_call(expected_jpg)
    mock_img.save.assert_any_call(expected_pdf)


@patch("app.detector_neumonia.showinfo")
@patch("app.detector_neumonia.capture_window")
def test_create_pdf_converts_image_to_rgb(
    mock_capture_window, mock_showinfo, app, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    app.text1 = Mock()
    app.text1.get.return_value = "123456789"
    app.root = Mock()
    mock_img = mock_capture_window.return_value

    app.create_pdf()

    mock_img.convert.assert_called_once_with("RGB")


@patch("app.detector_neumonia.showinfo")
@patch("app.detector_neumonia.capture_window")
def test_create_pdf_shows_success_message(
    mock_capture_window, mock_showinfo, app, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    app.text1 = Mock()
    app.text1.get.return_value = "123456789"
    app.root = Mock()
    mock_img = mock_capture_window.return_value
    mock_img.convert.return_value = mock_img

    app.create_pdf()

    mock_showinfo.assert_called_once()


@patch("app.detector_neumonia.showinfo")
@patch("app.detector_neumonia.capture_window")
def test_create_pdf_captures_the_app_window(
    mock_capture_window, mock_showinfo, app, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    app.text1 = Mock()
    app.text1.get.return_value = "123456789"
    app.root = Mock()
    mock_img = mock_capture_window.return_value
    mock_img.convert.return_value = mock_img

    app.create_pdf()

    mock_capture_window.assert_called_once_with(app.root)

    # Para correr en el terminak -> uv run pytest tests/test_detector_neumonia.py -v