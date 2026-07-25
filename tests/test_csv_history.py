from unittest.mock import patch

from uao_neumonia.infrastructure.csv_history import save_record


@patch("uao_neumonia.infrastructure.csv_history.csv.writer")
@patch("uao_neumonia.infrastructure.csv_history.open")
def test_save_record_opens_csv_in_append_mode(mock_open, mock_writer):
    save_record("12345", "normal", 95.5)
    mock_open.assert_called_once_with("historial.csv", "a", newline="")


@patch("uao_neumonia.infrastructure.csv_history.csv.writer")
@patch("uao_neumonia.infrastructure.csv_history.open")
def test_save_record_uses_dash_delimiter(mock_open, mock_writer):
    save_record("12345", "normal", 95.5)
    _, kwargs = mock_writer.call_args
    assert kwargs.get("delimiter") == "-"
