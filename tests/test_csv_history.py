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


@patch("uao_neumonia.infrastructure.csv_history.csv.writer")
@patch("uao_neumonia.infrastructure.csv_history.open")
def test_save_record_writes_patient_id(mock_open, mock_writer):
    save_record("P-001", "bacteriana", 99.0)
    writer_instance = mock_writer.return_value
    writer_instance.writerow.assert_called_once()
    args = writer_instance.writerow.call_args[0][0]
    assert args[0] == "P-001"


@patch("uao_neumonia.infrastructure.csv_history.csv.writer")
@patch("uao_neumonia.infrastructure.csv_history.open")
def test_save_record_writes_label(mock_open, mock_writer):
    save_record("123", "viral", 80.0)
    writer_instance = mock_writer.return_value
    args = writer_instance.writerow.call_args[0][0]
    assert args[1] == "viral"


@patch("uao_neumonia.infrastructure.csv_history.csv.writer")
@patch("uao_neumonia.infrastructure.csv_history.open")
def test_save_record_writes_probability_with_percent(mock_open, mock_writer):
    save_record("123", "normal", 75.3)
    writer_instance = mock_writer.return_value
    args = writer_instance.writerow.call_args[0][0]
    assert args[2] == "75.30%"


@patch("uao_neumonia.infrastructure.csv_history.csv.writer")
@patch("uao_neumonia.infrastructure.csv_history.open")
def test_save_record_calls_writerow_once(mock_open, mock_writer):
    save_record("123", "normal", 50.0)
    writer_instance = mock_writer.return_value
    writer_instance.writerow.assert_called_once()
