"""Persistencia de resultados de predicción en formato CSV.

Guarda cada diagnóstico en ``historial.csv`` con el formato:
``cédula-clase-probabilidad`` separado por guiones.
"""

import csv

from uao_neumonia.config import CSV_FILENAME


def save_record(patient_id: str, label: str, probability: float) -> None:
    """Agrega una línea al historial CSV con el resultado de la predicción.

    Args:
        patient_id: Identificador del paciente (cédula).
        label: Clase diagnosticada ("bacteriana", "viral" o "normal").
        probability: Probabilidad de la clase en porcentaje (0-100).
    """
    with open(CSV_FILENAME, "a", newline="") as csvfile:
        w = csv.writer(csvfile, delimiter="-")
        w.writerow([patient_id, label, f"{probability:.2f}%"])
