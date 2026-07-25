"""Modelos de dominio de la aplicación.

Define las estructuras de datos que se utilizan para transportar
información entre las capas de la aplicación.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class PredictionResult:
    """Resultado de una predicción sobre una radiografía.

    Attributes:
        label: Clase diagnosticada ("bacteriana", "viral" o "normal").
        probability: Probabilidad de la clase en porcentaje (0-100).
        heatmap: Mapa de calor Grad-CAM superpuesto, array (512, 512, 3) uint8 RGB.
    """

    label: str
    probability: float
    heatmap: np.ndarray
