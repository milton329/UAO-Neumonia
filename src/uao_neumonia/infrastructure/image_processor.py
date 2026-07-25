"""Preprocesamiento de la imagen de radiografía antes de la inferencia.

Convierte la imagen cruda en el formato de tensor que espera el modelo:
512x512, escala de grises, ecualización CLAHE y normalizado entre 0 y 1.
"""

import cv2
import numpy as np

from uao_neumonia.config import CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE, IMG_SIZE


def preprocess(array: np.ndarray) -> np.ndarray:
    """Prepara un array de imagen para la entrada del modelo.

    Pipeline: resize a 512×512 → escala de grises → CLAHE →
    normalización 0-1 → expandir dimensiones a formato batch ``(1, 512, 512, 1)``.

    Args:
        array: Imagen como array numpy en formato BGR (alto, ancho, 3).

    Returns:
        Tensor normalizado con forma ``(1, 512, 512, 1)`` listo para ``model.predict()``.
    """
    array = cv2.resize(array, IMG_SIZE)
    array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
    array = clahe.apply(array)
    array = array / 255
    array = np.expand_dims(array, axis=-1)
    array = np.expand_dims(array, axis=0)
    return array
