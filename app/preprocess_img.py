#!/usr/bin/env python
"""Preprocesamiento de la imagen de radiografía antes de pasarla al modelo."""

import cv2
import numpy as np


def preprocess(array):
    """Prepara un array de imagen para el modelo: resize a 512x512, escala de
    grises, ecualización de histograma (CLAHE), normalización entre 0 y 1, y
    conversión a formato de batch (tensor) que espera el modelo.

    Args:
        array: imagen como array numpy (BGR).

    Returns:
        Array numpy con shape (1, 512, 512, 1) normalizado entre 0 y 1.
    """
    array = cv2.resize(array, (512, 512))
    array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    array = clahe.apply(array)
    array = array / 255
    array = np.expand_dims(array, axis=-1)
    array = np.expand_dims(array, axis=0)
    return array
