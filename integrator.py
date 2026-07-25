#!/usr/bin/env python
"""Integra lectura, preprocesamiento, predicción y Grad-CAM para la interfaz."""

import numpy as np

from grad_cam import grad_cam
from load_model import model_fun
from preprocess_img import preprocess

LABELS = {0: "bacteriana", 1: "normal", 2: "viral"}


def predict(array):
    """Ejecuta el flujo completo de diagnóstico sobre una imagen ya leída:
    preprocesa, predice la clase con el modelo y genera el heatmap Grad-CAM.

    Args:
        array: imagen original como array numpy (BGR), tal como la retornan
            `read_dicom_file` o `read_jpg_file`.

    Returns:
        Tupla (label, proba, heatmap): etiqueta de clase ("bacteriana",
        "normal" o "viral"), probabilidad en porcentaje, e imagen con el
        heatmap superpuesto.
    """
    batch_array_img = preprocess(array)
    model = model_fun()
    predictions = model.predict(batch_array_img)
    prediction = np.argmax(predictions)
    proba = np.max(predictions) * 100
    label = LABELS.get(prediction, "")
    heatmap = grad_cam(array)
    return label, proba, heatmap
