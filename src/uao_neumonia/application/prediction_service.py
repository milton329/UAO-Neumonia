"""Orquestación del flujo de diagnóstico.

Coordinación entre los módulos de infraestructura: preprocesamiento,
carga del modelo, inferencia y generación de Grad-CAM.
"""

import numpy as np

from uao_neumonia.config import LABELS
from uao_neumonia.domain.models import PredictionResult
from uao_neumonia.infrastructure.grad_cam_generator import generate_grad_cam
from uao_neumonia.infrastructure.image_processor import preprocess
from uao_neumonia.infrastructure.model_loader import load_model


def predict(array: np.ndarray) -> PredictionResult:
    """Ejecuta el flujo completo de diagnóstico sobre una imagen.

    Pipeline: preprocesar → cargar modelo → predecir → obtener
    clase y probabilidad → generar Grad-CAM.

    Args:
        array: Imagen original como array numpy (BGR).

    Returns:
        ``PredictionResult`` con la etiqueta, probabilidad y heatmap.
    """
    batch_array_img = preprocess(array)
    model = load_model()
    predictions = model.predict(batch_array_img)
    prediction = np.argmax(predictions)
    proba = float(np.max(predictions) * 100)
    label = LABELS.get(prediction, "")
    heatmap = generate_grad_cam(array)
    return PredictionResult(label=label, probability=proba, heatmap=heatmap)
