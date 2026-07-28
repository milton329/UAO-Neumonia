#!/usr/bin/env python
"""Carga del modelo de red neuronal convolucional ya entrenado."""

from functools import lru_cache
from pathlib import Path

import tensorflow as tf

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "conv_MLP_84.h5"


# Se cachea: el modelo pesa ~117MB e integrator.predict()/grad_cam.grad_cam()
# lo invocaban varias veces por cada predicción, recargándolo cada vez.
@lru_cache(maxsize=1)
def model_fun():
    """Carga y retorna el modelo entrenado de clasificación de neumonía.

    Returns:
        Instancia de `tf.keras.Model` cargada desde `models/conv_MLP_84.h5`.
    """
    return tf.keras.models.load_model(MODEL_PATH, compile=False)
