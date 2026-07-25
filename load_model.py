#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Carga del modelo de red neuronal convolucional ya entrenado."""

import tensorflow as tf

MODEL_PATH = "conv_MLP_84.h5"


def model_fun():
    """Carga y retorna el modelo entrenado de clasificación de neumonía.

    Returns:
        Instancia de `tf.keras.Model` cargada desde `conv_MLP_84.h5`.
    """
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model
