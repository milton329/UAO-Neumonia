#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fixtures compartidos entre las pruebas unitarias."""

import numpy as np
import pytest
import tensorflow as tf


@pytest.fixture
def tiny_cnn_model():
    """Modelo Keras diminuto, con una capa 'conv10_thisone', suficiente para
    ejercitar grad_cam.grad_cam sin depender del modelo real de 113MB."""
    inputs = tf.keras.Input(shape=(512, 512, 1))
    x = tf.keras.layers.Conv2D(4, 3, padding="same", name="conv10_thisone")(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(3, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)


@pytest.fixture
def sample_bgr_image():
    rng = np.random.default_rng(seed=42)
    return rng.integers(0, 255, size=(512, 512, 3), dtype=np.uint8)
