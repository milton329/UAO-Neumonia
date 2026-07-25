#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generación del mapa de calor Grad-CAM sobre la imagen de radiografía."""

import cv2
import numpy as np
import tensorflow as tf

from load_model import model_fun
from preprocess_img import preprocess


def grad_cam(array):
    """Genera una imagen con un mapa de calor Grad-CAM superpuesto, resaltando
    las regiones de la radiografía que más influyeron en la predicción del
    modelo.

    Args:
        array: imagen original como array numpy (BGR).

    Returns:
        Array numpy (RGB) con el heatmap superpuesto sobre la imagen original.
    """
    img = preprocess(array)
    model = model_fun()
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer("conv10_thisone").output, model.output]
    )
    with tf.GradientTape() as tape:
        inputs = tf.cast(img, tf.float32)
        outputs = grad_model(inputs)
        conv_outputs = outputs[0]
        # Handle single tensor or list of tensors from multi-output models
        if len(outputs) == 2:
            raw_preds = tf.cast(outputs[1], tf.float32)
        else:
            raw_preds = tf.stack([tf.reshape(tf.cast(outputs[i], tf.float32), [-1]) for i in range(1, len(outputs))], axis=0)
            raw_preds = tf.expand_dims(raw_preds, 0)
        predictions = tf.reshape(raw_preds, [1, -1])
        argmax = int(np.argmax(predictions[0]))
        loss = predictions[:, argmax]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = cv2.resize(heatmap, (512, 512))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    img2 = cv2.resize(array, (512, 512))
    hif = 0.8
    transparency = heatmap * hif
    transparency = transparency.astype(np.uint8)
    superimposed_img = cv2.add(transparency, img2)
    superimposed_img = superimposed_img.astype(np.uint8)
    return superimposed_img[:, :, ::-1]
