"""Generación del mapa de calor Grad-CAM sobre la radiografía.

Grad-CAM (Gradient-weighted Class Activation Mapping) calcula el gradiente
de la salida correspondiente a la clase predicha respecto a las neuronas
de la última capa convolucional, produciendo un heatmap que resalta las
regiones más influyentes en la decisión del modelo.
"""

import cv2
import numpy as np
import tensorflow as tf

from uao_neumonia.config import CONV_LAYER_NAME, HEATMAP_OPACITY, IMG_SIZE
from uao_neumonia.infrastructure.image_processor import preprocess
from uao_neumonia.infrastructure.model_loader import load_model


def generate_grad_cam(array: np.ndarray) -> np.ndarray:
    """Genera un mapa de calor Grad-CAM superpuesto sobre la imagen original.

    Args:
        array: Imagen original como array numpy en formato BGR.

    Returns:
        Array RGB ``(512, 512, 3)`` uint8 con el heatmap superpuesto.
    """
    img = preprocess(array)
    model = load_model()
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(CONV_LAYER_NAME).output, model.output],
    )
    with tf.GradientTape() as tape:
        inputs = tf.cast(img, tf.float32)
        outputs = grad_model(inputs)
        conv_outputs = outputs[0]
        if len(outputs) == 2:
            raw_preds = tf.cast(outputs[1], tf.float32)
        else:
            raw_preds = tf.stack(
                [
                    tf.reshape(tf.cast(outputs[i], tf.float32), [-1])
                    for i in range(1, len(outputs))
                ],
                axis=0,
            )
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
    heatmap = cv2.resize(heatmap, IMG_SIZE)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    img2 = cv2.resize(array, IMG_SIZE)
    transparency = heatmap * HEATMAP_OPACITY
    transparency = transparency.astype(np.uint8)
    superimposed_img = cv2.add(transparency, img2)
    superimposed_img = superimposed_img.astype(np.uint8)
    return superimposed_img[:, :, ::-1]
