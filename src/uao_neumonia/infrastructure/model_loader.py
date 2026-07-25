"""Carga del modelo de clasificación entrenado.

Gestiona la carga de `conv_MLP_84.h5` con caché singleton para evitar
recargar los ~117 MB del modelo en múltiples invocaciones.
"""

from functools import lru_cache

import tensorflow as tf

from uao_neumonia.config import MODEL_PATH


@lru_cache(maxsize=1)
def load_model() -> tf.keras.Model:
    """Carga y retorna el modelo entrenado de clasificación.

    El resultado se cachea con ``lru_cache`` para que las llamadas
    posteriores reutilicen la misma instancia sin recargar el archivo.

    Returns:
        Modelo ``tf.keras.Model`` compilado desde el archivo ``.h5``.
    """
    return tf.keras.models.load_model(MODEL_PATH, compile=False)
