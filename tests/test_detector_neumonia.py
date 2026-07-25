import numpy as np
import pytest

from detector_neumonia import MODEL_PATH, predict, preprocess

REQUIERE_MODELO = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="modelo entrenado no disponible en models/ (no se versiona en git)",
)


def test_preprocess_devuelve_batch_normalizado():
    imagen = (np.random.rand(300, 400, 3) * 255).astype("uint8")

    batch = preprocess(imagen)

    assert batch.shape == (1, 512, 512, 1)
    assert batch.min() >= 0.0
    assert batch.max() <= 1.0


@REQUIERE_MODELO
def test_predict_clasifica_y_genera_heatmap():
    imagen = (np.random.rand(512, 512, 3) * 255).astype("uint8")

    label, proba, heatmap = predict(imagen)

    assert label in {"bacteriana", "normal", "viral"}
    assert 0.0 <= proba <= 100.0
    assert heatmap.shape == (512, 512, 3)
