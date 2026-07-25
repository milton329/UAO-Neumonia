"""Lectura de imágenes de radiografía en formato DICOM y JPG/PNG.

Cada función recibe una ruta y retorna el array numpy listo para
preprocesar junto con una imagen PIL para mostrar en la interfaz.
"""

import cv2
import numpy as np
import pydicom as dicom
from PIL import Image


def read_dicom(path: str) -> tuple[np.ndarray, Image.Image]:
    """Lee un archivo DICOM y lo convierte a array RGB.

    Args:
        path: Ruta al archivo ``.dcm``.

    Returns:
        Tupla ``(img_RGB, img2show)`` donde ``img_RGB`` es un array numpy
        de 3 canales normalizado a 0-255 y ``img2show`` es una imagen PIL
        para mostrar en la interfaz.
    """
    img = dicom.dcmread(path)
    img_array = img.pixel_array
    img2show = Image.fromarray(img_array)
    img2 = img_array.astype(float)
    img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
    img2 = np.uint8(img2)
    img_RGB = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    return img_RGB, img2show


def read_standard_image(path: str) -> tuple[np.ndarray, Image.Image]:
    """Lee un archivo JPG/PNG como array numpy.

    Args:
        path: Ruta al archivo de imagen.

    Returns:
        Tupla ``(img_array, img2show)`` donde el array está normalizado
        a 0-255 y ``img2show`` es una imagen PIL para la interfaz.
    """
    img = cv2.imread(path)
    img_array = np.asarray(img)
    img2show = Image.fromarray(img_array)
    img2 = img_array.astype(float)
    img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
    img2 = np.uint8(img2)
    return img2, img2show
