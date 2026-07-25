"""Constantes centralizadas de configuración de la aplicación.

Todas las rutas, parámetros de preprocesamiento, etiquetas y
configuración de la interfaz gráfica se definen aquí para facilitar
su modificación sin buscar en múltiples archivos.
"""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "conv_MLP_84.h5"

LABELS = {0: "bacteriana", 1: "normal", 2: "viral"}

IMG_SIZE = (512, 512)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (4, 4)

CONV_LAYER_NAME = "conv10_thisone"
HEATMAP_OPACITY = 0.8

WINDOW_SIZE = "815x560"
WINDOW_TITLE = "Herramienta para la detección rápida de neumonía"
WINDOW_SUBTITLE = "SOFTWARE PARA EL APOYO AL DIAGNÓSTICO MÉDICO DE NEUMONÍA"

CSV_FILENAME = "historial.csv"
PDF_DIR = Path.cwd()
