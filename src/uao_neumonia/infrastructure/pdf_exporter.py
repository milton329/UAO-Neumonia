"""Exportación de resultados a PDF.

Compone un collage con la imagen original, el heatmap Grad-CAM, el
diagnóstico y la probabilidad, y lo guarda como PDF usando Pillow.
No depende de ``tkcap`` ni de captura de pantalla, por lo que funciona
en macOS, Linux y Windows.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def export_pdf(
    patient_id: str,
    original_image: np.ndarray,
    heatmap_image: np.ndarray,
    label: str,
    probability: float,
    output_dir: str | Path | None = None,
) -> str:
    """Genera un PDF con el resultado del diagnóstico.

    Crea un collage de 1000×500 px con la imagen original a la izquierda,
    el heatmap a la derecha, y el diagnóstico y probabilidad en la parte
    inferior.

    Args:
        patient_id: Identificador del paciente para nombrar el archivo.
        original_image: Array numpy de la imagen original (BGR).
        heatmap_image: Array numpy del heatmap Grad-CAM (RGB).
        label: Clase diagnosticada.
        probability: Probabilidad en porcentaje.
        output_dir: Directorio de salida. Por defecto, el directorio actual.

    Returns:
        Ruta absoluta al archivo PDF generado.
    """
    if output_dir is None:
        output_dir = Path.cwd()
    original_pil = Image.fromarray(original_image).resize((400, 400), Image.LANCZOS)
    heatmap_pil = Image.fromarray(heatmap_image).resize((400, 400), Image.LANCZOS)

    collage = Image.new("RGB", (1000, 500), "white")
    collage.paste(original_pil, (50, 50))
    collage.paste(heatmap_pil, (550, 50))

    draw = ImageDraw.Draw(collage)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    except OSError:
        font = ImageFont.load_default()
        font_title = ImageFont.load_default()

    draw.text((50, 20), f"Reporte - Paciente: {patient_id}", fill="black", font=font_title)
    draw.text((50, 460), f"Diagnóstico: {label}", fill="black", font=font)
    draw.text((550, 460), f"Probabilidad: {probability:.2f}%", fill="black", font=font)

    pdf_path = Path(output_dir) / f"Reporte_{patient_id}.pdf"
    collage.save(str(pdf_path), "PDF", resolution=100.0)
    return str(pdf_path)
