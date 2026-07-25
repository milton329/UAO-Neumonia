"""Interfaz gráfica de usuario de la herramienta de detección de neumonía.

Ventana construida con Tkinter que permite cargar una radiografía,
ejecutar la predicción, visualizar el heatmap Grad-CAM, guardar el
resultado en el historial CSV y exportar un reporte PDF.
"""

from tkinter import END, StringVar, Text, Tk, filedialog, font, ttk
from tkinter.messagebox import WARNING, askokcancel, showinfo

import numpy as np
from PIL import Image, ImageTk

from uao_neumonia.application.prediction_service import predict
from uao_neumonia.domain.models import PredictionResult
from uao_neumonia.infrastructure.csv_history import save_record
from uao_neumonia.infrastructure.image_reader import read_dicom, read_standard_image
from uao_neumonia.infrastructure.pdf_exporter import export_pdf


class App:
    """Ventana principal de la aplicación.

    Gestiona la carga de imágenes, la ejecución del modelo, la
    visualización de resultados y la exportación a CSV/PDF.
    """
    def __init__(self):
        """Inicializa la ventana, widgets y entra en el bucle de eventos."""
        self.root = Tk()
        self.root.title("Herramienta para la detección rápida de neumonía")

        fonti = font.Font(weight="bold")

        self.root.geometry("815x560")
        self.root.resizable(0, 0)

        self.lab1 = ttk.Label(self.root, text="Imagen Radiográfica", font=fonti)
        self.lab2 = ttk.Label(self.root, text="Imagen con Heatmap", font=fonti)
        self.lab3 = ttk.Label(self.root, text="Resultado:", font=fonti)
        self.lab4 = ttk.Label(self.root, text="Cédula Paciente:", font=fonti)
        self.lab5 = ttk.Label(
            self.root,
            text="SOFTWARE PARA EL APOYO AL DIAGNÓSTICO MÉDICO DE NEUMONÍA",
            font=fonti,
        )
        self.lab6 = ttk.Label(self.root, text="Probabilidad:", font=fonti)

        self.ID = StringVar()
        self.result = StringVar()

        self.text1 = ttk.Entry(self.root, textvariable=self.ID, width=10)

        self.ID_content = self.text1.get()

        self.text_img1 = Text(self.root, width=31, height=15)
        self.text_img2 = Text(self.root, width=31, height=15)
        self.text2 = Text(self.root)
        self.text3 = Text(self.root)

        self.button1 = ttk.Button(
            self.root, text="Predecir", state="disabled", command=self.run_model
        )
        self.button2 = ttk.Button(
            self.root, text="Cargar Imagen", command=self.load_img_file
        )
        self.button3 = ttk.Button(self.root, text="Borrar", command=self.delete)
        self.button4 = ttk.Button(self.root, text="PDF", command=self.create_pdf)
        self.button6 = ttk.Button(
            self.root, text="Guardar", command=self.save_results_csv
        )

        self.lab1.place(x=110, y=65)
        self.lab2.place(x=545, y=65)
        self.lab3.place(x=500, y=350)
        self.lab4.place(x=65, y=350)
        self.lab5.place(x=122, y=25)
        self.lab6.place(x=500, y=400)
        self.button1.place(x=220, y=460)
        self.button2.place(x=70, y=460)
        self.button3.place(x=670, y=460)
        self.button4.place(x=520, y=460)
        self.button6.place(x=370, y=460)
        self.text1.place(x=200, y=350)
        self.text2.place(x=610, y=350, width=90, height=30)
        self.text3.place(x=610, y=400, width=90, height=30)
        self.text_img1.place(x=65, y=90)
        self.text_img2.place(x=500, y=90)

        self.text1.focus_set()

        self.array: np.ndarray | None = None
        self.label = ""
        self.proba = 0.0
        self.heatmap: np.ndarray | None = None

        self.root.mainloop()

    def load_img_file(self):
        """Abre el explorador de archivos y carga una radiografía.

        Lee la imagen (DICOM o JPG/PNG) seleccionada, la muestra en el
        panel izquierdo y habilita el botón Predecir.
        """
        filepath = filedialog.askopenfilename(
            initialdir="/",
            title="Select image",
            filetypes=(
                ("DICOM", "*.dcm"),
                ("JPEG", "*.jpeg"),
                ("jpg files", "*.jpg"),
                ("png files", "*.png"),
            ),
        )
        if filepath:
            if filepath.endswith(".dcm"):
                self.array, img2show = read_dicom(filepath)
            else:
                self.array, img2show = read_standard_image(filepath)
            self.img1 = img2show.resize((250, 250), Image.LANCZOS)
            self.img1 = ImageTk.PhotoImage(self.img1)
            self.text_img1.image_create(END, image=self.img1)
            self.button1["state"] = "normal"

    def run_model(self):
        """Ejecuta la predicción y muestra los resultados.

        Obtiene la clase, probabilidad y heatmap Grad-CAM a través del
        servicio de predicción y los despliega en la interfaz.
        """
        result: PredictionResult = predict(self.array)
        self.label = result.label
        self.proba = result.probability
        self.heatmap = result.heatmap
        self.img2 = Image.fromarray(self.heatmap)
        self.img2 = self.img2.resize((250, 250), Image.LANCZOS)
        self.img2 = ImageTk.PhotoImage(self.img2)
        self.text_img2.image_create(END, image=self.img2)
        self.text2.insert(END, self.label)
        self.text3.insert(END, f"{self.proba:.2f}%")

    def _cedula_paciente(self) -> str:
        """Retorna la cédula ingresada o un valor por defecto.

        Returns:
            Cédula del paciente, o ``"sin_cedula"`` si el campo está vacío.
        """
        cedula = self.text1.get().strip()
        return cedula if cedula else "sin_cedula"

    def save_results_csv(self):
        """Guarda el resultado actual en el historial CSV."""
        save_record(self.text1.get(), self.label, self.proba)
        showinfo(title="Guardar", message="Los datos se guardaron con éxito.")

    def create_pdf(self):
        """Genera un reporte PDF con el diagnóstico actual."""
        cedula = self._cedula_paciente()
        export_pdf(
            patient_id=cedula,
            original_image=self.array,
            heatmap_image=self.heatmap,
            label=self.label,
            probability=self.proba,
        )
        showinfo(title="PDF", message="El PDF fue generado con éxito.")

    def delete(self):
        """Limpia todos los campos de la interfaz, previa confirmación."""
        answer = askokcancel(
            title="Confirmación", message="Se borrarán todos los datos.", icon=WARNING
        )
        if answer:
            self.text1.delete(0, "end")
            self.text2.delete(1.0, "end")
            self.text3.delete(1.0, "end")
            self.text_img1.delete("1.0", "end")
            self.text_img2.delete("1.0", "end")
            showinfo(title="Borrar", message="Los datos se borraron con éxito")


def main() -> int:
    """Punto de entrada de la aplicación gráfica.

    Crea la ventana principal y entra en el bucle de eventos de Tkinter.

    Returns:
        Código de salida 0 indicando ejecución exitosa.
    """
    App()
    return 0


if __name__ == "__main__":
    main()
