#!/usr/bin/env python
"""Interfaz gráfica (Tkinter) de la herramienta de detección de neumonía."""

import csv
import os
import re
import threading
from tkinter import END, StringVar, Text, filedialog
from tkinter.messagebox import WARNING, askokcancel, showerror, showinfo, showwarning

import ttkbootstrap as ttk
from PIL import Image, ImageTk

from app.integrator import predict
from app.read_img import read_dicom_file, read_jpg_file
from app.window_capture import capture_window

REPORTS_DIR = "reportes"
CEDULA_PATTERN = re.compile(r"^\d{6,10}$")


class App:
    """Ventana principal: carga una radiografía, ejecuta la predicción y
    permite guardar/exportar el resultado."""

    def __init__(self):
        self.root = ttk.Window(
            title="Herramienta para la detección rápida de neumonía",
            themename="flatly",
            resizable=(False, False),
        )
        self.root.configure(padx=16, pady=12)

        self._setup_style()

        header = ttk.Label(
            self.root,
            text="Software para el apoyo al diagnóstico médico de neumonía",
            style="Header.TLabel",
        )
        header.grid(row=0, column=0, columnspan=2, pady=(0, 12), sticky="w")

        #   PANELES DE IMAGEN
        img_frame_1 = ttk.LabelFrame(self.root, text="Imagen radiográfica", padding=8)
        img_frame_1.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        self.text_img1 = Text(img_frame_1, width=31, height=15, borderwidth=0)
        self.text_img1.grid(row=0, column=0)

        img_frame_2 = ttk.LabelFrame(self.root, text="Imagen con heatmap", padding=8)
        img_frame_2.grid(row=1, column=1, sticky="nsew")
        self.text_img2 = Text(img_frame_2, width=31, height=15, borderwidth=0)
        self.text_img2.grid(row=0, column=0)

        #   PANEL DE DATOS DEL PACIENTE Y RESULTADO
        info_frame = ttk.LabelFrame(
            self.root, text="Datos del paciente y resultado", padding=12
        )
        info_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=12)
        for col in (1, 3, 5):
            info_frame.columnconfigure(col, weight=1)

        self.ID = StringVar()
        self.result = StringVar()

        ttk.Label(info_frame, text="Cédula paciente:", style="Field.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 6)
        )
        vcmd = (self.root.register(self._validate_cedula_keystroke), "%P")
        self.text1 = ttk.Entry(
            info_frame,
            textvariable=self.ID,
            width=16,
            validate="key",
            validatecommand=vcmd,
        )
        self.text1.grid(row=0, column=1, sticky="w", padx=(0, 18))

        ttk.Label(info_frame, text="Resultado:", style="Field.TLabel").grid(
            row=0, column=2, sticky="w", padx=(0, 6)
        )
        self.text2 = Text(info_frame, width=14, height=1, **self._entry_look())
        self.text2.grid(row=0, column=3, sticky="w", padx=(0, 18))

        ttk.Label(info_frame, text="Probabilidad:", style="Field.TLabel").grid(
            row=0, column=4, sticky="w", padx=(0, 6)
        )
        self.text3 = Text(info_frame, width=10, height=1, **self._entry_look())
        self.text3.grid(row=0, column=5, sticky="w")

        #   BOTONES DE ACCIÓN
        button_bar = ttk.Frame(self.root)
        button_bar.grid(row=3, column=0, columnspan=2, pady=(4, 0))

        self.button2 = ttk.Button(
            button_bar,
            text="📁  Cargar imagen",
            command=self.load_img_file,
            bootstyle="primary-outline",
        )
        self.button1 = ttk.Button(
            button_bar,
            text="🔍  Predecir",
            state="disabled",
            command=self.run_model,
            bootstyle="success",
        )
        self.button6 = ttk.Button(
            button_bar,
            text="💾  Guardar",
            command=self.save_results_csv,
            bootstyle="info-outline",
        )
        self.button4 = ttk.Button(
            button_bar,
            text="📄  PDF",
            command=self.create_pdf,
            bootstyle="info-outline",
        )
        self.button3 = ttk.Button(
            button_bar,
            text="🗑️  Borrar",
            command=self.delete,
            bootstyle="danger-outline",
        )

        for i, btn in enumerate(
            (self.button2, self.button1, self.button6, self.button4, self.button3)
        ):
            btn.grid(row=0, column=i, padx=6)

        #   BARRA DE PROGRESO (oculta hasta que se está prediciendo)
        self.progress = ttk.Progressbar(
            self.root, mode="indeterminate", bootstyle="success-striped"
        )

        #   FOCUS EN LA CÉDULA DEL PACIENTE
        self.text1.focus_set()

        self.array = None
        self.label = ""
        self.proba = 0

        #   RUN LOOP
        self.root.mainloop()

    @staticmethod
    def _validate_cedula_keystroke(proposed_value):
        """Restringe en tiempo real el campo de cédula: solo dígitos y
        máximo 10 caracteres. La validación de mínimo (6 dígitos) se hace
        aparte, al presionar 'Predecir', porque no se puede exigir un
        mínimo mientras el usuario todavía está escribiendo."""
        return proposed_value == "" or (
            proposed_value.isdigit() and len(proposed_value) <= 10
        )

    @staticmethod
    def _entry_look():
        """Opciones de estilo para que los widgets Text (Resultado,
        Probabilidad) se vean como los ttk.Entry del tema 'flatly', ya que
        Text no hereda el estilo ttk automáticamente."""
        return {
            "borderwidth": 0,
            "relief": "flat",
            "highlightthickness": 1,
            "highlightbackground": "#ced4da",
            "highlightcolor": "#75b798",
            "font": ("Segoe UI", 10),
            "padx": 6,
            "pady": 4,
        }

    def _setup_style(self):
        """Ajusta fuentes sobre el tema 'flatly' de ttkbootstrap para dar
        jerarquía visual al encabezado y a las etiquetas de campos."""
        style = ttk.Style()
        style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"))
        style.configure("Field.TLabel", font=("Segoe UI", 10, "bold"))
        style.configure("TLabelframe.Label", font=("Segoe UI", 10, "bold"))
        style.configure("TButton", font=("Segoe UI", 10), padding=6)

    #   METHODS
    def load_img_file(self):
        """Abre el explorador de archivos, lee la imagen (DICOM o JPG/PNG)
        seleccionada y la muestra en el panel izquierdo."""
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
                self.array, img2show = read_dicom_file(filepath)
            else:
                self.array, img2show = read_jpg_file(filepath)
            self.img1 = img2show.resize((250, 250), Image.LANCZOS)
            self.img1 = ImageTk.PhotoImage(self.img1)
            self.text_img1.image_create(END, image=self.img1)
            self.button1["state"] = "normal"

    def run_model(self):
        """Valida la cédula del paciente y, si es válida, lanza la
        predicción en un hilo aparte (para no congelar la ventana) mientras
        se muestra una barra de progreso indeterminada.

        La cédula debe tener entre 6 y 10 dígitos numéricos, para que los
        reportes generados después siempre queden identificados con un
        valor razonable.
        """
        cedula = self.text1.get().strip()
        if not CEDULA_PATTERN.fullmatch(cedula):
            showwarning(
                title="Cédula inválida",
                message=(
                    "Ingrese una cédula válida (solo números, entre 6 y "
                    "10 dígitos) antes de predecir."
                ),
            )
            self.text1.focus_set()
            return

        self._set_processing(True)
        threading.Thread(target=self._predict_in_background, daemon=True).start()

    def _predict_in_background(self):
        """Corre predict() fuera del hilo principal para no congelar la
        interfaz, y devuelve el resultado (o el error) al hilo principal
        con root.after(), que es la forma segura de tocar widgets de
        Tkinter desde otro hilo."""
        try:
            result = predict(self.array)
        except Exception as exc:  # noqa: BLE001 - se muestra al usuario
            self.root.after(0, self._on_prediction_error, exc)
            return
        self.root.after(0, self._on_prediction_ready, result)

    def _on_prediction_ready(self, result):
        """Actualiza la interfaz con el resultado de la predicción, ya de
        vuelta en el hilo principal."""
        self.label, self.proba, self.heatmap = result
        self.img2 = Image.fromarray(self.heatmap)
        self.img2 = self.img2.resize((250, 250), Image.LANCZOS)
        self.img2 = ImageTk.PhotoImage(self.img2)
        self.text_img2.image_create(END, image=self.img2)
        self.text2.insert(END, self.label)
        self.text3.insert(END, f"{self.proba:.2f}" + "%")
        self._set_processing(False)

    def _on_prediction_error(self, exc):
        """Muestra el error al usuario si la predicción falla en el hilo
        de fondo, y deja la interfaz lista para reintentar."""
        self._set_processing(False)
        showerror(title="Error al predecir", message=str(exc))

    def _set_processing(self, processing):
        """Muestra/oculta la barra de progreso y habilita/deshabilita el
        botón 'Predecir' mientras se está calculando un resultado."""
        if processing:
            self.button1["state"] = "disabled"
            self.progress.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(8, 0))
            self.progress.start(12)
        else:
            self.progress.stop()
            self.progress.grid_remove()
            self.button1["state"] = "normal"

    def _cedula_paciente(self):
        """Retorna la cédula ingresada por el usuario, o 'sin_cedula' si el
        campo quedó vacío, para no dejar los reportes sin identificar."""
        cedula = self.text1.get().strip()
        return cedula if cedula else "sin_cedula"

    def save_results_csv(self):
        """Agrega el resultado actual (cédula, clase, probabilidad) al
        historial en CSV, dentro de la carpeta reportes/."""
        os.makedirs(REPORTS_DIR, exist_ok=True)
        historial_path = os.path.join(REPORTS_DIR, "historial.csv")
        with open(historial_path, "a") as csvfile:
            w = csv.writer(csvfile, delimiter="-")
            w.writerow([self.text1.get(), self.label, f"{self.proba:.2f}" + "%"])
            showinfo(title="Guardar", message="Los datos se guardaron con éxito.")

    def create_pdf(self):
        """Genera un PDF con lo mostrado en pantalla, dentro de la carpeta
        reportes/, nombrado con la cédula del paciente para no sobreescribir
        reportes de pacientes distintos."""
        cedula = self._cedula_paciente()
        os.makedirs(REPORTS_DIR, exist_ok=True)
        img = capture_window(self.root)
        jpg_path = os.path.join(REPORTS_DIR, f"Reporte_{cedula}.jpg")
        img.save(jpg_path)
        img = img.convert("RGB")
        pdf_path = os.path.join(REPORTS_DIR, f"Reporte_{cedula}.pdf")
        img.save(pdf_path)
        showinfo(title="PDF", message="El PDF fue generado con éxito.")

    def delete(self):
        """Limpia los campos de la interfaz, previa confirmación del usuario."""
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


def main():
    App()
    return 0


if __name__ == "__main__":
    main()
