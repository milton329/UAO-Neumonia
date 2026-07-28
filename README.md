# UAO-Neumonía

Herramienta de escritorio para el apoyo al diagnóstico rápido de neumonía a partir de radiografías de tórax, usando Deep Learning y Grad-CAM para explicar visualmente cada predicción.

![Python](https://img.shields.io/badge/python-3.12%2B-blue?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/managed%20with-uv-DE5FE9)
![Ruff](https://img.shields.io/badge/lint-ruff-D7FF64?logo=ruff&logoColor=black)
![pytest](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)
![Docker](https://img.shields.io/badge/container-docker-2496ED?logo=docker&logoColor=white)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

> **Nota:** este es un proyecto académico (Especialización en Inteligencia Artificial, UAO). Es una herramienta de **apoyo**, no un dispositivo médico validado clínicamente; ninguna predicción generada aquí debe usarse como diagnóstico definitivo sin la revisión de un profesional de la salud.

---

## Tabla de contenido

- [¿Qué hace?](#qué-hace)
- [Arquitectura](#arquitectura)
- [Acerca del modelo](#acerca-del-modelo)
- [Acerca de Grad-CAM](#acerca-de-grad-cam)
- [Instalación](#instalación)
- [Ejecución](#ejecución)
- [Uso de la interfaz gráfica](#uso-de-la-interfaz-gráfica)
- [Exportación a PDF](#exportación-a-pdf)
- [Pruebas unitarias](#pruebas-unitarias)
- [Makefile](#makefile)
- [Docker](#docker)
- [Licencia](#licencia)
- [Autores](#autores)

---

## ¿Qué hace?

Clasifica una radiografía de tórax (DICOM, JPG o PNG) en una de tres categorías:

1. Neumonía Bacteriana
2. Neumonía Viral
3. Sin Neumonía

Además de la clase y su probabilidad, genera un **mapa de calor Grad-CAM** superpuesto sobre la imagen original, resaltando las regiones que más influyeron en la predicción del modelo. Desde la interfaz se puede guardar el resultado en un historial CSV y exportar un reporte en PDF.

## Arquitectura

El proyecto es modular: la interfaz gráfica no contiene lógica de inferencia, solo orquesta llamadas a módulos independientes y con responsabilidad única.

```mermaid
flowchart TD
    main[main.py] --> gui["presentation/app.py<br/>Interfaz Tkinter"]
    gui -->|Cargar Imagen| read[infrastructure/image_reader.py]
    gui -->|Predecir| svc[application/prediction_service.py]
    svc --> prep[infrastructure/image_processor.py]
    svc --> model[infrastructure/model_loader.py]
    svc --> gc[infrastructure/grad_cam_generator.py]
    gc --> prep
    gc --> model
    model -.lee.-> h5[("models/conv_MLP_84.h5")]
    gui -->|Guardar| csv[infrastructure/csv_history.py]
    gui -->|PDF| pdf[infrastructure/pdf_exporter.py]
```

| Módulo | Responsabilidad |
|---|---|---|
| `main.py` | Punto de entrada de la aplicación. |
| `presentation/app.py` | Interfaz gráfica (Tkinter): carga de imágenes, botones, historial y exportación a PDF. No contiene lógica de inferencia. |
| `infrastructure/image_reader.py` | Lee un archivo DICOM o JPG/PNG y lo convierte a un array numpy listo para preprocesar. |
| `infrastructure/image_processor.py` | Preprocesa el array: resize a 512×512, escala de grises, ecualización CLAHE, normalización 0–1, formato de batch. |
| `infrastructure/model_loader.py` | Carga (una sola vez, con caché) el modelo entrenado desde `models/conv_MLP_84.h5`. |
| `infrastructure/grad_cam_generator.py` | Genera el mapa de calor Grad-CAM sobre la imagen, usando `tf.GradientTape`. |
| `application/prediction_service.py` | Orquesta preprocesamiento → predicción → Grad-CAM y retorna `PredictionResult` a la interfaz. |
| `infrastructure/csv_history.py` | Persiste los resultados en el archivo `historial.csv`. |
| `infrastructure/pdf_exporter.py` | Genera un reporte PDF con imagen, heatmap y diagnóstico (sin tkcap). |
| `config.py` | Constantes centralizadas del proyecto. |
| `domain/models.py` | Modelos de dominio (`PredictionResult`). |

### Árbol de archivos

```
UAO-Neumonia/
├── .github/
│   └── pull_request_template.md
├── models/
│   └── conv_MLP_84.h5        # no versionado en git (~117MB), ver Instalación
├── src/
│   └── uao_neumonia/
│       ├── __init__.py
│       ├── config.py                   # constantes centralizadas
│       ├── domain/
│       │   ├── __init__.py
│       │   └── models.py               # PredictionResult (dataclass)
│       ├── infrastructure/
│       │   ├── __init__.py
│       │   ├── model_loader.py         # carga del modelo
│       │   ├── image_reader.py         # lectura DICOM / JPG / PNG
│       │   ├── image_processor.py      # preprocesamiento
│       │   ├── grad_cam_generator.py   # Grad-CAM heatmap
│       │   ├── csv_history.py          # historial CSV
│       │   └── pdf_exporter.py         # exportación a PDF
│       ├── application/
│       │   ├── __init__.py
│       │   └── prediction_service.py   # orquestación
│       └── presentation/
│           ├── __init__.py
│           └── app.py                  # interfaz gráfica (Tkinter)
├── tests/
│   ├── conftest.py
│   ├── test_app.py
│   ├── test_config.py
│   ├── test_csv_history.py
│   ├── test_domain_models.py
│   ├── test_grad_cam_generator.py
│   ├── test_image_processor.py
│   ├── test_image_reader.py
│   ├── test_model_loader.py
│   ├── test_pdf_exporter.py
│   └── test_prediction_service.py
├── main.py                   # punto de entrada
├── Dockerfile
├── .dockerignore
├── Makefile
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
├── LICENSE
└── README.md
```

## Acerca del modelo

La red neuronal convolucional (CNN) está basada en el modelo propuesto por F. Pasa, V. Golkov, F. Pfeifer, D. Cremers & D. Pfeifer en su artículo *Efficient Deep Network Architectures for Fast Chest X-Ray Tuberculosis Screening and Visualization*.

Está compuesta por 5 bloques convolucionales, cada uno con 3 convoluciones (dos secuenciales y una conexión *skip* que evita el desvanecimiento del gradiente en profundidad), con 16, 32, 48, 64 y 80 filtros de 3×3 respectivamente.

Después de cada bloque hay una capa de max pooling, y tras el último bloque, una capa de Average Pooling seguida de tres capas fully-connected (1024, 1024 y 3 neuronas). Para regularizar se usan 3 capas de Dropout al 20%: dos en los bloques 4 y 5, y otra después de la primera capa densa.

## Acerca de Grad-CAM

Grad-CAM (*Gradient-weighted Class Activation Mapping*) resalta las regiones de una imagen más relevantes para la clasificación. Calcula el gradiente de la salida correspondiente a la clase predicha respecto a las neuronas de la última capa convolucional, obteniendo así la importancia de cada neurona en la decisión. Con esos pesos se hace una combinación lineal con el mapa de activaciones de esa capa, produciendo un heatmap que se superpone sobre la radiografía original: las zonas más "calientes" son las que más influyeron en la predicción.

## Instalación

Requisitos: **Python 3.12+** y [**uv**](https://docs.astral.sh/uv/getting-started/installation/) como gestor de paquetes y entornos.

```bash
git clone https://github.com/milton329/UAO-Neumonia.git
cd UAO-Neumonia

# Instala/sincroniza todas las dependencias (equivalente: make install)
uv sync
```

`uv sync` crea el entorno virtual (`.venv/`) y resuelve las dependencias exactas del `uv.lock`, sin necesidad de `pip install -r requirements.txt` ni de activar el entorno manualmente.

**El modelo entrenado no viaja en git** (pesa ~117MB). Copia `conv_MLP_84.h5` dentro de `models/` antes de ejecutar la app:

```
UAO-Neumonia/
└── models/
    └── conv_MLP_84.h5
```

Imágenes de radiografía de prueba (DICOM) disponibles en [este Drive](https://drive.google.com/drive/folders/1WOuL0wdVC6aojy8IfssHcqZ4Up14dy0g?usp=drive_link).

## Ejecución

```bash
make detector
# equivalente a: uv run python main.py
```

Ver [Makefile](#makefile) para el resto de comandos disponibles, o [Docker](#docker) para correrlo en contenedor.

## Uso de la interfaz gráfica

1. Ingresa la cédula del paciente en la caja de texto.
2. Presiona **Cargar Imagen** y selecciona un archivo DICOM, JPG o PNG.
3. Presiona **Predecir** y espera unos segundos hasta ver el resultado y el heatmap.
4. Presiona **Guardar** para agregar el resultado al historial (`historial.csv`).
5. Presiona **PDF** para exportar un reporte (`Reporte_<cédula>.pdf`).
6. Presiona **Borrar** para limpiar la interfaz y cargar una nueva imagen.

## Exportación a PDF

El botón **PDF** de la interfaz genera un reporte con la imagen original, el heatmap Grad-CAM, el diagnóstico y la probabilidad. Inicialmente se implementó con la librería **`tkcap`**, que capturaba la pantalla de la ventana y guardaba la imagen como PDF. Ese enfoque presentaba dos problemas:

1. **No funcionaba en macOS** — `tkcap` depende de `python-xlib` (X11), pero Tkinter en macOS usa Aqua Tk, no X11.
2. **Captura frágil** — tomaba un screenshot de la región de la pantalla, no de la ventana en sí, por lo que cualquier elemento superpuesto (notificaciones, otras ventanas) podía contaminar el reporte.

### Solución implementada

Se reemplazó `tkcap` por una composición programática con **Pillow** (que ya estaba en las dependencias del proyecto). El nuevo módulo `infrastructure/pdf_exporter.py`:

1. Toma los datos directamente de la memoria (arrays numpy de la imagen original y el heatmap), sin capturar pantalla.
2. Compone un collage de 1000×500 px con Pillow: imagen original a la izquierda, heatmap a la derecha, diagnóstico y probabilidad en la parte inferior.
3. Guarda el collage como PDF con `img.save(pdf_path, "PDF")`.

Esto eliminó la dependencia de `tkcap` y `python-xlib`, y la generación de PDF funciona correctamente en **macOS, Linux y Windows** sin ninguna dependencia adicional del sistema.

## Pruebas unitarias

```bash
make test
# equivalente a: uv run pytest
```

La suite (**105 pruebas**) cubre todos los módulos del paquete `uao_neumonia`: configuración, modelos de dominio, carga del modelo, lectura de imágenes, preprocesamiento, Grad-CAM, servicio de predicción, historial CSV, exportación PDF e interfaz gráfica. La mayoría usa mocks o un modelo Keras diminuto (fixture `tiny_cnn_model`), por lo que corren en segundos y **no dependen de tener el modelo real de 117MB descargado**. Para medir cobertura:

```bash
uv run pytest --cov
```

## Makefile

Todos los comandos corren dentro del entorno de `uv`, sin necesidad de activarlo manualmente:

| Comando | Qué hace |
|---|---|
| `make help` | Lista todos los targets disponibles (target por defecto). |
| `make install` | Instala/sincroniza las dependencias del proyecto (`uv sync`). |
| `make detector` | Lanza la interfaz gráfica de la aplicación. |
| `make test` | Corre la suite de pruebas unitarias (`pytest`). |
| `make lint` | Revisa el estilo del código con `ruff`, sin modificar archivos. |
| `make format` | Formatea el código automáticamente con `ruff`. |
| `make clean` | Elimina cachés y artefactos generados localmente (`__pycache__`, `.pytest_cache`, `.ruff_cache`). |
| `make docker-build` | Construye la imagen Docker de la aplicación. |
| `make docker-run` | Corre el contenedor, montando `models/` como volumen. |

## Docker

La imagen se construye en dos etapas (build con `uv`, runtime liviano) y **no incluye el modelo** — se monta como volumen en tiempo de ejecución:

```bash
make docker-build
make docker-run
# equivalente a:
docker build -t uao-neumonia .
docker run --rm -v "$(pwd)/models:/app/models" uao-neumonia
```

## Licencia

Este proyecto se distribuye bajo la licencia [MIT](LICENSE).

## Autores

Proyecto desarrollado como parte de la Especialización en Inteligencia Artificial de la Universidad Autónoma de Occidente (UAO).

- Jhanluy Bolívar
- Milton Jaramillo
- David Antonio Paredes Bravo
- Juan Diego Estupiñán

**Docente:** Jan Polanco Velasco

---

Basado en el proyecto original de Isabella Torres Revelo ([@isa-tr](https://github.com/isa-tr)) y Nicolas Diaz Salazar ([@nicolasdiazsalazar](https://github.com/nicolasdiazsalazar)).
