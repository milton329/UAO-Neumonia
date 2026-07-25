# syntax=docker/dockerfile:1
# =============================================================================
# Dockerfile - UAO Neumonía
# Build multi-stage: una etapa "builder" resuelve el entorno con uv y otra,
# final y liviana, solo contiene lo necesario para ejecutar la aplicación.
# Así la imagen que se distribuye no carga con uv, la caché de paquetes ni
# herramientas de build que solo hacen falta durante la instalación.
# =============================================================================

# -----------------------------------------------------------------------------
# Etapa 1: builder
# Resuelve e instala las dependencias del proyecto en un entorno virtual
# aislado (.venv) usando uv + el lockfile, para builds reproducibles.
# -----------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS builder

# Se copia el binario de uv (pineado a una versión) desde la imagen oficial
# de Astral, en vez de instalarlo con pip (más rápido y no ensucia el venv).
COPY --from=ghcr.io/astral-sh/uv:0.11.30 /uv /uvx /bin/

# UV_COMPILE_BYTECODE: precompila los .pyc durante "uv sync" para que el
#   contenedor arranque más rápido (el costo se paga una vez, en el build).
# UV_LINK_MODE=copy: copia los paquetes en vez de hacer hardlink desde la
#   caché; evita fallos al cruzar capas/filesystems distintos de Docker.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Paso 1 de la instalación: solo se copian los manifiestos de dependencias
# (todavía no el código) y se resuelve el entorno. Mientras pyproject.toml y
# uv.lock no cambien, Docker reutiliza esta capa de caché aunque el código sí
# cambie, evitando reinstalar todo en cada build.
# El cache mount conserva la caché de descargas de uv entre builds locales.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

# Paso 2: ahora sí se copia el código fuente y se completa la instalación
# (incluye el propio proyecto como paquete editable dentro del venv).
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# -----------------------------------------------------------------------------
# Etapa 2: imagen final de ejecución
# No incluye uv, la caché de pip/uv ni archivos intermedios del build; solo
# el entorno virtual ya resuelto y el código de la aplicación.
# -----------------------------------------------------------------------------
FROM python:3.12-slim-bookworm

# Dependencias de sistema necesarias en TIEMPO DE EJECUCIÓN (no de build):
#   - python3-tk: Tkinter no viene incluido en la imagen "slim"
#   - libgl1 / libglib2.0-0: librerías nativas que necesita opencv-python
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        python3-tk \
        libgl1 \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Usuario sin privilegios para no correr la aplicación como root dentro
# del contenedor (buena práctica de seguridad).
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Se copia SOLO el resultado de la etapa builder (código + venv resuelto),
# ya con el dueño correcto para el usuario no-root.
COPY --from=builder --chown=appuser:appuser /app /app

# Se antepone el venv al PATH para poder invocar "python" directamente,
# sin necesitar "uv run" ni activar el entorno manualmente.
ENV PATH="/app/.venv/bin:$PATH"

USER appuser

# El modelo entrenado (models/conv_MLP_84.h5) NO se copia a la imagen:
# es un binario pesado gestionado fuera de git/Docker (ver .gitignore).
# Debe montarse como volumen al ejecutar el contenedor, por ejemplo:
#   docker run -v ./models:/app/models uao-neumonia
VOLUME ["/app/models"]

CMD ["python", "main.py"]
