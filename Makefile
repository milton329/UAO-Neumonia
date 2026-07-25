# =============================================================================
# Makefile - UAO Neumonía
# Atajos para las tareas comunes de desarrollo. Todo se ejecuta a través de
# "uv run"/"uv sync" para que siempre corra dentro del entorno virtual del
# proyecto (.venv), sin depender de que esté activado manualmente.
#
# Uso: make <target>        (ej: make install, make test)
#      make                 (sin target -> muestra esta ayuda)
# =============================================================================

IMAGE_NAME := uao-neumonia

.DEFAULT_GOAL := help

.PHONY: help install detector test lint format clean docker-build docker-run

help: ## Muestra esta ayuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Instala/sincroniza dependencias del proyecto (incluye las de desarrollo)
	uv sync

detector: ## Lanza la interfaz gráfica de la herramienta de detección de neumonía
	uv run python main.py

test: ## Corre la suite de pruebas unitarias (pytest)
	uv run pytest

lint: ## Revisa el estilo del código con ruff, sin modificar archivos
	uv run ruff check .

format: ## Formatea el código automáticamente con ruff
	uv run ruff format .

clean: ## Elimina cachés y artefactos generados localmente
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +

docker-build: ## Construye la imagen Docker de la aplicación
	docker build -t $(IMAGE_NAME) .

docker-run: ## Corre el contenedor, montando models/ (el modelo no viaja en la imagen)
	docker run --rm -v "$(CURDIR)/models:/app/models" $(IMAGE_NAME)
