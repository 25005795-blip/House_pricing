"""Utilidades de MLflow compartidas por los notebooks de entrenamiento e inferencia."""

import os
import warnings

import mlflow
from mlflow.entities import Experiment

from house_pricing.config import MLFLOW_ARTIFACTS_DIR, MLFLOW_TRACKING_URI


def _silenciar_ruido_mlflow() -> None:
    """Oculta mensajes informativos de MLflow que no aportan en los notebooks."""
    # Barras de progreso al validar el input_example de cada modelo registrado
    os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"
    # Al registrar los datasets crudos: sus columnas enteras ya se declaran como float en la firma del modelo
    warnings.filterwarnings("ignore", message=r"Hint: Inferred schema contains integer column")
    warnings.filterwarnings("ignore", message=r"The specified dataset source can be interpreted in multiple ways")


def configurar_mlflow(nombre_experimento: str) -> Experiment:
    """Apunta MLflow al tracking local del proyecto y activa el experimento indicado.

    Si el experimento no existe se crea con sus artefactos en `mlruns/`, de modo que
    la ubicación no dependa del directorio desde el que se ejecute el notebook.
    """
    _silenciar_ruido_mlflow()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(nombre_experimento) is None:
        mlflow.create_experiment(
            nombre_experimento,
            artifact_location=(MLFLOW_ARTIFACTS_DIR / nombre_experimento).as_uri(),
        )
    return mlflow.set_experiment(nombre_experimento)
