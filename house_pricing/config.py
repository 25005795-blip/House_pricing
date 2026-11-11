from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# MLflow: tracking local en SQLite y artefactos en mlruns/, ambos en la raíz del proyecto
MLFLOW_TRACKING_URI = f"sqlite:///{(PROJ_ROOT / 'mlflow.db').as_posix()}"
MLFLOW_ARTIFACTS_DIR = PROJ_ROOT / "mlruns"
MLFLOW_EXPERIMENT_TRAINING = "house_pricing_entrenamiento"
MLFLOW_EXPERIMENT_INFERENCE = "house_pricing_inferencia"
REGISTERED_MODEL_NAME = "house_pricing_model"
CHAMPION_ALIAS = "champion"
CHALLENGER_ALIASES = ["challenger_1", "challenger_2"]

# MLflow serializa los modelos sklearn con skops, que solo carga tipos explícitamente confiables.
# Los pipelines del proyecto usan transformadores de feature-engine, árboles y el optimizador Adam del MLP;
# se declaran confiables porque los modelos se entrenan localmente en este proyecto.
SKOPS_TRUSTED_TYPES = [
    "feature_engine.creation.relative_features.RelativeFeatures",
    "feature_engine.encoding.ordinal.OrdinalEncoder",
    "feature_engine.imputation.categorical.CategoricalImputer",
    "feature_engine.imputation.mean_median.MeanMedianImputer",
    "feature_engine.outliers.winsorizer.Winsorizer",
    "feature_engine.transformation.yeojohnson.YeoJohnsonTransformer",
    "feature_engine.wrappers.wrappers.SklearnTransformerWrapper",
    "sklearn.neural_network._stochastic_optimizers.AdamOptimizer",
    "sklearn.tree._tree.Tree",
]

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
