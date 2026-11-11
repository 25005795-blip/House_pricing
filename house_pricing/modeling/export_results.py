"""Reentrena el torneo de modelos del notebook 3.0 y exporta sus resultados a JSON.

Las rejillas, la partición y el esquema de validación replican el notebook
3.0-mo-entrenamiento-modelos; si cambian allí, deben actualizarse aquí.
"""

import json
from pathlib import Path

import joblib
from loguru import logger
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import make_scorer, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import typer
import yaml

from house_pricing.config import MODELS_DIR, PROJ_ROOT, RAW_DATA_DIR, REPORTS_DIR

app = typer.Typer()

TEST_SIZE = 0.2
RANDOM_STATE = 42
N_FOLDS = 10
REFERENCE_YEAR = "YrSold"

# Criterios de coherencia: un modelo que incumpla alguno se descarta
MAX_PCT_OUT_OF_RANGE = 0.05
MAX_KS = 0.15
MAX_NRMSE_VAL = 0.06


def rmse_real(y_true_log, y_pred_log):
    """RMSE en dólares a partir del target y la predicción en escala log1p."""
    return root_mean_squared_error(np.expm1(y_true_log), np.expm1(y_pred_log))


def build_models():
    """Modelos y rejillas de hiperparámetros del torneo."""
    return {
        "Ridge": (Ridge(), {"model__alpha": [0.1, 1, 10, 30, 100]}),
        "Lasso": (Lasso(max_iter=50_000), {"model__alpha": [0.0001, 0.0005, 0.001, 0.005]}),
        "ElasticNet": (
            ElasticNet(max_iter=50_000),
            {"model__alpha": [0.0005, 0.001, 0.005], "model__l1_ratio": [0.2, 0.5, 0.8]},
        ),
        "RandomForest": (
            RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE),
            {"model__max_depth": [None, 15], "model__min_samples_leaf": [1, 3], "model__max_features": [0.33, 1.0]},
        ),
        "GradientBoosting": (
            GradientBoostingRegressor(subsample=0.8, random_state=RANDOM_STATE),
            {"model__n_estimators": [300, 600], "model__learning_rate": [0.03, 0.06], "model__max_depth": [2, 3]},
        ),
        "MLP": (
            TransformedTargetRegressor(
                regressor=MLPRegressor(max_iter=3000, early_stopping=True, random_state=RANDOM_STATE),
                transformer=StandardScaler(),
            ),
            {"model__regressor__hidden_layer_sizes": [(64,), (64, 32), (128, 64)], "model__regressor__alpha": [0.001, 0.1]},
        ),
    }


def short_params(params):
    """Quita los prefijos del pipeline de los nombres de hiperparámetros."""
    return {key.split("__")[-1]: (str(value) if isinstance(value, tuple) else value) for key, value in params.items()}


def coherence_checks(pred_test, train_target, nrmse_val):
    """Evalúa los criterios de coherencia sobre las predicciones de test."""
    target_min, target_max = train_target.min(), train_target.max()
    invalid = int((~np.isfinite(pred_test)).sum() + (pred_test <= 0).sum())
    pct_out = float(((pred_test < target_min) | (pred_test > target_max)).mean())
    ks = float(ks_2samp(pred_test, train_target).statistic)
    checks = [
        {"criterio": "Valores inválidos", "valor": invalid, "umbral": 0, "cumple": invalid == 0},
        {"criterio": "% fuera del rango de train", "valor": pct_out, "umbral": MAX_PCT_OUT_OF_RANGE,
         "cumple": pct_out <= MAX_PCT_OUT_OF_RANGE},
        {"criterio": "KS vs SalePrice de train", "valor": ks, "umbral": MAX_KS, "cumple": ks <= MAX_KS},
        {"criterio": "NRMSE de validación", "valor": nrmse_val, "umbral": MAX_NRMSE_VAL,
         "cumple": nrmse_val <= MAX_NRMSE_VAL},
    ]
    return checks


@app.command()
def main(
    config_path: Path = PROJ_ROOT / "pipeline_config.yaml",
    train_path: Path = RAW_DATA_DIR / "train.csv",
    test_path: Path = RAW_DATA_DIR / "test.csv",
    preprocessing_path: Path = MODELS_DIR / "preprocessing_pipeline.pkl",
    output_path: Path = REPORTS_DIR / "model_results.json",
):
    with open(config_path, encoding="utf-8") as f:
        pipeline_config = yaml.safe_load(f)
    target = pipeline_config["TARGET"]
    input_features = pipeline_config["MODEL_FEATURES"] + [REFERENCE_YEAR]

    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)
    X = train_raw[input_features]
    y = np.log1p(train_raw[target])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    X_test = test_raw[input_features]

    preprocessing = clone(joblib.load(preprocessing_path))
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scorer = make_scorer(rmse_real, greater_is_better=False)
    train_target = train_raw[target].to_numpy()
    target_range = float(np.expm1(y_train).max() - np.expm1(y_train).min())

    models = {}
    for name, (model, grid) in tqdm(build_models().items(), desc="Torneo"):
        logger.info(f"Entrenando {name}...")
        search = GridSearchCV(
            Pipeline(steps=[("preprocessing", clone(preprocessing)), ("model", model)]),
            param_grid=grid, scoring=scorer, cv=cv, n_jobs=-1, return_train_score=True, refit=True,
        )
        search.fit(X_train, y_train)

        results = search.cv_results_
        best = search.best_index_
        pred_val = np.expm1(search.predict(X_val))
        pred_test = np.expm1(search.predict(X_test))
        rmse_train = float(-results["mean_train_score"][best])
        rmse_val = float(root_mean_squared_error(np.expm1(y_val), pred_val))
        nrmse_val = rmse_val / target_range
        checks = coherence_checks(pred_test, train_target, nrmse_val)

        models[name] = {
            "best_params": short_params(search.best_params_),
            "metrics": {
                "rmse_train": rmse_train,
                "rmse_cv": float(-search.best_score_),
                "rmse_cv_std": float(results["std_test_score"][best]),
                "rmse_val": rmse_val,
                "nrmse_val": nrmse_val,
                "brecha_train_val": (rmse_val - rmse_train) / rmse_train,
            },
            "fold_rmse": [float(-results[f"split{i}_test_score"][best]) for i in range(N_FOLDS)],
            "configs": [
                {
                    "params": short_params(params),
                    "rmse_cv": float(-results["mean_test_score"][i]),
                    "rmse_cv_std": float(results["std_test_score"][i]),
                    "rmse_train": float(-results["mean_train_score"][i]),
                    "rank": int(results["rank_test_score"][i]),
                }
                for i, params in enumerate(results["params"])
            ],
            "coherence": checks,
            "descartado": not all(check["cumple"] for check in checks),
            "pred_val": [round(float(v), 0) for v in pred_val],
            "pred_test": [round(float(v), 0) for v in pred_test],
        }
        logger.info(f"{name}: RMSE val = {rmse_val:,.0f} | descartado = {models[name]['descartado']}")

    output = {
        "target": target,
        "n_folds": N_FOLDS,
        "target_range_train": target_range,
        "thresholds": {"max_pct_out_of_range": MAX_PCT_OUT_OF_RANGE, "max_ks": MAX_KS, "max_nrmse_val": MAX_NRMSE_VAL},
        "train_target": [float(v) for v in train_target],
        "y_val": [round(float(v), 0) for v in np.expm1(y_val)],
        "models": models,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)
    logger.success(f"Resultados guardados en {output_path}")


if __name__ == "__main__":
    app()
