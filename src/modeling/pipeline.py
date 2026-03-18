from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.pipeline import load_processed_multi_year
from src.modeling.feature_engineering import build_modeling_dataframe
from src.modeling.baseline import baseline_naive_last_value
from src.modeling.validation import get_fixed_time_folds
from src.modeling.evaluation import evaluate_predictions
from src.modeling.modelos import (
    train_linear,
    train_ridge,
    predict,
    extract_linear_coefficients,
)


MODEL_FEATURE_COLS = [
    "lag_sv_ratio_1",
    "lag_sv_ratio_2",
    "state_sv_ratio_mean_1",
]

MODEL_TARGET_COL = "target_sv_ratio_next"


def run_model_diagnostics(
    base_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Ejecuta el flujo de evaluación para:
    - Estimador ingenuo
    - Regresión lineal
    - Ridge

    Devuelve:
    - results_df: resultados por fold
    - summary_df: promedio por modelo
    - predictions_df: predicciones fuera de muestra consolidadas
    - coef_linear: coeficientes del modelo lineal entrenado con todo el dataset útil
    - coef_ridge: coeficientes del modelo ridge entrenado con todo el dataset útil
    """
    from src.modeling.diagnostics import build_prediction_frame

    df_all = load_processed_multi_year(
        base_dir,
        (2009, 2012, 2015, 2018, 2021, 2024),
    )

    model_df = build_modeling_dataframe(df_all)
    model_df = model_df.dropna(
        subset=MODEL_FEATURE_COLS + [MODEL_TARGET_COL]
    ).copy()

    folds = get_fixed_time_folds()

    results: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []

    for fold in folds:
        train_years = fold["train_years"]
        test_year = fold["test_year"]

        train_df = model_df[model_df["year"].isin(train_years)].copy()
        test_df = model_df[model_df["year"] == test_year].copy()

        # --------------------------------------------------
        # ESTIMADOR INGENUO
        # --------------------------------------------------
        pred_df = baseline_naive_last_value(
            test_df,
            target_col=MODEL_TARGET_COL,
            lag_col="lag_sv_ratio_1",
        )

        metrics = evaluate_predictions(pred_df)
        metrics["model"] = "naive_last_value"
        metrics["test_year"] = test_year
        results.append(metrics)

        prediction_frames.append(
            build_prediction_frame(
                pred_df=pred_df,
                model_name="naive_last_value",
                target_col=MODEL_TARGET_COL,
                test_year=test_year,
            )
        )

        # --------------------------------------------------
        # REGRESIÓN LINEAL
        # --------------------------------------------------
        linear_model = train_linear(
            train_df,
            MODEL_TARGET_COL,
            features=MODEL_FEATURE_COLS,
        )

        pred_df = predict(
            linear_model,
            test_df,
            features=MODEL_FEATURE_COLS,
        )
        pred_df["y_true"] = pred_df[MODEL_TARGET_COL]

        metrics = evaluate_predictions(pred_df)
        metrics["model"] = "linear_regression"
        metrics["test_year"] = test_year
        results.append(metrics)

        prediction_frames.append(
            build_prediction_frame(
                pred_df=pred_df,
                model_name="linear_regression",
                target_col=MODEL_TARGET_COL,
                test_year=test_year,
            )
        )

        # --------------------------------------------------
        # RIDGE
        # --------------------------------------------------
        ridge_model = train_ridge(
            train_df,
            MODEL_TARGET_COL,
            alpha=1.0,
            features=MODEL_FEATURE_COLS,
        )

        pred_df = predict(
            ridge_model,
            test_df,
            features=MODEL_FEATURE_COLS,
        )
        pred_df["y_true"] = pred_df[MODEL_TARGET_COL]

        metrics = evaluate_predictions(pred_df)
        metrics["model"] = "ridge"
        metrics["test_year"] = test_year
        results.append(metrics)

        prediction_frames.append(
            build_prediction_frame(
                pred_df=pred_df,
                model_name="ridge",
                target_col=MODEL_TARGET_COL,
                test_year=test_year,
            )
        )

    results_df = pd.DataFrame(results)

    summary_df = (
        results_df.groupby("model", as_index=False)[["mae", "rmse", "n"]]
        .mean()
        .sort_values("mae")
        .reset_index(drop=True)
    )

    predictions_df = pd.concat(prediction_frames, ignore_index=True)

    # --------------------------------------------------
    # COEFICIENTES FINALES
    # Entrenados con todo el dataset útil
    # --------------------------------------------------
    linear_model = train_linear(
        model_df,
        MODEL_TARGET_COL,
        features=MODEL_FEATURE_COLS,
    )
    ridge_model = train_ridge(
        model_df,
        MODEL_TARGET_COL,
        alpha=1.0,
        features=MODEL_FEATURE_COLS,
    )

    coef_linear = extract_linear_coefficients(
        linear_model,
        MODEL_FEATURE_COLS,
    )
    coef_ridge = extract_linear_coefficients(
        ridge_model,
        MODEL_FEATURE_COLS,
    )

    return results_df, summary_df, predictions_df, coef_linear, coef_ridge


def build_forecast_2027_outputs(
    base_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construye la base de pronóstico 2027 usando 2024 como último año observado.

    Devuelve:
    - linear_out
    - ridge_out

    Cada DataFrame contiene:
    CVEGEO, municipio, state_code, LN, y_pred, model, forecast_year
    """
    df_all = load_processed_multi_year(
        base_dir,
        (2009, 2012, 2015, 2018, 2021, 2024),
    )

    model_df = build_modeling_dataframe(df_all)

    train_df = model_df.dropna(
        subset=MODEL_FEATURE_COLS + [MODEL_TARGET_COL]
    ).copy()

    # Para pronóstico usamos 2024 como base
    forecast_df = model_df[model_df["year"] == 2024].dropna(
        subset=MODEL_FEATURE_COLS
    ).copy()

    linear_model = train_linear(
        train_df,
        MODEL_TARGET_COL,
        features=MODEL_FEATURE_COLS,
    )

    ridge_model = train_ridge(
        train_df,
        MODEL_TARGET_COL,
        alpha=1.0,
        features=MODEL_FEATURE_COLS,
    )

    linear_pred = predict(
        linear_model,
        forecast_df,
        features=MODEL_FEATURE_COLS,
    ).copy()

    ridge_pred = predict(
        ridge_model,
        forecast_df,
        features=MODEL_FEATURE_COLS,
    ).copy()

    linear_out = linear_pred[
        ["CVEGEO", "municipio", "state_code", "LN", "y_pred"]
    ].copy()
    ridge_out = ridge_pred[
        ["CVEGEO", "municipio", "state_code", "LN", "y_pred"]
    ].copy()

    linear_out["model"] = "linear_regression"
    linear_out["forecast_year"] = 2027

    ridge_out["model"] = "ridge"
    ridge_out["forecast_year"] = 2027

    return linear_out, ridge_out