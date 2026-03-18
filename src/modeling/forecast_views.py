from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt


def add_prediction_interval_from_oos_error(
    forecast_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    model_name: str,
    z_value: float = 1.96,
) -> pd.DataFrame:
    """
    Construye un intervalo de predicción aproximado usando la desviación
    estándar del error fuera de muestra del modelo.

    Parámetros
    ----------
    forecast_df : DataFrame
        Debe incluir al menos la columna 'y_pred'.
    predictions_df : DataFrame
        Debe incluir columnas 'model' y 'error'.
    model_name : str
        Nombre interno del modelo, por ejemplo:
        - 'linear_regression'
        - 'ridge'
    z_value : float
        Multiplicador para la banda. 1.96 ~ banda aproximada del 95%.

    Devuelve
    --------
    DataFrame con columnas adicionales:
    - pi_lower
    - pi_upper
    """
    out = forecast_df.copy()

    if "y_pred" not in out.columns:
        raise ValueError("forecast_df debe contener la columna 'y_pred'.")

    error_std = (
        predictions_df.loc[predictions_df["model"] == model_name, "error"]
        .dropna()
        .std()
    )

    if pd.isna(error_std):
        raise ValueError(
            f"No fue posible estimar la desviación estándar del error para el modelo {model_name!r}."
        )

    out["pi_lower"] = (out["y_pred"] - z_value * error_std).clip(lower=0)
    out["pi_upper"] = (out["y_pred"] + z_value * error_std).clip(upper=1)

    return out


def plot_forecast_ranked(
    forecast_df: pd.DataFrame,
    model_display_name: str,
    figsize: tuple[int, int] = (10, 4),
):
    """
    Grafica el pronóstico municipal 2027 ordenando municipios por valor predicho.

    Requiere columnas:
    - y_pred
    - pi_lower
    - pi_upper
    """
    required_cols = {"y_pred", "pi_lower", "pi_upper"}
    missing = required_cols - set(forecast_df.columns)

    if missing:
        raise ValueError(
            f"forecast_df no contiene las columnas requeridas: {sorted(missing)}"
        )

    df = forecast_df.copy().sort_values("y_pred").reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        df["rank"],
        df["y_pred"],
        linewidth=1.8,
        label="Estimación puntual",
    )

    ax.fill_between(
        df["rank"],
        df["pi_lower"],
        df["pi_upper"],
        alpha=0.25,
        label="Intervalo de predicción aprox.",
    )

    ax.set_title(f"Pronóstico municipal 2027 - {model_display_name}")
    ax.set_xlabel("Municipios ordenados por predicción")
    ax.set_ylabel("Proporción de participación esperada")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    return fig, ax