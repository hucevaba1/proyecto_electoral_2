from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import plotly.express as px


def build_prediction_frame(
    pred_df: pd.DataFrame,
    model_name: str,
    target_col: str,
    test_year: int,
) -> pd.DataFrame:
    """
    Estandariza el output de predicción para diagnóstico.

    Devuelve columnas:
        model, test_year, CVEGEO, municipio, state_code, year,
        y_true, y_pred, error, abs_error
    """
    work = pred_df.copy()

    if "y_true" not in work.columns:
        work["y_true"] = work[target_col]

    work["error"] = work["y_pred"] - work["y_true"]
    work["abs_error"] = work["error"].abs()

    cols = [
        "CVEGEO",
        "municipio",
        "state_code",
        "year",
        "y_true",
        "y_pred",
        "error",
        "abs_error",
    ]

    out = work[cols].copy()
    out["model"] = model_name
    out["test_year"] = test_year

    ordered_cols = [
        "model",
        "test_year",
        "CVEGEO",
        "municipio",
        "state_code",
        "year",
        "y_true",
        "y_pred",
        "error",
        "abs_error",
    ]
    return out[ordered_cols]


def plot_predicted_vs_real(
    predictions_df: pd.DataFrame,
    model_name: str,
    test_year: int | None = None,
    figsize: tuple[int, int] = (8, 8),
):
    """
    Scatter plot de predicho vs real.
    """
    df = predictions_df[predictions_df["model"] == model_name].copy()

    if test_year is not None:
        df = df[df["test_year"] == test_year].copy()

    if df.empty:
        raise ValueError("No hay datos para el modelo/año solicitado.")

    x = df["y_true"].to_numpy()
    y = df["y_pred"].to_numpy()

    min_val = float(min(x.min(), y.min()))
    max_val = float(max(x.max(), y.max()))

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, y, alpha=0.35)

    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
        linewidth=1.5,
    )

    title = f"Predicho vs real - {model_name}"
    if test_year is not None:
        title += f" ({test_year})"

    ax.set_title(title)
    ax.set_xlabel("Valor real")
    ax.set_ylabel("Valor predicho")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, ax


def plot_error_distribution(
    predictions_df: pd.DataFrame,
    model_name: str,
    test_year: int | None = None,
    bins: int = 40,
    figsize: tuple[int, int] = (9, 5),
):
    """
    Histograma de errores: y_pred - y_true.
    """
    df = predictions_df[predictions_df["model"] == model_name].copy()

    if test_year is not None:
        df = df[df["test_year"] == test_year].copy()

    if df.empty:
        raise ValueError("No hay datos para el modelo/año solicitado.")

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(df["error"].dropna(), bins=bins)

    ax.axvline(0, linestyle="--", linewidth=1.5)

    title = f"Distribución de errores - {model_name}"
    if test_year is not None:
        title += f" ({test_year})"

    ax.set_title(title)
    ax.set_xlabel("Error = predicho - real")
    ax.set_ylabel("Frecuencia")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, ax


def load_municipal_geometries_for_diagnostics(
    shp_path: str | Path,
) -> gpd.GeoDataFrame:
    """
    Carga geometrías municipales y normaliza CVEGEO.
    """
    gdf = gpd.read_file(shp_path)

    gdf["CVE_ENT"] = gdf["CVE_ENT"].astype(str).str.zfill(2)
    gdf["CVE_MUN"] = gdf["CVE_MUN"].astype(str).str.zfill(3)
    gdf["CVEGEO"] = gdf["CVE_ENT"] + gdf["CVE_MUN"]

    return gdf


def build_error_map_frame(
    predictions_df: pd.DataFrame,
    gdf_municipios: gpd.GeoDataFrame,
    model_name: str,
    test_year: int,
) -> gpd.GeoDataFrame:
    """
    Une predicciones con geometrías para mapear el error municipal.
    """
    pred = predictions_df[
        (predictions_df["model"] == model_name)
        & (predictions_df["test_year"] == test_year)
    ].copy()

    if pred.empty:
        raise ValueError("No hay predicciones para el modelo/año solicitado.")

    pred["CVEGEO"] = pred["CVEGEO"].astype(str)

    gdf = gdf_municipios.copy()
    gdf["CVEGEO"] = gdf["CVEGEO"].astype(str)

    merged = gdf.merge(
        pred[["CVEGEO", "municipio", "y_true", "y_pred", "error", "abs_error"]],
        on="CVEGEO",
        how="left",
        validate="1:1",
    )

    return merged


def build_error_choropleth_plotly(
    gdf_error: gpd.GeoDataFrame,
    color_col: str = "error",
    title: str | None = None,
    width: int = 1000,
    height: int = 700,
):
    """
    Mapa coroplético interactivo del error municipal.
    color_col:
        - "error" para error con signo
        - "abs_error" para magnitud del error
    """
    valid_cols = {"error", "abs_error"}
    if color_col not in valid_cols:
        raise ValueError(f"color_col debe ser una de {valid_cols}")

    plot_gdf = gdf_error.copy()

    if plot_gdf.crs is None:
        raise ValueError("El GeoDataFrame no tiene CRS definido.")

    plot_gdf = plot_gdf[plot_gdf.geometry.notna()].copy()
    plot_gdf = plot_gdf.to_crs(epsg=4326)
    plot_gdf = plot_gdf.explode(index_parts=False).reset_index(drop=True)

    plot_gdf["feature_id"] = plot_gdf["CVEGEO"].astype(str)
    plot_gdf["municipio_label"] = plot_gdf["municipio"].fillna(plot_gdf["NOMGEO"])

    geojson_dict = json.loads(
        plot_gdf[["feature_id", "geometry"]].to_json()
    )

    for feature, fid in zip(geojson_dict["features"], plot_gdf["feature_id"]):
        feature["id"] = fid

    if color_col == "error":
        color_scale = "RdBu"
        range_bound = float(np.nanmax(np.abs(plot_gdf["error"])))
        range_color = (-range_bound, range_bound)
        if title is None:
            title = "Error municipal del modelo"
        hovertemplate = (
            "<b>%{hovertext}</b><br>"
            "Real: %{customdata[0]:.3f}<br>"
            "Predicho: %{customdata[1]:.3f}<br>"
            "Error: %{customdata[2]:.3f}<br>"
            "Error absoluto: %{customdata[3]:.3f}"
            "<extra></extra>"
        )
    else:
        color_scale = "OrRd"
        range_color = None
        if title is None:
            title = "Error absoluto municipal del modelo"
        hovertemplate = (
            "<b>%{hovertext}</b><br>"
            "Real: %{customdata[0]:.3f}<br>"
            "Predicho: %{customdata[1]:.3f}<br>"
            "Error: %{customdata[2]:.3f}<br>"
            "Error absoluto: %{customdata[3]:.3f}"
            "<extra></extra>"
        )

    fig = px.choropleth(
        plot_gdf,
        geojson=geojson_dict,
        locations="feature_id",
        featureidkey="id",
        color=color_col,
        color_continuous_scale=color_scale,
        range_color=range_color,
        hover_name="municipio_label",
        custom_data=["y_true", "y_pred", "error", "abs_error"],
    )

    fig.update_geos(
        fitbounds="geojson",
        visible=False,
    )

    fig.update_traces(
        marker_line_color="black",
        marker_line_width=0.4,
        hovertemplate=hovertemplate,
    )

    fig.update_layout(
        title=dict(
            text=title,
            x=0.02,
            xanchor="left",
        ),
        margin=dict(l=0, r=0, t=70, b=0),
        width=width,
        height=height,
    )

    return fig