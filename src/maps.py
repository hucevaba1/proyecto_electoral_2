from __future__ import annotations

from pathlib import Path
import json

import geopandas as gpd
import pandas as pd
import plotly.express as px
import numpy as np

def load_municipal_geometries(shp_path: str | Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(shp_path)

    gdf["CVE_ENT"] = gdf["CVE_ENT"].astype(str).str.zfill(2)
    gdf["CVE_MUN"] = gdf["CVE_MUN"].astype(str).str.zfill(3)
    gdf["CVEGEO"] = gdf["CVE_ENT"] + gdf["CVE_MUN"]

    return gdf


def build_municipal_metrics(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    work["EDOCVE"] = work["EDOCVE"].astype("int64").astype(str).str.zfill(2)
    work["MPIOCVE"] = work["MPIOCVE"].astype("int64").astype(str).str.zfill(3)
    work["CVEGEO"] = work["EDOCVE"] + work["MPIOCVE"]

    agg = (
        work.groupby("CVEGEO", as_index=False, observed=True)[["SV", "NV", "NS", "LN"]]
        .sum()
    )

    mun_names = (
        work[["CVEGEO", "MPIONOM"]]
        .drop_duplicates(subset=["CVEGEO"])
    )

    out = agg.merge(
        mun_names,
        on="CVEGEO",
        how="left",
        validate="1:1",
    )

    out["total_marked"] = out["SV"] + out["NV"]

    out["sv_ratio"] = np.where(out["LN"] > 0, out["SV"] / out["LN"], np.nan)
    out["nv_ratio"] = np.where(out["LN"] > 0, out["NV"] / out["LN"], np.nan)
    out["ns_ratio"] = np.where(out["LN"] > 0, out["NS"] / out["LN"], np.nan)

    return out



def prepare_state_map_data(
    df: pd.DataFrame,
    gdf: gpd.GeoDataFrame,
    state_code_num: int,
) -> gpd.GeoDataFrame:
    df_state = df[df["EDOCVE"] == state_code_num].copy()
    df_mun = build_municipal_metrics(df_state)

    state_code_str = str(state_code_num).zfill(2)
    gdf_state = gdf[gdf["CVE_ENT"] == state_code_str].copy()

    gdf_merged = gdf_state.merge(
        df_mun,
        on="CVEGEO",
        how="left",
        validate="1:1",
    )

    return gdf_merged


def build_state_choropleth_plotly(
    gdf_map: gpd.GeoDataFrame,
    column: str = "sv_ratio",
    state_name: str | None = None,
    year: int | None = None,
    width: int = 1000,
    height: int = 700,
):
    """
    Construye un mapa coroplético interactivo municipal con Plotly.
    """
    valid_columns = {"sv_ratio", "nv_ratio", "SV", "NV"}
    if column not in valid_columns:
        raise ValueError(f"column debe ser una de {valid_columns}")

    plot_gdf = gdf_map.copy()

    if plot_gdf.crs is None:
        raise ValueError("El GeoDataFrame no tiene CRS definido.")

    plot_gdf["municipio_label"] = plot_gdf["MPIONOM"].fillna(plot_gdf["NOMGEO"])
    plot_gdf["CVEGEO"] = plot_gdf["CVEGEO"].astype(str)

    plot_gdf = plot_gdf[plot_gdf.geometry.notna()].copy()
    plot_gdf = plot_gdf.to_crs(epsg=4326)
    plot_gdf = plot_gdf.explode(index_parts=False).reset_index(drop=True)
    plot_gdf["feature_id"] = plot_gdf["CVEGEO"]

    custom_data_cols = ["sv_ratio", "nv_ratio", "SV", "NV"]

    if column == "sv_ratio":
        color_scale = "Greens"
        title_text = "Participación municipal"
        colorbar_title = "%"
        range_color = (0, 1)
        hovertemplate = (
            "<b>%{hovertext}</b><br>"
            "Participación: %{customdata[0]:.2%}<br>"
            "Abstencionismo: %{customdata[1]:.2%}"
            "<extra></extra>"
        )
    elif column == "nv_ratio":
        color_scale = "Reds"
        title_text = "Abstencionismo municipal"
        colorbar_title = "%"
        range_color = (0, 1)
        hovertemplate = (
            "<b>%{hovertext}</b><br>"
            "Participación: %{customdata[0]:.2%}<br>"
            "Abstencionismo: %{customdata[1]:.2%}"
            "<extra></extra>"
        )
    elif column == "SV":
        color_scale = "Greens"
        title_text = "Participación total municipal"
        colorbar_title = ""
        range_color = None
        hovertemplate = (
            "<b>%{hovertext}</b><br>"
            "Participación total: %{customdata[2]:,}<br>"
            "Abstencionismo total: %{customdata[3]:,}"
            "<extra></extra>"
        )
    else:
        color_scale = "Reds"
        title_text = "Abstencionismo total municipal"
        colorbar_title = ""
        range_color = None
        hovertemplate = (
            "<b>%{hovertext}</b><br>"
            "Participación total: %{customdata[2]:,}<br>"
            "Abstencionismo total: %{customdata[3]:,}"
            "<extra></extra>"
        )

    geojson_dict = json.loads(
        plot_gdf[["feature_id", "geometry"]].to_json()
    )

    for feature, fid in zip(geojson_dict["features"], plot_gdf["feature_id"]):
        feature["id"] = fid

    subtitle_parts = []
    if state_name:
        subtitle_parts.append(state_name)
    if year is not None:
        subtitle_parts.append(str(year))
    subtitle_text = " - ".join(subtitle_parts)

    fig = px.choropleth(
        plot_gdf,
        geojson=geojson_dict,
        locations="feature_id",
        featureidkey="id",
        color=column,
        color_continuous_scale=color_scale,
        range_color=range_color,
        hover_name="municipio_label",
        custom_data=custom_data_cols,
    )

    fig.update_geos(
        fitbounds="geojson",
        visible=False,
    )

    fig.update_traces(
        marker_line_color="black",
        marker_line_width=0.6,
        hovertemplate=hovertemplate,
    )

    full_title = title_text
    if subtitle_text:
        full_title += f"<br><sup>{subtitle_text}</sup>"

    fig.update_layout(
        title=dict(text=full_title, x=0.02, xanchor="left"),
        margin=dict(l=0, r=0, t=70, b=0),
        coloraxis_colorbar=dict(
            title=colorbar_title,
            len=0.75,
            thickness=18,
            y=0.5,
            yanchor="middle",
        ),
        width=width,
        height=height,
    )

    return fig
