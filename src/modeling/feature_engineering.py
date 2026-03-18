from __future__ import annotations

import pandas as pd
import numpy as np


def build_municipio_year_base(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega el dataset base a nivel municipio-año.
    """
    work = df.copy()

    work["EDOCVE"] = work["EDOCVE"].astype("Int32")
    work["MPIOCVE"] = work["MPIOCVE"].astype("Int32")

    base = (
        work.groupby(
            ["state_code", "EDOCVE", "MPIOCVE", "MPIONOM", "year"],
            observed=True,
            as_index=False,
        )[["SV", "NV", "NS", "LN"]]
        .sum()
    )

    base["CVEGEO"] = (
        base["EDOCVE"].astype("Int64").astype(str).str.zfill(2)
        + base["MPIOCVE"].astype("Int64").astype(str).str.zfill(3)
    )

    base["municipio"] = base["MPIONOM"].astype("string")
    base["total_marked"] = base["SV"] + base["NV"]

    base["sv_ratio"] = np.where(
        base["LN"] > 0,
        base["SV"] / base["LN"],
        np.nan,
    )
    base["nv_ratio"] = np.where(
        base["LN"] > 0,
        base["NV"] / base["LN"],
        np.nan,
    )
    base["ns_ratio"] = np.where(
        base["LN"] > 0,
        base["NS"] / base["LN"],
        np.nan,
    )

    return base.sort_values(["CVEGEO", "year"]).reset_index(drop=True)


def build_composition_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye proporciones de composición municipal usando LN como peso.
    Requiere columnas derivadas: sexo, generacion, tipo_seccion.
    """
    work = df.copy()

    keys = ["state_code", "EDOCVE", "MPIOCVE", "MPIONOM", "year"]

    total_ln = (
        work.groupby(keys, observed=True, as_index=False)[["LN"]]
        .sum()
        .rename(columns={"LN": "LN_total"})
    )

    sexo = (
        work.groupby(keys + ["sexo"], observed=True, as_index=False)[["LN"]]
        .sum()
        .pivot(index=keys, columns="sexo", values="LN")
        .reset_index()
    )
    sexo.columns = [str(c) for c in sexo.columns]

    edad = (
        work.groupby(keys + ["generacion"], observed=True, as_index=False)[["LN"]]
        .sum()
        .pivot(index=keys, columns="generacion", values="LN")
        .reset_index()
    )
    edad.columns = [str(c) for c in edad.columns]

    seccion = (
        work.groupby(keys + ["tipo_seccion"], observed=True, as_index=False)[["LN"]]
        .sum()
        .pivot(index=keys, columns="tipo_seccion", values="LN")
        .reset_index()
    )
    seccion.columns = [str(c) for c in seccion.columns]

    out = total_ln.merge(sexo, on=keys, how="left")
    out = out.merge(edad, on=keys, how="left")
    out = out.merge(seccion, on=keys, how="left")

    value_cols = [c for c in out.columns if c not in keys]
    out[value_cols] = out[value_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    def safe_prop(num_col: str, den_col: str = "LN_total") -> pd.Series:
        return np.where(out[den_col] > 0, out[num_col] / out[den_col], np.nan)

    # Sexo
    if "Hombre" in out.columns:
        out["prop_hombre"] = safe_prop("Hombre")
    else:
        out["prop_hombre"] = np.nan

    if "Mujer" in out.columns:
        out["prop_mujer"] = safe_prop("Mujer")
    else:
        out["prop_mujer"] = np.nan

    if "No binario" in out.columns:
        out["prop_no_binario"] = safe_prop("No binario")
    else:
        out["prop_no_binario"] = np.nan

    # Edad
    if "Joven" in out.columns:
        out["prop_joven"] = safe_prop("Joven")
    else:
        out["prop_joven"] = np.nan

    if "Adulto joven" in out.columns:
        out["prop_adulto_joven"] = safe_prop("Adulto joven")
    else:
        out["prop_adulto_joven"] = np.nan

    if "Adulto" in out.columns:
        out["prop_adulto"] = safe_prop("Adulto")
    else:
        out["prop_adulto"] = np.nan

    if "Adulto mayor" in out.columns:
        out["prop_adulto_mayor"] = safe_prop("Adulto mayor")
    else:
        out["prop_adulto_mayor"] = np.nan

    # Tipo de sección
    if "Urbana" in out.columns:
        out["prop_urbana"] = safe_prop("Urbana")
    else:
        out["prop_urbana"] = np.nan

    if "Mixta" in out.columns:
        out["prop_mixta"] = safe_prop("Mixta")
    else:
        out["prop_mixta"] = np.nan

    if "Rural" in out.columns:
        out["prop_rural"] = safe_prop("Rural")
    else:
        out["prop_rural"] = np.nan

    keep_cols = keys + [c for c in out.columns if c.startswith("prop_")]
    return out[keep_cols].copy()


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega rezagos y cambios por municipio.
    """
    out = df.copy().sort_values(["CVEGEO", "year"]).reset_index(drop=True)

    by = out.groupby("CVEGEO", observed=True)

    out["lag_sv_ratio_1"] = by["sv_ratio"].shift(1)
    out["lag_sv_ratio_2"] = by["sv_ratio"].shift(2)
    out["lag_nv_ratio_1"] = by["nv_ratio"].shift(1)
    out["lag_ns_ratio_1"] = by["ns_ratio"].shift(1)

    out["lag_SV_1"] = by["SV"].shift(1)
    out["lag_NV_1"] = by["NV"].shift(1)
    out["lag_LN_1"] = by["LN"].shift(1)
    out["lag_LN_2"] = by["LN"].shift(2)

    out["delta_sv_ratio_1"] = out["lag_sv_ratio_1"] - out["lag_sv_ratio_2"]
    out["delta_ln_1"] = out["lag_LN_1"] - out["lag_LN_2"]

    return out


def add_state_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega contexto estatal rezagado.
    """
    out = df.copy()

    state_year = (
        out.groupby(["state_code", "year"], observed=True, as_index=False)[["SV", "NV", "NS", "LN"]]
        .sum()
    )

    state_year["state_sv_ratio"] = np.where(
        state_year["LN"] > 0,
        state_year["SV"] / state_year["LN"],
        np.nan,
    )

    state_year = state_year.sort_values(["state_code", "year"]).reset_index(drop=True)

    state_year["state_sv_ratio_mean_1"] = (
        state_year.groupby("state_code", observed=True)["state_sv_ratio"].shift(1)
    )

    out = out.merge(
        state_year[["state_code", "year", "state_sv_ratio_mean_1"]],
        on=["state_code", "year"],
        how="left",
        validate="m:1",
    )

    out["state_sv_ratio_mean_diff_1"] = (
        out["lag_sv_ratio_1"] - out["state_sv_ratio_mean_1"]
    )

    return out


def add_next_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea targets de la siguiente elección por municipio.
    """
    out = df.copy().sort_values(["CVEGEO", "year"]).reset_index(drop=True)

    by = out.groupby("CVEGEO", observed=True)
    out["target_sv_ratio_next"] = by["sv_ratio"].shift(-1)
    out["target_SV_next"] = by["SV"].shift(-1)

    return out


def build_modeling_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye el dataframe final para modelado.
    """
    base = build_municipio_year_base(df)
    composition = build_composition_features(df)

    out = base.merge(
        composition,
        on=["state_code", "EDOCVE", "MPIOCVE", "MPIONOM", "year"],
        how="left",
        validate="1:1",
    )

    out = add_lag_features(out)
    out = add_state_context_features(out)
    out = add_next_targets(out)

    return out.sort_values(["CVEGEO", "year"]).reset_index(drop=True)
