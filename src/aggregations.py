from __future__ import annotations
import pandas as pd
import numpy as np


# =====================================================
# FUNCIÓN BASE DE CÁLCULO DE RATIOS
# =====================================================
def _compute_ratios(
    df: pd.DataFrame,
    sv_col: str = "SV",
    nv_col: str = "NV",
    ns_col: str = "NS",
    ln_col: str = "LN",
) -> pd.DataFrame:
    """
    Añade columnas:
        total_marked = SV + NV
        sv_ratio = SV / LN
        nv_ratio = NV / LN
        ns_ratio = NS / LN
    """
    out = df.copy()

    if ln_col not in out.columns:
        raise ValueError(f"La columna {ln_col!r} no existe en el DataFrame.")

    out["total_marked"] = out[sv_col] + out[nv_col]

    out["sv_ratio"] = np.where(
        out[ln_col] > 0,
        out[sv_col] / out[ln_col],
        np.nan,
    )

    out["nv_ratio"] = np.where(
        out[ln_col] > 0,
        out[nv_col] / out[ln_col],
        np.nan,
    )

    if ns_col in out.columns:
        out["ns_ratio"] = np.where(
            out[ln_col] > 0,
            out[ns_col] / out[ln_col],
            np.nan,
        )
    else:
        out["ns_ratio"] = np.nan

    return out


# =====================================================
# AGRUPACIÓN SIMPLE
# =====================================================
def aggregate_group(
    df: pd.DataFrame,
    group_col: str,
    sv_col: str = "SV",
    nv_col: str = "NV",
    ns_col: str = "NS",
    ln_col: str = "LN",
) -> pd.DataFrame:
    values = [sv_col, nv_col, ns_col, ln_col]

    out = (
        df.groupby(group_col, observed=True)[values]
        .sum(min_count=1)
    )

    for c in values:
        out[c] = out[c].fillna(0).astype("int32")

    out = _compute_ratios(
        out,
        sv_col=sv_col,
        nv_col=nv_col,
        ns_col=ns_col,
        ln_col=ln_col,
    )

    return out.sort_values("sv_ratio", ascending=False)


# =====================================================
# AGRUPACIÓN DIMENSIÓN + AÑO
# =====================================================
def aggregate_group_year(
    df: pd.DataFrame,
    group_col: str,
    year_col: str = "year",
    sv_col: str = "SV",
    nv_col: str = "NV",
    ns_col: str = "NS",
    ln_col: str = "LN",
) -> pd.DataFrame:
    out = (
        df.groupby([group_col, year_col], observed=True)[[sv_col, nv_col, ns_col, ln_col]]
        .sum(min_count=1)
        .reset_index()
    )

    for c in [sv_col, nv_col, ns_col, ln_col]:
        out[c] = out[c].fillna(0).astype("int32")

    out = _compute_ratios(
        out,
        sv_col=sv_col,
        nv_col=nv_col,
        ns_col=ns_col,
        ln_col=ln_col,
    )

    return out.sort_values([group_col, year_col]).reset_index(drop=True)


# =====================================================
# AGRUPACIÓN ESTADO + AÑO
# =====================================================
def aggregate_state_year(
    df: pd.DataFrame,
    sv_col: str = "SV",
    nv_col: str = "NV",
    ns_col: str = "NS",
    ln_col: str = "LN",
) -> pd.DataFrame:
    out = (
        df.groupby(["state_code", "year"], observed=True)[[sv_col, nv_col, ns_col, ln_col]]
        .sum(min_count=1)
        .reset_index()
    )

    for c in [sv_col, nv_col, ns_col, ln_col]:
        out[c] = out[c].fillna(0).astype("int32")

    out = _compute_ratios(
        out,
        sv_col=sv_col,
        nv_col=nv_col,
        ns_col=ns_col,
        ln_col=ln_col,
    )

    return out.sort_values(["state_code", "year"]).reset_index(drop=True)
