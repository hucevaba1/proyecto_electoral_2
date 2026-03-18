from __future__ import annotations

from pathlib import Path
import pandas as pd
from typing import Iterable

from src.data_loader import load_data, load_year_data, load_state_data
from src.data_cleaning import optimize_types
from src.transformations import standardize_dimensions

#--------------------------------------------------------------------
# PIPELINE COMPLETO: optimización + dimensiones derivadas
#--------------------------------------------------------------------
def prepare_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica el pipeline completo de preparación:
    1) optimización de tipos
    2) estandarización de dimensiones derivadas

    Devuelve un DataFrame listo para análisis y visualización.
    """
    df = optimize_types(df_raw)
    df = standardize_dimensions(df)
    return df

#--------------------------------------------------------------------
# CARGA MULTIANUAL: todos los estados, años seleccionados + optimización y transformaciones implicadas en prepare_dataframe
#--------------------------------------------------------------------
def load_processed_data(
    base_dir: str | Path,
    years: tuple[int, ...] = (2009, 2012, 2015, 2018, 2021, 2024),
) -> pd.DataFrame:
    """
    Carga todos los datos para los años indicados y devuelve
    un DataFrame procesado.
    """
    df_raw = load_data(base_dir=base_dir, years=years)
    return prepare_dataframe(df_raw)

#--------------------------------------------------------------------
# CARGA POR AÑO: todos los estados (o subconjunto) para un año específico + optimización y transformaciones implicadas en prepare_dataframe
#--------------------------------------------------------------------
def load_processed_year(
    base_dir: str | Path,
    year: int,
    states: Iterable[str] | None = None,
    strict_states: bool = True,
) -> pd.DataFrame:
    """
    Carga un año específico y devuelve un DataFrame procesado.
    Puede filtrar por uno o varios estados.
    """
    df_raw = load_year_data(
        base_dir=base_dir,
        year=year,
        states=states,
        strict_states=strict_states,
    )
    return prepare_dataframe(df_raw)

#--------------------------------------------------------------------
# CARGA POR ESTADO: todas las elecciones de una entidad + optimización y transformaciones implicadas en prepare_dataframe
#--------------------------------------------------------------------
def load_processed_state(
    base_dir: str | Path,
    state_code: str,
    years: tuple[int, ...] = (2009, 2012, 2015, 2018, 2021, 2024),
) -> pd.DataFrame:
    """
    Carga un estado específico para uno o varios años y devuelve
    un DataFrame procesado.
    """
    df_raw = load_state_data(
        base_dir=base_dir,
        state_code=state_code,
        years=years,
    )
    return prepare_dataframe(df_raw)

#--------------------------------------------------------------------
# ALIAS: carga multianual para series temporales + optimización y transformaciones implicadas en prepare_dataframe
#--------------------------------------------------------------------
def load_processed_multi_year(
    base_dir: str | Path,
    years: tuple[int, ...],
) -> pd.DataFrame:
    """
    Alias explícito para series temporales o análisis multianuales.
    """
    return load_processed_data(base_dir=base_dir, years=years)