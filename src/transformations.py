from __future__ import annotations
import pandas as pd


#--------------------------------------------------------------------
# DIMENSIONES DERIVADAS: edad como grupos
#--------------------------------------------------------------------

def add_age_group(
    df: pd.DataFrame,
    age_col: str = "EDAD",
    out_col: str = "generacion",
) -> None:
    bins = [18, 30, 45, 65, float("inf")]
    labels = ["Joven", "Adulto joven", "Adulto", "Adulto mayor"]
    df[out_col] = pd.cut(df[age_col], bins=bins, labels=labels, right=False)

#--------------------------------------------------------------------
# DIMENSIONES DERIVADAS: sexo 
#--------------------------------------------------------------------

def add_sex_label(
    df: pd.DataFrame,
    sex_col: str = "SEXO",
    out_col: str = "sexo",
) -> None:
    mapping = {0: "Hombre", 1: "Mujer", 2: "No binario"}
    df[out_col] = df[sex_col].map(mapping).astype("category")

#--------------------------------------------------------------------
# DIMENSIONES DERIVADAS: tipo de sección
#--------------------------------------------------------------------

def add_section_type_label(
    df: pd.DataFrame,
    sec_col: str = "TIPOSEC",
    out_col: str = "tipo_seccion",
) -> None:
    mapping = {"U": "Urbana", "M": "Mixta", "R": "Rural"}
    df[out_col] = df[sec_col].map(mapping).astype("category")

#--------------------------------------------------------------------
# DIMENSIONES DERIVADAS: función general para estandarizar todas las dimensiones derivadas 
#--------------------------------------------------------------------
def standardize_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    add_age_group(out)
    add_sex_label(out)
    add_section_type_label(out)
    return out

