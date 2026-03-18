import pandas as pd

def optimize_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    int_cols = [
        "SV", "NV", "LN", "NS", "EDAD", "SEXO",
        "EDOCVE", "MPIOCVE", "SECCION", "year",
        "AELEC", "DEL", "DEF"
    ]

    for col in int_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int32")

    cat_cols = ["EDONOM", "MPIONOM", "TIPOSEC", "state_code"]
    for col in cat_cols:
        if col in out.columns:
            out[col] = out[col].astype("category")

    if "FELECCION" in out.columns:
        out["FELECCION"] = pd.to_datetime(out["FELECCION"], errors="coerce")

    return out
