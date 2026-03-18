from __future__ import annotations

import pandas as pd

# =====================================================
# AÑADE EL PRONOSRTICO NAIVE (Yt1 = Yt0)
# =====================================================

def baseline_naive_last_value(
    df: pd.DataFrame,
    target_col: str = "target_sv_ratio_next",
    lag_col: str = "lag_sv_ratio_1",
) -> pd.DataFrame:
    """
    Baseline naive:
    predice el siguiente valor igual al último observado.
    """
    out = df.copy()
    out["y_true"] = out[target_col]
    out["y_pred"] = out[lag_col]
    return out

#=====================================================
# AÑADE EL PRONÓSTICO DE LA MEDIA POR ESTADO (Yt1 = media de Yt0 por estado)
#=====================================================
def baseline_historical_mean(
    df: pd.DataFrame,
    target_col: str = "target_sv_ratio_next",
) -> pd.DataFrame:
    """
    Baseline de promedio histórico municipal hasta t.
    """
    out = df.copy().sort_values(["CVEGEO", "year"]).reset_index(drop=True)

    expanding_mean = (
        out.groupby("CVEGEO", observed=True)["sv_ratio"]
        .transform(lambda s: s.expanding().mean())
    )

    # El promedio histórico disponible en t debe excluir el target futuro,
    # así que el valor en t sirve como predicción de t+1.
    out["y_true"] = out[target_col]
    out["y_pred"] = expanding_mean

    return out