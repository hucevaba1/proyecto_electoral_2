from __future__ import annotations

import pandas as pd

# =====================================================
# CREA FOLDERS PARA ENTRENAMIENTO: 2018 CON 2015, 2012 Y 2009 | 2021 CON 2018, 2015, 2012 Y 2009 | 2024 CON TODOS LOS AÑOS ANTERIORES
# =====================================================
def get_fixed_time_folds() -> list[dict[str, list[int] | int]]:
    """
    Folds temporales manuales para 6 elecciones.
    """
    return [
        {"train_years": [2009, 2012, 2015], "test_year": 2018},
        {"train_years": [2009, 2012, 2015, 2018], "test_year": 2021},
    ]

def split_fold(
    df: pd.DataFrame,
    train_years: list[int],
    test_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["year"].isin(train_years)].copy()
    test = df[df["year"] == test_year].copy()

    return train, test

