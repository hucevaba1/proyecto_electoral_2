from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_predictions(df: pd.DataFrame) -> dict[str, float]:
    work = df.dropna(subset=["y_true", "y_pred"]).copy()

    if work.empty:
        return {"mae": np.nan, "rmse": np.nan, "n": 0}

    mae = mean_absolute_error(work["y_true"], work["y_pred"])
    rmse = mean_squared_error(work["y_true"], work["y_pred"]) ** 0.5


    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "n": int(len(work)),
    }