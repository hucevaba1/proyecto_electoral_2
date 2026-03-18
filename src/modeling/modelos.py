from __future__ import annotations

import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor


# --------------------------------------------------
# FEATURES UTILIZADAS
# --------------------------------------------------

DEFAULT_FEATURES = [
    "lag_sv_ratio_1",
    "lag_sv_ratio_2", 
    "state_sv_ratio_mean_1",
]



# --------------------------------------------------
# ENTRENAMIENTO
# --------------------------------------------------

def train_linear(
    train_df: pd.DataFrame,
    target: str,
    features: list[str] | None = None,
):
    features = features or DEFAULT_FEATURES
    X = train_df[features]
    y = train_df[target]

    model = LinearRegression()
    model.fit(X, y)
    return model


def train_ridge(
    train_df: pd.DataFrame,
    target: str,
    alpha: float = 1.0,
    features: list[str] | None = None,
):
    features = features or DEFAULT_FEATURES
    X = train_df[features]
    y = train_df[target]

    model = Ridge(alpha=alpha)
    model.fit(X, y)
    return model


def train_random_forest(
    train_df: pd.DataFrame,
    target: str,
    features: list[str] | None = None,
):
    features = features or DEFAULT_FEATURES
    X = train_df[features]
    y = train_df[target]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model

# --------------------------------------------------
# PREDICCIÓN
# --------------------------------------------------

def predict(
    model,
    df: pd.DataFrame,
    features: list[str] | None = None,
):
    features = features or DEFAULT_FEATURES
    X = df[features]

    out = df.copy()
    out["y_pred"] = model.predict(X)
    return out

# --------------------------------------------------
# COEFICIENTES
# --------------------------------------------------

def extract_linear_coefficients(model, features):

    coef_df = pd.DataFrame({
        "feature": features,
        "coefficient": model.coef_
    })

    coef_df = coef_df.sort_values(
        "coefficient",
        key=abs,
        ascending=False
    )

    return coef_df