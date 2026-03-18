from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt

from src.pipeline import load_processed_multi_year
from src.modeling.feature_engineering import build_modeling_dataframe
from src.modeling.baseline import baseline_naive_last_value
from src.modeling.validation import get_fixed_time_folds
from src.modeling.evaluation import evaluate_predictions
from src.config import PROJECT_ROOT
from src.modeling.modelos import (
    train_linear,
    train_ridge,
    train_random_forest,
    predict,
    extract_linear_coefficients,
)
from src.modeling.diagnostics import (
    build_prediction_frame,
    plot_predicted_vs_real,
    plot_error_distribution,
    load_municipal_geometries_for_diagnostics,
    build_error_map_frame,
    build_error_choropleth_plotly,
)


# --------------------------------------------------
# CONFIGURACIÓN
# --------------------------------------------------
FEATURE_COLS = [
    "lag_sv_ratio_1",
    "lag_sv_ratio_2",
    "state_sv_ratio_mean_1",
]

TARGET_COL = "target_sv_ratio_next"

SHAPEFILE_PATH = PROJECT_ROOT / "data" / "geometrias_municipios" / "mun22cw.shp"


# --------------------------------------------------
# CARGA Y FEATURE ENGINEERING
# --------------------------------------------------
df_all = load_processed_multi_year(
    PROJECT_ROOT,
    (2009, 2012, 2015, 2018, 2021, 2024),
)

model_df = build_modeling_dataframe(df_all)


# --------------------------------------------------
# REVISIÓN DE FEATURES
# --------------------------------------------------
print("\nShape inicial model_df:")
print(model_df.shape)

print("\nColumnas:")
print(model_df.columns.tolist())

print("\nPrimeras filas clave:")
print(
    model_df[
        ["CVEGEO", "year", "sv_ratio", "lag_sv_ratio_1", "target_sv_ratio_next"]
    ].head(10)
)

print("\nMunicipios únicos:")
print(model_df["CVEGEO"].nunique())

print("\nMissing values por feature:")
print(model_df[FEATURE_COLS + [TARGET_COL]].isna().sum())

print("\nResumen descriptivo de features:")
print(model_df[FEATURE_COLS].describe().T)

print("\nCorrelación entre features:")
print(model_df[FEATURE_COLS].corr().round(3))

review_df = model_df[FEATURE_COLS + [TARGET_COL]].dropna().copy()
print("\nCorrelación con target_sv_ratio_next:")
print(review_df.corr()[TARGET_COL].sort_values(ascending=False))


# --------------------------------------------------
# LIMPIEZA PARA MODELADO
# --------------------------------------------------
model_df = model_df.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()

print("\nShape final model_df tras dropna en features + target:")
print(model_df.shape)


# --------------------------------------------------
# VALIDACIÓN TEMPORAL
# --------------------------------------------------
folds = get_fixed_time_folds()

results = []
prediction_frames = []

for fold in folds:
    train_years = fold["train_years"]
    test_year = fold["test_year"]

    train_df = model_df[model_df["year"].isin(train_years)].copy()
    test_df = model_df[model_df["year"] == test_year].copy()

    print(f"\nFold test_year={test_year}")
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

    # -----------------------------
    # BASELINE NAIVE
    # -----------------------------
    pred_df = baseline_naive_last_value(
        test_df,
        target_col=TARGET_COL,
        lag_col="lag_sv_ratio_1",
    )

    metrics = evaluate_predictions(pred_df)
    metrics["model"] = "naive_last_value"
    metrics["test_year"] = test_year
    results.append(metrics)

    prediction_frames.append(
        build_prediction_frame(
            pred_df=pred_df,
            model_name="naive_last_value",
            target_col=TARGET_COL,
            test_year=test_year,
        )
    )

    # -----------------------------
    # REGRESIÓN LINEAL
    # -----------------------------
    model = train_linear(
        train_df,
        TARGET_COL,
        features=FEATURE_COLS,
    )

    pred_df = predict(
        model,
        test_df,
        features=FEATURE_COLS,
    )
    pred_df["y_true"] = pred_df[TARGET_COL]

    metrics = evaluate_predictions(pred_df)
    metrics["model"] = "linear_regression"
    metrics["test_year"] = test_year
    results.append(metrics)

    prediction_frames.append(
        build_prediction_frame(
            pred_df=pred_df,
            model_name="linear_regression",
            target_col=TARGET_COL,
            test_year=test_year,
        )
    )

    # -----------------------------
    # RIDGE
    # -----------------------------
    model = train_ridge(
        train_df,
        TARGET_COL,
        alpha=1.0,
        features=FEATURE_COLS,
    )

    pred_df = predict(
        model,
        test_df,
        features=FEATURE_COLS,
    )
    pred_df["y_true"] = pred_df[TARGET_COL]

    metrics = evaluate_predictions(pred_df)
    metrics["model"] = "ridge"
    metrics["test_year"] = test_year
    results.append(metrics)

    prediction_frames.append(
        build_prediction_frame(
            pred_df=pred_df,
            model_name="ridge",
            target_col=TARGET_COL,
            test_year=test_year,
        )
    )

    # -----------------------------
    # RANDOM FOREST
    # -----------------------------
    model = train_random_forest(
        train_df,
        TARGET_COL,
        features=FEATURE_COLS,
    )

    pred_df = predict(
        model,
        test_df,
        features=FEATURE_COLS,
    )
    pred_df["y_true"] = pred_df[TARGET_COL]

    metrics = evaluate_predictions(pred_df)
    metrics["model"] = "random_forest"
    metrics["test_year"] = test_year
    results.append(metrics)

    prediction_frames.append(
        build_prediction_frame(
            pred_df=pred_df,
            model_name="random_forest",
            target_col=TARGET_COL,
            test_year=test_year,
        )
    )


# --------------------------------------------------
# RESULTADOS
# --------------------------------------------------
results_df = pd.DataFrame(results)

print("\nResultados por fold y modelo:\n")
print(results_df.to_string(index=False))

summary_df = (
    results_df.groupby("model", as_index=False)[["mae", "rmse", "n"]]
    .mean()
    .sort_values("mae")
)

print("\nResumen promedio por modelo:\n")
print(summary_df.to_string(index=False))


# --------------------------------------------------
# COEFICIENTES FINALES
# Entrenamos con todo el dataset limpio disponible
# --------------------------------------------------
linear_model = train_linear(
    model_df,
    TARGET_COL,
    features=FEATURE_COLS,
)

coef_linear = extract_linear_coefficients(
    linear_model,
    FEATURE_COLS,
)

print("\nCoeficientes regresión lineal:\n")
print(coef_linear.to_string(index=False))


ridge_model = train_ridge(
    model_df,
    TARGET_COL,
    alpha=1.0,
    features=FEATURE_COLS,
)

coef_ridge = extract_linear_coefficients(
    ridge_model,
    FEATURE_COLS,
)

print("\nCoeficientes Ridge:\n")
print(coef_ridge.to_string(index=False))


# --------------------------------------------------
# DATAFRAME CONSOLIDADO DE PREDICCIONES
# --------------------------------------------------
predictions_df = pd.concat(prediction_frames, ignore_index=True)

print("\nShape predictions_df:")
print(predictions_df.shape)

print("\nPrimeras filas predictions_df:")
print(predictions_df.head())


# --------------------------------------------------
# GRÁFICOS DE DIAGNÓSTICO
# --------------------------------------------------
fig_pred_real, ax_pred_real = plot_predicted_vs_real(
    predictions_df,
    model_name="ridge",
    test_year=2021,
)

fig_error_dist, ax_error_dist = plot_error_distribution(
    predictions_df,
    model_name="ridge",
    test_year=2021,
)

# En algunos entornos locales fig.show() no abre ventana; por eso guardamos siempre.
fig_pred_real.savefig(
    PROJECT_ROOT / "ridge_pred_vs_real_2021.png",
    dpi=150,
    bbox_inches="tight",
)

fig_error_dist.savefig(
    PROJECT_ROOT / "ridge_error_dist_2021.png",
    dpi=150,
    bbox_inches="tight",
)

print("\nGráficos guardados:")
print(PROJECT_ROOT / "ridge_pred_vs_real_2021.png")
print(PROJECT_ROOT / "ridge_error_dist_2021.png")


# --------------------------------------------------
# MAPAS DE ERROR
# --------------------------------------------------
gdf_municipios = load_municipal_geometries_for_diagnostics(SHAPEFILE_PATH)

gdf_error = build_error_map_frame(
    predictions_df=predictions_df,
    gdf_municipios=gdf_municipios,
    model_name="ridge",
    test_year=2021,
)

fig_map_error = build_error_choropleth_plotly(
    gdf_error,
    color_col="error",
    title="Error municipal Ridge - 2021",
)

fig_map_abs_error = build_error_choropleth_plotly(
    gdf_error,
    color_col="abs_error",
    title="Error absoluto municipal Ridge - 2021",
)

fig_map_error.write_html(PROJECT_ROOT / "ridge_error_map_2021.html")
fig_map_abs_error.write_html(PROJECT_ROOT / "ridge_abs_error_map_2021.html")

print("\nMapas guardados:")
print(PROJECT_ROOT / "ridge_error_map_2021.html")
print(PROJECT_ROOT / "ridge_abs_error_map_2021.html")

#--------------------------------------------------
# ANÁLISIS DE MUNICIPIOS POR ERROR
#--------------------------------------------------
top_errors = (
    predictions_df
    .query("model == 'ridge' and test_year == 2021")
    .sort_values("abs_error", ascending=False)
    .head(25)
)

print(top_errors[
    [
        "CVEGEO",
        "municipio",
        "state_code",
        "y_true",
        "y_pred",
        "error",
        "abs_error",
    ]
])

top_features = model_df.merge(
    top_errors[["CVEGEO"]],
    on="CVEGEO",
    how="inner"
)

print(
    top_features[
        [
            "CVEGEO",
            "municipio",
            "year",
            "sv_ratio",
            "lag_sv_ratio_1",
            "lag_sv_ratio_2",
            "state_sv_ratio_mean_1",
            "prop_urbana",
            "prop_joven",
        ]
    ].head(20)
)

volatility = (
    model_df
    .groupby("CVEGEO")["sv_ratio"]
    .std()
    .rename("turnout_volatility")
)

volatility = volatility.reset_index()

analysis_df = top_errors.merge(volatility, on="CVEGEO")

print(
    analysis_df[
        [
            "municipio",
            "abs_error",
            "turnout_volatility"
        ]
    ].sort_values("abs_error", ascending=False)
)

volatility.describe()
analysis_df["turnout_volatility"].describe()

size_analysis = (
    predictions_df
    .query("model == 'ridge' and test_year == 2021")
    .merge(model_df[["CVEGEO","LN"]], on="CVEGEO")
)

print(
    size_analysis.sort_values("abs_error", ascending=False)[
        ["municipio","LN","abs_error"]
    ].head(20)
)

analysis = (
    predictions_df
    .query("model == 'ridge' and test_year == 2021")
    .merge(model_df[["CVEGEO", "LN"]], on="CVEGEO")
)

analysis["ln_size"] = pd.qcut(analysis["LN"], 5)

print("\nError promedio por tamaño LN:")
print(
    analysis
    .groupby("ln_size")["abs_error"]
    .mean().sort_index()
)

print("\nNúmero de observaciones por grupo:")
print(
    analysis
    .groupby("ln_size")["abs_error"]
    .count()
)

plt.figure(figsize=(8,6))

plt.scatter(
    analysis["LN"],
    analysis["abs_error"],
    alpha=0.35
)

plt.xscale("log")

plt.xlabel("Lista nominal (escala log)")
plt.ylabel("Error absoluto del modelo")

plt.title("Error de predicción vs tamaño del electorado")

plt.grid(True, alpha=0.3)

plt.savefig(
    PROJECT_ROOT / "error_vs_ln_ridge_2021.png",
    dpi=150,
    bbox_inches="tight"
)

