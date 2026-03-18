# -------------------------------------------------
# IMPORTACIÓN DE LIBRERÍAS Y CONFIGURACIÓN DE RUTAS
# -------------------------------------------------
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.pipeline import (
    load_processed_year,
    load_processed_multi_year,
)
from src.views import (
    get_participation_view_for_chart,
    build_participation_chart_altair,
    get_state_year_view,
    get_group_year_view,
    build_state_year_charts_altair,
)
from src.constants import (
    STATE_LABELS,
    STATE_CODE_FROM_LABEL,
    GROUPINGS,
    GROUPING_TITLES,
    AVAILABLE_YEARS,
    TIME_SERIES_PALETTE,
    TIME_SERIES_DIMENSIONS,
    MODEL_LABELS,
    MODEL_KEYS_FROM_LABEL,
    FEATURE_LABELS,
)
from src.maps import (
    load_municipal_geometries,
    prepare_state_map_data,
    build_state_choropleth_plotly,
)
from src.modeling.pipeline import (
    run_model_diagnostics,
    build_forecast_2027_outputs,
)
from src.modeling.forecast_views import (
    add_prediction_interval_from_oos_error,
    plot_forecast_ranked,
)
from src.modeling.diagnostics import (
    plot_predicted_vs_real,
    plot_error_distribution,
    build_error_map_frame,
    build_error_choropleth_plotly,
)

# -------------------------------------------------
# RUTA DE SHAPEFILES DE GEOMETRÍAS MUNICIPALES
# -------------------------------------------------
SHAPEFILE_PATH = PROJECT_ROOT / "data" / "geometrias_municipios" / "mun22cw.shp"

# -------------------------------------------------
# CONFIGURACIÓN DE LA APP
# -------------------------------------------------
st.set_page_config(
    page_title="Explorador de participación electoral",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -------------------------------------------------
# CARGA DE DATOS CACHEADA
# -------------------------------------------------
@st.cache_data(show_spinner=False)
def get_year_processed_cached(base_dir: Path, year: int) -> pd.DataFrame:
    return load_processed_year(base_dir=base_dir, year=year)


@st.cache_data(show_spinner=False)
def get_multi_year_processed_cached(
    base_dir: Path,
    years: tuple[int, ...],
) -> pd.DataFrame:
    return load_processed_multi_year(base_dir=base_dir, years=years)


@st.cache_data(show_spinner=False)
def get_municipal_geometries_cached(shp_path: str):
    return load_municipal_geometries(shp_path)

# -------------------------------------------------
# HEADER
# -------------------------------------------------
def render_header() -> None:
    st.markdown(
        """
        <h1 style='text-align: center;'>
            Explorador de participación electoral 🗳️
        </h1>
        <p style='text-align: center; font-size: 1.2rem;'>
            ¿Sobre qué te gustaría conocer más?
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <style>
        .stTabs [data-baseweb="tab-list"] {
            justify-content: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# -------------------------------------------------
# HELPERS DE UI / MÉTRICAS
# -------------------------------------------------
def compute_weighted_metrics(df: pd.DataFrame) -> tuple[float, float]:
    sv_total = int(df["SV"].sum()) if "SV" in df.columns else 0
    nv_total = int(df["NV"].sum()) if "NV" in df.columns else 0
    ln_total = int(df["LN"].sum()) if "LN" in df.columns else 0

    if ln_total == 0:
        return 0.0, 0.0

    return sv_total / ln_total, nv_total / ln_total


def render_centered_legend(
    df: pd.DataFrame,
    color_domain: list[str],
    color_range: list[str],
) -> None:
    label_map = (
        df[["state_code", "state_label"]]
        .drop_duplicates()
        .assign(state_code=lambda d: d["state_code"].astype(str))
        .set_index("state_code")["state_label"]
        .to_dict()
    )

    legend_items = []
    for i, code in enumerate(color_domain):
        label = label_map.get(str(code), str(code))
        color = color_range[i]

        legend_items.append(
            f'<span style="display:inline-flex; align-items:center; margin:0 12px 8px 12px; font-size:0.95rem;">'
            f'<span style="width:12px; height:12px; border-radius:50%; background:{color}; display:inline-block; margin-right:6px;"></span>'
            f"{label}"
            f"</span>"
        )

    legend_html = (
        '<div style="text-align:center; margin-top:18px; margin-bottom:4px; line-height:1.8;">'
        + "".join(legend_items)
        + "</div>"
    )

    st.markdown(legend_html, unsafe_allow_html=True)


def render_time_series_metrics(df: pd.DataFrame) -> None:
    participacion, sin_marca = compute_weighted_metrics(df)

    col_meta1, col_meta2, col_meta3, col_meta4 = st.columns(4)

    with col_meta1:
        st.metric("Series totales", len(df["state_code"].unique()))

    with col_meta2:
        st.metric("Años", len(df["year"].unique()))

    with col_meta3:
        st.metric("Participación ponderada", f"{participacion:.1%}")

    with col_meta4:
        st.metric("Abstencionismo ponderado", f"{sin_marca:.1%}")

# -------------------------------------------------
# TAB 1: POR ESTADO
# -------------------------------------------------
def render_tab_estado() -> None:
    st.caption(
        "Selecciona un estado, año y categoría para analizar la participación electoral y el abstencionismo."
    )

    col_controls, col_results = st.columns([1, 3], gap="large")

    with col_controls:
        with st.container(border=True):
            st.subheader("Filtrar por")

            state_label = st.selectbox(
                "Entidad federativa 📍",
                options=sorted(STATE_LABELS.values()),
                help="Selecciona la entidad federativa.",
                key="tab_estado_state",
            )
            state_code = STATE_CODE_FROM_LABEL[state_label]

            year = st.selectbox(
                "Año 🗓️",
                options=list(AVAILABLE_YEARS),
                index=0,
                help="Selecciona el año.",
                key="tab_estado_year",
            )

            grouping_label = st.selectbox(
                "Categoría 🔭",
                options=list(GROUPINGS.keys()),
                help="Selecciona cómo quieres agrupar los datos.",
                key="tab_estado_grouping",
            )

            top_n: int | None = None
            if grouping_label == "Mostrar por municipio":
                top_n = st.number_input(
                    "Mostrar Top N municipios",
                    min_value=1,
                    max_value=600,
                    value=15,
                    step=1,
                    help="Filtra un Top de municipios según la participación total. El resto se agrupa como 'Otros'.",
                    key="tab_estado_topn",
                )

    with col_results:
        with st.status("Cargando datos...", expanded=False) as status:
            df_year = get_year_processed_cached(PROJECT_ROOT, int(year))
            status.write("Datos del año cargados")

            df_state = df_year[df_year["state_code"] == state_code].copy()
            status.write("Filtrado por estado aplicado")

            view = get_participation_view_for_chart(
                df=df_state,
                grouping_label=grouping_label,
                top_n=top_n,
            )
            status.write("Vista agregada generada")

            status.update(label="Datos listos", state="complete")

        sv_total = int(view["SV"].sum()) if "SV" in view.columns else 0
        nv_total = int(view["NV"].sum()) if "NV" in view.columns else 0
        total_municipios = df_state["MPIONOM"].nunique()

        m1, m2, m3 = st.columns(3)
        m1.metric("Participación total en el estado", f"{sv_total:,}")
        m2.metric("Abstencionismo total en el estado", f"{nv_total:,}")
        m3.metric("Total de municipios", f"{total_municipios:,}")

        categoria_nombre = GROUPING_TITLES.get(grouping_label, grouping_label)
        title = f"Composición de participación en {state_label} - {year}. Según {categoria_nombre}"

        with st.spinner("Renderizando visualización..."):
            chart = build_participation_chart_altair(
                view,
                title=title,
            )
            st.altair_chart(chart, use_container_width=True)

    st.divider()

    st.subheader("Mapa municipal")

    col_map_controls, col_map_results = st.columns([1, 3], gap="large")

    with col_map_controls:
        with st.container(border=True):
            st.subheader("Ver mapa por")

            map_mode = st.selectbox(
                "Mostrar valores como:",
                options=["Porcentaje", "Total nominal"],
                index=0,
                help="Cambia entre porcentajes y valores absolutos.",
                key="tab_estado_map_mode",
            )

    with col_map_results:
        with st.status("Preparando mapa municipal...", expanded=False) as status:
            gdf_municipios = get_municipal_geometries_cached(str(SHAPEFILE_PATH))
            status.write("Geometrías municipales cargadas")

            state_code_num = int(df_state["EDOCVE"].iloc[0])

            gdf_map = prepare_state_map_data(
                df=df_year,
                gdf=gdf_municipios,
                state_code_num=state_code_num,
            )
            status.write("Datos municipales del mapa preparados")

            status.update(label="Mapa listo", state="complete")

        if map_mode == "Porcentaje":
            col_left, col_right = "sv_ratio", "nv_ratio"
        else:
            col_left, col_right = "SV", "NV"

        map_col1, map_col2 = st.columns(2, gap="large")

        with map_col1:
            fig_left = build_state_choropleth_plotly(
                gdf_map=gdf_map,
                column=col_left,
                state_name=state_label,
                year=int(year),
            )
            st.plotly_chart(fig_left, use_container_width=True)

        with map_col2:
            fig_right = build_state_choropleth_plotly(
                gdf_map=gdf_map,
                column=col_right,
                state_name=state_label,
                year=int(year),
            )
            st.plotly_chart(fig_right, use_container_width=True)

# -------------------------------------------------
# TAB 2: POR AÑOS
# -------------------------------------------------
def render_tab_anios() -> None:
    st.caption(
        "Observa la participación y el abstencionismo por categoría a lo largo del tiempo a nivel nacional o estatal."
    )

    years = tuple(int(y) for y in AVAILABLE_YEARS)
    df_all = get_multi_year_processed_cached(PROJECT_ROOT, years)

    st.subheader("Análisis nacional")

    col_controls_nacional, col_results_nacional = st.columns([1, 3], gap="large")

    with col_controls_nacional:
        with st.container(border=True):
            st.subheader("Filtrar por")

            national_mode = st.selectbox(
                "Modo de análisis",
                options=["Ver país", "Comparación entre estados"],
                index=0,
                help="Elige entre datos agregados a nivel nacional o la comparación entre estados.",
                key="tab_anios_national_mode",
            )

            selected_state_labels = []

            if national_mode == "Ver país":
                national_dimension = st.selectbox(
                    "Categoría:",
                    options=["Sexo", "Edad", "Tipo de sección"],
                    index=0,
                    key="tab_anios_national_dimension",
                )
            else:
                selected_state_labels = st.multiselect(
                    "Selecciona los estados",
                    options=sorted(STATE_LABELS.values()),
                    default=["Aguascalientes"],
                    help="Selecciona uno o varios estados para comparar sus trayectorias.",
                    key="tab_anios_states",
                )

    with col_results_nacional:
        with st.status("Preparando análisis nacional...", expanded=False) as status:
            if national_mode == "Ver país":
                group_col = TIME_SERIES_DIMENSIONS[national_dimension]

                time_view = get_group_year_view(df_all, group_col=group_col)
                status.write(
                    f"Agregación nacional por {national_dimension.lower()} y año completada"
                )

                color_domain = list(time_view["state_code"].astype(str).unique())

            else:
                if len(selected_state_labels) == 0:
                    st.info("Selecciona al menos un estado para mostrar la gráfica.")
                    return

                selected_state_codes = [
                    STATE_CODE_FROM_LABEL[label] for label in selected_state_labels
                ]

                time_view = get_state_year_view(df_all)
                time_view = time_view[
                    time_view["state_code"].isin(selected_state_codes)
                ]

                status.write("Agregación por estado y año completada")
                status.write("Filtro por estados aplicado")

                color_domain = selected_state_codes

            color_range = [
                TIME_SERIES_PALETTE[i % len(TIME_SERIES_PALETTE)]
                for i in range(len(color_domain))
            ]

            status.update(label="Análisis nacional listo", state="complete")

        with st.spinner("Renderizando gráficas..."):
            if national_mode == "Ver país":
                title_part = f"Participación nacional por {national_dimension.lower()}"
                title_nv = f"Abstencionismo a nivel nacional por {national_dimension.lower()}"
            else:
                title_part = "Participación por estado"
                title_nv = "Abstencionismo por estado"

            c1, c2 = build_state_year_charts_altair(
                time_view,
                title_participacion=title_part,
                title_abstencion=title_nv,
                color_domain=color_domain,
                color_range=color_range,
            )

        render_time_series_metrics(time_view)

        st.divider()

        col_part, col_nv = st.columns(2, gap="large")

        with col_part:
            st.subheader("Participación")
            st.altair_chart(c1, use_container_width=True)

        with col_nv:
            st.subheader("Abstencionismo")
            st.altair_chart(c2, use_container_width=True)

        render_centered_legend(time_view, color_domain, color_range)

    st.divider()

    st.subheader("Análisis estatal")

    col_controls_state, col_results_state = st.columns([1, 3], gap="large")

    with col_controls_state:
        with st.container(border=True):
            st.subheader("Filtrar por")

            state_label = st.selectbox(
                "Selecciona un estado",
                options=sorted(STATE_LABELS.values()),
                key="tab_anios_state_section_state",
            )

            state_dimension = st.selectbox(
                "Categoría:",
                options=["Sexo", "Edad", "Tipo de sección"],
                key="tab_anios_state_section_dimension",
            )

            state_code = STATE_CODE_FROM_LABEL[state_label]

    with col_results_state:
        with st.status("Preparando análisis por estado...", expanded=False) as status:
            df_state = df_all[df_all["state_code"] == state_code]

            status.write("Filtro por estado aplicado")

            group_col = TIME_SERIES_DIMENSIONS[state_dimension]

            time_view_state = get_group_year_view(df_state, group_col=group_col)
            status.write(f"Agregación por {state_dimension.lower()} y año completada")

            color_domain = list(time_view_state["state_code"].astype(str).unique())
            color_range = [
                TIME_SERIES_PALETTE[i % len(TIME_SERIES_PALETTE)]
                for i in range(len(color_domain))
            ]

            status.update(label="Análisis por estado listo", state="complete")

        with st.spinner("Renderizando gráficas..."):
            title_part = f"Participación en {state_label} por {state_dimension.lower()}"
            title_nv = f"Abstencionismo en {state_label} por {state_dimension.lower()}"

            c1_state, c2_state = build_state_year_charts_altair(
                time_view_state,
                title_participacion=title_part,
                title_abstencion=title_nv,
                color_domain=color_domain,
                color_range=color_range,
            )

        render_time_series_metrics(time_view_state)

        st.divider()

        col_part, col_nv = st.columns(2, gap="large")

        with col_part:
            st.subheader("Participación")
            st.altair_chart(c1_state, use_container_width=True)

        with col_nv:
            st.subheader("Abstencionismo")
            st.altair_chart(c2_state, use_container_width=True)

        render_centered_legend(time_view_state, color_domain, color_range)

# -------------------------------------------------
# TAB 3: MODELO PREDICTIVO
# -------------------------------------------------
def render_tab_modelo() -> None:
    st.caption(
        "Comparación de modelos predictivos para anticipar la proporción de participación municipal en la siguiente elección."
    )

    with st.status("Ejecutando evaluación del modelo...", expanded=False) as status:
        results_df, summary_df, predictions_df, coef_linear, coef_ridge = run_model_diagnostics(PROJECT_ROOT)
        linear_2027_df, ridge_2027_df = build_forecast_2027_outputs(PROJECT_ROOT)
        gdf_municipios = get_municipal_geometries_cached(str(SHAPEFILE_PATH))
        status.update(label="Diagnósticos listos", state="complete")

    linear_2027_df = add_prediction_interval_from_oos_error(
        linear_2027_df,
        predictions_df,
        model_name="linear_regression",
    )

    ridge_2027_df = add_prediction_interval_from_oos_error(
        ridge_2027_df,
        predictions_df,
        model_name="ridge",
    )

    # -------------------------------------------------
    # INTRODUCCIÓN
    # -------------------------------------------------
    st.markdown(
        """
        En esta sección encontrarás la comparación de tres modelos predictivos para estimar la proporción de participación municipal en la siguiente elección:
        un estimador ingenuo, una regresión lineal y un modelo Ridge. Los modelos se entrenan con información histórica del
        propio municipio y del contexto estatal, usando como regresores la proporción de participación en elecciones anteriores
        y la proporción media de participación del estado.

        El principal hallazgo es que ambos modelos lineales superan al estimador ingenuo y que el error de predicción disminuye
        conforme aumenta el tamaño del electorado municipal. Esto significa que **la participación es más predecible en municipios
        grandes y más volátil en municipios pequeños**, lo cual es consistente con la literatura académica sobre el tema. 

        Aquí están algunas definiciones que te ayudarán a entender más fácilmente esta sección: 

        * **MAE** es el error absoluto promedio, mientras más pequeño, mejor. 
        * **RMSE** penaliza más los errores grandes, por lo que ayuda a identificar modelos que fallan con mayor fuerza en algunos municipios. 
        * **Regresores** son las variables que utiliza el modelo para hacer la predicción. 
        * **Coeficientes** indican cuánto pesa cada uno de los regresores dentro del modelo.
        """
    )

    st.divider()

    # -------------------------------------------------
    # RESUMEN DE MÉTRICAS
    # -------------------------------------------------
    st.subheader("Comparación de modelos")

    c1, c2, c3 = st.columns(3)
    best_row = summary_df.iloc[0]

    c1.metric(
        "Mejor modelo",
        MODEL_LABELS.get(best_row["model"], str(best_row["model"]))
    )
    c2.metric("MAE promedio mínimo", f'{best_row["mae"]:.3f}')
    c3.metric("RMSE promedio mínimo", f'{best_row["rmse"]:.3f}')

    summary_df_display = summary_df.copy()
    summary_df_display["model"] = summary_df_display["model"].map(MODEL_LABELS)

    st.dataframe(summary_df_display, use_container_width=True, hide_index=True)

    st.divider()

    # -------------------------------------------------
    # COEFICIENTES
    # -------------------------------------------------
    st.subheader("Coeficientes de los modelos")

    coef_linear_display = coef_linear.copy()
    coef_linear_display["feature"] = coef_linear_display["feature"].map(FEATURE_LABELS)
    coef_linear_display = coef_linear_display.rename(
        columns={
            "feature": "Regresores",
            "coefficient": "Coeficientes",
        }
    )

    coef_ridge_display = coef_ridge.copy()
    coef_ridge_display["feature"] = coef_ridge_display["feature"].map(FEATURE_LABELS)
    coef_ridge_display = coef_ridge_display.rename(
        columns={
            "feature": "Regresores",
            "coefficient": "Coeficientes",
        }
    )

    col_coef1, col_coef2 = st.columns(2, gap="large")

    with col_coef1:
        st.markdown("**Regresión lineal**")
        st.latex(r"\hat{y}_{t+1} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_3")
        st.dataframe(coef_linear_display, use_container_width=True, hide_index=True)

        fig_linear_2027, _ = plot_forecast_ranked(
            linear_2027_df,
            model_display_name="Regresión lineal",
        )
        st.pyplot(fig_linear_2027, clear_figure=True)

        st.caption(
            "La línea muestra la estimación puntual de participación municipal esperada para 2027. "
            "La banda sombreada representa un intervalo de predicción aproximado construido a partir del error fuera de muestra del modelo."
        )

    with col_coef2:
        st.markdown("**Ridge**")
        st.latex(
            r"\hat{y}_{t+1} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_3,\quad \mathrm{con\ penalización}\ \lambda \sum_{j=1}^{p}\beta_j^2"
        )
        st.dataframe(coef_ridge_display, use_container_width=True, hide_index=True)

        fig_ridge_2027, _ = plot_forecast_ranked(
            ridge_2027_df,
            model_display_name="Ridge",
        )
        st.pyplot(fig_ridge_2027, clear_figure=True)

        st.caption(
            "La línea muestra la estimación puntual de participación municipal esperada para 2027. "
            "La banda sombreada representa un intervalo de predicción aproximado construido a partir del error fuera de muestra del modelo."
        )

    st.divider()

    # -------------------------------------------------
    # DIAGNÓSTICOS VISUALES
    # -------------------------------------------------
    st.subheader("Diagnósticos visuales")
    st.caption(
        "Los siguientes gráficos muestran la relación entre los valores reales y predichos, la distribución de los errores y el mapa de errores en la estimación para 2021. Su objetivo es facilitar la comparación el performance entre los modelos."
    )

    diag_tab_linear, diag_tab_ridge = st.tabs(["Regresión lineal", "Ridge"])

    with diag_tab_linear:
        selected_model = "linear_regression"
        selected_year = 2021

        fig_pred_real, _ = plot_predicted_vs_real(
            predictions_df,
            model_name=selected_model,
            test_year=selected_year,
        )

        fig_error_dist, _ = plot_error_distribution(
            predictions_df,
            model_name=selected_model,
            test_year=selected_year,
        )

        gdf_error = build_error_map_frame(
            predictions_df=predictions_df,
            gdf_municipios=gdf_municipios,
            model_name=selected_model,
            test_year=selected_year,
        )

        fig_map_error = build_error_choropleth_plotly(
            gdf_error,
            color_col="error",
            title="Error municipal - Regresión lineal - 2021",
        )

        fig_map_abs_error = build_error_choropleth_plotly(
            gdf_error,
            color_col="abs_error",
            title="Error absoluto municipal - Regresión lineal - 2021",
        )

        row1_col1, row1_col2 = st.columns(2, gap="large")
        row2_col1, row2_col2 = st.columns(2, gap="large")

        with row1_col1:
            st.pyplot(fig_pred_real, clear_figure=True)
            st.caption(
                "Este gráfico compara el valor real observado con el valor predicho por el modelo. "
                "Mientras más cerca estén los puntos de la diagonal, mejor es el ajuste. Los valores para los municipios más pequeños son los que más se alejan de la diagonal, es decir, los que tienen el peor ajuste en el modelo."
            )

        with row1_col2:
            st.pyplot(fig_error_dist, clear_figure=True)
            st.caption(
                "Este histograma muestra cómo se distribuyen los errores de predicción. "
                "Un modelo bien calibrado tiende a concentrar sus errores cerca de cero y tener una distribución simétrica."
            )

        with row2_col1:
            st.plotly_chart(fig_map_error, use_container_width=True)
            st.caption(
                "Este mapa representa el error con signo (sobre o subestimación). Valores positivos indican sobreestimación del modelo "
                "y valores negativos subestimación de la participación."
            )

        with row2_col2:
            st.plotly_chart(fig_map_abs_error, use_container_width=True)
            st.caption(
                "Este mapa muestra la magnitud del error sin importar su signo. "
                "Sirve para identificar los municipios donde el modelo falla más en su predicción."
            )

    with diag_tab_ridge:
        selected_model = "ridge"
        selected_year = 2021

        fig_pred_real, _ = plot_predicted_vs_real(
            predictions_df,
            model_name=selected_model,
            test_year=selected_year,
        )

        fig_error_dist, _ = plot_error_distribution(
            predictions_df,
            model_name=selected_model,
            test_year=selected_year,
        )

        gdf_error = build_error_map_frame(
            predictions_df=predictions_df,
            gdf_municipios=gdf_municipios,
            model_name=selected_model,
            test_year=selected_year,
        )

        fig_map_error = build_error_choropleth_plotly(
            gdf_error,
            color_col="error",
            title="Error municipal - Ridge - 2021",
        )

        fig_map_abs_error = build_error_choropleth_plotly(
            gdf_error,
            color_col="abs_error",
            title="Error absoluto municipal - Ridge - 2021",
        )

        row1_col1, row1_col2 = st.columns(2, gap="large")
        row2_col1, row2_col2 = st.columns(2, gap="large")

        with row1_col1:
            st.pyplot(fig_pred_real, clear_figure=True)
            st.caption(
                "Este gráfico compara el valor real observado con el valor predicho por el modelo. "
                "Mientras más cerca estén los puntos de la diagonal, mejor es el ajuste."
            )

        with row1_col2:
            st.pyplot(fig_error_dist, clear_figure=True)
            st.caption(
                "Este histograma muestra cómo se distribuyen los errores de predicción. "
                "Un modelo bien calibrado tiende a concentrar sus errores cerca de cero."
            )

        with row2_col1:
            st.plotly_chart(fig_map_error, use_container_width=True)
            st.caption(
                "Este mapa representa el error con signo. Valores positivos indican sobreestimación del modelo "
                "y valores negativos subestimación."
            )

        with row2_col2:
            st.plotly_chart(fig_map_abs_error, use_container_width=True)
            st.caption(
                "Este mapa muestra la magnitud del error sin importar su signo. "
                "Sirve para identificar los municipios donde el modelo falla más."
            )

    st.divider()

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
def render_footer() -> None:
    st.divider()
    st.markdown(
        '<p class="footer" style="text-align: center;">'
        "Este Explorador de Participación Electoral es un proyecto de mi portafolio de trabajo "
        "como analista de datos. Es de elaboración propia con base en datos públicos del Instituto Nacional Electoral (INE). "
        "Para conocer más visita mi Github"
        "</p>",
        unsafe_allow_html=True,
    )

# -------------------------------------------------
# APP
# -------------------------------------------------
render_header()

tab_estado, tab_anios, tab_modelo = st.tabs(
    ["Por estado", "Por años", "Modelo predictivo"]
)

with tab_estado:
    render_tab_estado()

with tab_anios:
    render_tab_anios()

with tab_modelo:
    render_tab_modelo()

render_footer()