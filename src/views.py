from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import altair as alt

from src.constants import STATE_LABELS, GROUPINGS
from src.aggregations import (
    aggregate_group,
    aggregate_group_year,
    aggregate_state_year,
)


def get_view(
    df: pd.DataFrame,
    grouping_label: str,
) -> pd.DataFrame:
    """
    Devuelve un DataFrame listo para graficar en la pestaña Por estado.
    """
    group_col = GROUPINGS.get(grouping_label)

    if group_col is None:
        raise ValueError(
            f"Grouping inválido: {grouping_label}. Opciones: {list(GROUPINGS.keys())}"
        )

    agg = aggregate_group(df, group_col)

    out = agg.reset_index().rename(columns={group_col: "categoria"})
    out["categoria"] = out["categoria"].astype("string")

    cols = [
    "categoria",
    "SV",
    "NV",
    "NS",
    "LN",
    "total_marked",
    "sv_ratio",
    "nv_ratio",
    "ns_ratio",
]

    return out[cols]


def limit_categories(
    view: pd.DataFrame,
    top_n: int | None,
    by: str = "SV",
    others: bool = True,
) -> pd.DataFrame:
    if top_n is None or top_n <= 0:
        return view

    if by not in view.columns:
        raise ValueError(f"Columna inválida para ordenar: {by}")

    ordered = view.sort_values(by, ascending=False)

    top = ordered.head(top_n)
    remainder = ordered.iloc[top_n:]

    if remainder.empty or not others:
        return top

    sv_sum = remainder["SV"].sum()
    nv_sum = remainder["NV"].sum()
    ns_sum = remainder["NS"].sum()
    ln_sum = remainder["LN"].sum()

    others_row = {
        "categoria": "Otros",
        "SV": sv_sum,
        "NV": nv_sum,
        "NS": ns_sum,
        "LN": ln_sum,
        "sv_ratio": sv_sum / ln_sum if ln_sum > 0 else 0,
        "nv_ratio": nv_sum / ln_sum if ln_sum > 0 else 0,
        "ns_ratio": ns_sum / ln_sum if ln_sum > 0 else 0,
    }

    others_df = pd.DataFrame([others_row])

    return pd.concat([top, others_df], ignore_index=True)


def get_participation_view_for_chart(
    df: pd.DataFrame,
    grouping_label: str,
    top_n: int | None = None,
) -> pd.DataFrame:
    view = get_view(df, grouping_label=grouping_label)

    if grouping_label == "Mostrar por municipio":
        view = limit_categories(view, top_n=top_n, by="SV", others=True)

    return view



def get_group_year_view(
    df: pd.DataFrame,
    group_col: str,
) -> pd.DataFrame:
    """
    View model para agrupación por dimensión y año para series de tiempo.
    """
    if group_col == "state_code":
        out = aggregate_state_year(df)
        out["state_label"] = out["state_code"].map(STATE_LABELS).fillna(
            out["state_code"].astype(str)
        )
    else:
        out = aggregate_group_year(df, group_col=group_col)
        out = out.rename(columns={group_col: "state_code"})
        out["state_label"] = out["state_code"].astype(str)

    out["state_code"] = out["state_code"].astype("string")
    out["year"] = out["year"].astype("int16")

    cols = [
        "state_code",
        "state_label",
        "year",
        "SV",
        "NV",
        "NS",
        "LN",
        "total_marked",
        "sv_ratio",
        "nv_ratio",
        "ns_ratio",
    ]

    return out[cols].sort_values(["state_code", "year"]).reset_index(drop=True)



def get_state_year_view(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wrapper específico para series temporales por estado.
    """
    return get_group_year_view(df, group_col="state_code")


def build_participation_figure_matplotlib(
    view: pd.DataFrame,
    include_abstentions: bool = False,
    categoria_nombre: str = "categoría",
    title: str | None = None,
    figsize: tuple[int, int] = (14, 7),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Devuelve una figura y ejes de Matplotlib a partir del view.
    """
    if title is None:
        title = f"% Participación electoral por {categoria_nombre}"

    df_plot = view.copy().sort_values("SV", ascending=False)
    categorias = df_plot["categoria"].astype(str).tolist()
    sv_pct = (df_plot["sv_ratio"] * 100).to_numpy()
    sv_nom = df_plot["SV"].to_numpy()

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("white")

    if not include_abstentions:
        ax.barh(categorias, sv_pct, color="#8fbc8f", edgecolor="none", linewidth=0)

        for i, (pct, nom) in enumerate(zip(sv_pct, sv_nom)):
            ax.text(pct + 0.5, i, f"{pct:.1f}% ({nom:,})", va="center", ha="left", fontsize=10)

        ax.set_xlabel("% Participación sobre lista nominal")

    else:
        nv_pct = (df_plot["nv_ratio"] * 100).to_numpy()
        nv_nom = df_plot["NV"].to_numpy()

        ax.barh(categorias, sv_pct, color="#8fbc8f", edgecolor="none", linewidth=0, label="Participación")
        ax.barh(categorias, nv_pct, left=sv_pct, color="#CD5C5C", edgecolor="none", linewidth=0, label="Abstencionismo")

        for i, (svp, nvp, svn, nvn) in enumerate(zip(sv_pct, nv_pct, sv_nom, nv_nom)):
            if svp >= 8:
                ax.text(svp / 2, i, f"{svp:.1f}%\n({svn:,})", va="center", ha="center", fontsize=9)
            else:
                ax.text(svp + 0.5, i, f"{svp:.1f}% ({svn:,})", va="center", ha="left", fontsize=9)

            if nvp >= 8:
                ax.text(svp + nvp / 2, i, f"{nvp:.1f}%\n({nvn:,})", va="center", ha="center", fontsize=9)
            else:
                ax.text(svp + nvp + 0.5, i, f"{nvp:.1f}% ({nvn:,})", va="center", ha="left", fontsize=9)

        ax.set_xlabel("% sobre lista nominal")
        max_total = (sv_pct + nv_pct).max()
        ax.set_xlim(0, max(100, max_total + 5))
        ax.legend(loc="lower right", frameon=False)

    for spine in ["right", "top", "left"]:
        ax.spines[spine].set_visible(False)

    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(axis="y", labelsize=9)
    ax.set_title(title, pad=12, fontweight="bold")
    fig.tight_layout()

    return fig, ax


def build_participation_chart_altair(
    view: pd.DataFrame,
    title: str | None = None,
) -> alt.Chart:
    """
    Renderer Altair para el explorador por estado.
    Muestra una sola barra apilada por categoría:
    - SV (verde)
    - NV (rojo)
    - NS (gris)
    Todas sobre lista nominal.
    """
    sv_total_all = int(view["SV"].sum()) if "SV" in view.columns else 0
    nv_total_all = int(view["NV"].sum()) if "NV" in view.columns else 0
    ns_total_all = int(view["NS"].sum()) if "NS" in view.columns else 0
    ln_total_all = int(view["LN"].sum()) if "LN" in view.columns else 0

    sv_total_pct = (sv_total_all / ln_total_all * 100) if ln_total_all > 0 else None
    nv_total_pct = (nv_total_all / ln_total_all * 100) if ln_total_all > 0 else None
    ns_total_pct = (ns_total_all / ln_total_all * 100) if ln_total_all > 0 else None

    df_plot = view.copy()
    df_plot["sv_pct"] = (df_plot["sv_ratio"] * 100).fillna(0)
    df_plot["nv_pct"] = (df_plot["nv_ratio"] * 100).fillna(0)
    df_plot["ns_pct"] = (df_plot["ns_ratio"] * 100).fillna(0)

    if title is None:
        title = "% Participación sobre lista nominal"

    subtitle_lines = []
    if ln_total_all > 0:
        subtitle_lines.append(
            f"SV: {sv_total_all:,} ({sv_total_pct:.1f}%)  |  "
            f"NV: {nv_total_all:,} ({nv_total_pct:.1f}%)  |  "
            f"NS: {ns_total_all:,} ({ns_total_pct:.1f}%)"
        )

    title_param = alt.TitleParams(text=title, subtitle=subtitle_lines)

    long = df_plot.melt(
        id_vars=["categoria", "SV", "NV", "NS"],
        value_vars=["sv_pct", "nv_pct", "ns_pct"],
        var_name="serie_raw",
        value_name="pct",
    )

    long["serie"] = long["serie_raw"].map(
        {
            "sv_pct": "Participación",
            "nv_pct": "Abstencionismo",
            "ns_pct": "Registros inválidos",
        }
    )

    long["total_serie"] = long.apply(
        lambda r: (
            r["SV"] if r["serie"] == "Participación"
            else r["NV"] if r["serie"] == "Abstencionismo"
            else r["NS"]
        ),
        axis=1,
    )

    long["serie_order"] = long["serie"].map(
        {"Participación": 0, "Abstencionismo": 1, "Registros inválidos": 2}
    ).astype("int8")

    chart = (
        alt.Chart(long, title=title_param)
        .mark_bar()
        .encode(
            y=alt.Y(
                "categoria:N",
                sort=alt.SortField(field="SV", order="descending"),
                title=None,
            ),
            x=alt.X(
                "pct:Q",
                stack="zero",
                title="% sobre lista nominal",
            ),
            color=alt.Color(
                "serie:N",
                title=None,
                sort=["Participación", "Abstencionismo", "Registros inválidos"],
                scale=alt.Scale(
                    domain=["Participación", "Abstencionismo", "Registros inválidos"],
                    range=["#8fbc8f", "#CD5C5C", "#9E9E9E"],
                ),
                legend=alt.Legend(orient="bottom"),
            ),
            order=alt.Order("serie_order:Q", sort="ascending"),
            tooltip=[
                alt.Tooltip("categoria:N", title="Serie"),
                alt.Tooltip("serie:N", title="Valor"),
                alt.Tooltip("pct:Q", title="%", format=".1f"),
                alt.Tooltip("total_serie:Q", title="Total", format=","),
            ],
        )
    )

    return chart.interactive()



def build_state_year_charts_altair(
    state_year_view: pd.DataFrame,
    title_participacion: str = "Participación por estado",
    title_abstencion: str = "Abstencionismo por estado",
    color_domain: list[str] | None = None,
    color_range: list[str] | None = None,
) -> tuple[alt.Chart, alt.Chart]:
    """
    Devuelve dos charts Altair para series temporales por grupo.
    """
    df = state_year_view.copy()

    if "state_label" not in df.columns and "state_code" in df.columns:
        df["state_label"] = df["state_code"].map(STATE_LABELS).fillna(
            df["state_code"].astype(str)
        )

    color_enc = alt.Color(
        "state_code:N",
        title=None,
        legend=None,
        scale=alt.Scale(domain=color_domain, range=color_range)
        if (color_domain and color_range)
        else alt.Undefined,
    )

    base = alt.Chart(df).encode(
        x=alt.X("year:O", title="Año"),
        color=color_enc,
        tooltip=[
            alt.Tooltip("state_label:N", title="Estado"),
            alt.Tooltip("year:O", title="Año"),
        ],
    )

    chart_participacion = (
        base.properties(title=alt.TitleParams(text=title_participacion))
        .mark_line(point=True)
        .encode(
            y=alt.Y(
                "sv_ratio:Q",
                title="%",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format=".0%"),
            ),
            tooltip=[
                alt.Tooltip("state_label:N", title="Estado"),
                alt.Tooltip("year:O", title="Año"),
                alt.Tooltip("sv_ratio:Q", title="% Participación", format=".2%"),
                alt.Tooltip("SV:Q", title="Participación total", format=","),
            ],
        )
        .interactive()
    )

    chart_abstencion = (
        base.properties(title=alt.TitleParams(text=title_abstencion))
        .mark_line(point=True)
        .encode(
            y=alt.Y(
                "nv_ratio:Q",
                title="%",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format=".0%"),
            ),
            tooltip=[
                alt.Tooltip("state_label:N", title="Estado"),
                alt.Tooltip("year:O", title="Año"),
                alt.Tooltip("nv_ratio:Q", title="% Abstencionismo", format=".2%"),
                alt.Tooltip("NV:Q", title="Abstención total", format=","),
            ],
        )
        .interactive()
    )

    return chart_participacion, chart_abstencion


def build_participation_figure(*args, **kwargs):
    """
    Wrapper para no romper imports existentes.
    """
    return build_participation_figure_matplotlib(*args, **kwargs)