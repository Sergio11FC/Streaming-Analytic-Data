#Nathalie Wilches Tamayo - Sergio Andrés Fernandez Castro

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
import io

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Streaming Financial Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paleta corporativa
BRAND_COLORS = {
    "Netflix": "#E50914",
    "Amazon": "#2E8B57",
    "Disney": "#1E90FF",
    "HBO":    "#6A0DAD",
}
ACCENT       = "#00D4FF"
BG_DARK      = "#0E1117"
CARD_BG      = "#1C1E26"
TEXT_PRIMARY = "#FFFFFF"
TEXT_MUTED   = "#A0AEC0"

# ─────────────────────────────────────────────
# CSS PERSONALIZADO
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Fondo y fuentes */
    .stApp { background-color: #0E1117; color: #FFFFFF; }
    .stApp h1, .stApp h2, .stApp h3 { color: #FFFFFF; }

    /* Tarjetas KPI */
    .kpi-card {
        background: linear-gradient(135deg, #1C1E26 0%, #252836 100%);
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s;
    }
    .kpi-card:hover { transform: translateY(-2px); }
    .kpi-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #A0AEC0;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 26px;
        font-weight: 800;
        color: #00D4FF;
        line-height: 1.1;
    }
    .kpi-delta {
        font-size: 12px;
        margin-top: 4px;
    }
    .kpi-delta.pos { color: #48BB78; }
    .kpi-delta.neg { color: #FC8181; }

    /* Tarjetas Insight */
    .insight-card {
        background: linear-gradient(135deg, #1C1E26 0%, #1A1D27 100%);
        border-left: 4px solid #00D4FF;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    .insight-title {
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #00D4FF;
        margin-bottom: 4px;
    }
    .insight-text { font-size: 14px; color: #E2E8F0; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #141620;
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    section[data-testid="stSidebar"] * { color: #E2E8F0 !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1C1E26;
        border-radius: 10px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #A0AEC0 !important;
        font-weight: 600;
        font-size: 13px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00D4FF22 !important;
        color: #00D4FF !important;
    }

    /* Selectbox / Multiselect */
    .stMultiSelect [data-baseweb="tag"] { background-color: #00D4FF22; }

    /* Upload area */
    .uploadedFile { background-color: #1C1E26 !important; }

    /* Separadores */
    hr { border-color: rgba(255,255,255,0.08); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# COLUMNAS ESPERADAS
# ─────────────────────────────────────────────
REQUIRED_COLS = [
    "Platform", "Year", "Quarter",
    "Total_Revenue", "Gross_Profit", "Operating_Income",
    "EBITDA", "Net_Income", "Operating_Expense",
    "Pretax_Income", "Tax_Provision",
    "Basic_EPS", "Diluted_EPS",
]

# ─────────────────────────────────────────────
# UTILIDADES DE FORMATO
# ─────────────────────────────────────────────
def fmt_millions(v: float, decimals: int = 1) -> str:
    """Formatea números grandes en M/B."""
    if pd.isna(v):
        return "N/A"
    abs_v = abs(v)
    sign  = "-" if v < 0 else ""
    if abs_v >= 1_000_000:
        return f"{sign}${abs_v/1_000_000:.{decimals}f}B"
    if abs_v >= 1_000:
        return f"{sign}${abs_v/1_000:.{decimals}f}M"
    return f"{sign}${abs_v:.{decimals}f}"

def fmt_pct(v: float, decimals: int = 1) -> str:
    if pd.isna(v):
        return "N/A"
    return f"{v:.{decimals}f}%"

def delta_class(v: float) -> str:
    return "pos" if v >= 0 else "neg"

def delta_arrow(v: float) -> str:
    return "▲" if v >= 0 else "▼"

# ─────────────────────────────────────────────
# CARGA Y VALIDACIÓN DE DATOS
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(file_bytes: bytes) -> pd.DataFrame:
    """Carga, valida y preprocesa el CSV."""
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as e:
        raise ValueError(f"No se pudo leer el archivo CSV: {e}")

    # Normalizar nombres de columnas
    df.columns = [c.strip().replace(" ", "_").replace("-", "_") for c in df.columns]

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Columnas faltantes en el CSV: {', '.join(missing)}\n\n"
            f"Columnas requeridas: {', '.join(REQUIRED_COLS)}"
        )

    numeric_cols = [c for c in REQUIRED_COLS if c not in ("Platform", "Year", "Quarter")]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")

    df["Year"]    = pd.to_numeric(df["Year"],    errors="coerce").astype("Int64")
    df["Quarter"] = pd.to_numeric(df["Quarter"], errors="coerce").astype("Int64")
    df["Platform"] = df["Platform"].str.strip()

    # Período legible
    df["Period"] = df["Year"].astype(str) + "-Q" + df["Quarter"].astype(str)
    df = df.sort_values(["Platform", "Year", "Quarter"]).reset_index(drop=True)

    # ── Métricas derivadas ──────────────────────
    df["Gross_Margin_Pct"]    = df["Gross_Profit"]    / df["Total_Revenue"] * 100
    df["Operating_Margin_Pct"]= df["Operating_Income"]/ df["Total_Revenue"] * 100
    df["EBITDA_Margin_Pct"]   = df["EBITDA"]          / df["Total_Revenue"] * 100
    df["OpEx_Ratio_Pct"]      = df["Operating_Expense"]/ df["Total_Revenue"] * 100
    df["Effective_Tax_Rate"]  = df["Tax_Provision"]   / df["Pretax_Income"]  * 100

    # Crecimiento QoQ por plataforma
    for col, new in [("Total_Revenue", "QoQ_Revenue_Growth"),
                     ("Net_Income",    "Net_Income_Growth")]:
        df[new] = df.groupby("Platform")[col].pct_change() * 100

    # Revenue Share por período
    tot_rev = df.groupby("Period")["Total_Revenue"].transform("sum")
    df["Revenue_Share_Pct"] = df["Total_Revenue"] / tot_rev * 100

    return df


def validate_upload(file) -> pd.DataFrame | None:
    """Valida y devuelve el DataFrame o muestra errores."""
    if file is None:
        return None
    try:
        df = load_data(file.getvalue())
        return df
    except ValueError as e:
        st.error(f"❌ Error de validación:\n\n{e}")
        return None
    except Exception as e:
        st.error(f"❌ Error inesperado al procesar el archivo:\n\n{e}")
        return None


# ─────────────────────────────────────────────
# FILTROS (sidebar)
# ─────────────────────────────────────────────
def apply_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Renderiza filtros en sidebar y devuelve df filtrado."""
    st.sidebar.markdown("## 🎛️ Filtros")
    st.sidebar.markdown("---")

    # Plataformas
    platforms_available = sorted(df["Platform"].unique().tolist())
    selected_platforms = st.sidebar.multiselect(
        "📺 Plataforma",
        options=platforms_available,
        default=platforms_available,
        help="Selecciona una o más plataformas para comparar.",
    )
    if not selected_platforms:
        selected_platforms = platforms_available

    # Años
    years_available = sorted(df["Year"].dropna().unique().tolist())
    selected_years  = st.sidebar.multiselect(
        "📅 Año",
        options=years_available,
        default=years_available,
    )
    if not selected_years:
        selected_years = years_available

    # Trimestres
    quarters_available = sorted(df["Quarter"].dropna().unique().tolist())
    selected_quarters  = st.sidebar.multiselect(
        "🗓️ Trimestre",
        options=[f"Q{q}" for q in quarters_available],
        default=[f"Q{q}" for q in quarters_available],
    )
    if not selected_quarters:
        selected_quarters = [f"Q{q}" for q in quarters_available]
    q_nums = [int(q[1:]) for q in selected_quarters]

    filtered = df[
        df["Platform"].isin(selected_platforms) &
        df["Year"].isin(selected_years) &
        df["Quarter"].isin(q_nums)
    ].copy()

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"**{len(filtered):,} registros** de {len(df):,} totales"
    )
    return filtered


# ─────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────
def render_kpi_cards(df: pd.DataFrame):
    """Muestra tarjetas KPI en la parte superior."""
    total_rev     = df["Total_Revenue"].sum()
    total_ebitda  = df["EBITDA"].sum()
    total_net_inc = df["Net_Income"].sum()
    avg_gross_m   = df["Gross_Margin_Pct"].mean()
    avg_op_m      = df["Operating_Margin_Pct"].mean()

    leader        = df.groupby("Platform")["Revenue_Share_Pct"].mean().idxmax() \
                    if not df.empty else "N/A"
    leader_share  = df.groupby("Platform")["Revenue_Share_Pct"].mean().max() \
                    if not df.empty else 0.0

    cols = st.columns(6)
    kpis = [
        ("Total Revenue",    fmt_millions(total_rev),         None),
        ("EBITDA",           fmt_millions(total_ebitda),       None),
        ("Net Income",       fmt_millions(total_net_inc),      None),
        ("Gross Margin",     fmt_pct(avg_gross_m),             None),
        ("Operating Margin", fmt_pct(avg_op_m),                None),
        ("Rev. Share Leader",f"{leader} ({fmt_pct(leader_share)})", None),
    ]
    for col, (label, value, delta) in zip(cols, kpis):
        with col:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────
# HELPER: template de gráfico oscuro
# ─────────────────────────────────────────────
DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#1C1E26",
    plot_bgcolor ="#1C1E26",
    font=dict(family="Inter, sans-serif", color="#E2E8F0"),
    title_font_size=15,
    title_font_color="#FFFFFF",
    legend=dict(
        bgcolor="rgba(0,0,0,0.3)",
        bordercolor="rgba(255,255,255,0.1)",
        borderwidth=1,
    ),
    margin=dict(l=50, r=30, t=50, b=50),
)

def get_color(platform: str) -> str:
    return BRAND_COLORS.get(platform, ACCENT)


# ─────────────────────────────────────────────
# PESTAÑA 1: RENTABILIDAD Y EFICIENCIA
# ─────────────────────────────────────────────
def tab_rentabilidad(df: pd.DataFrame):
    st.markdown("### 💰 Rentabilidad y Eficiencia Operativa")
    if df.empty:
        st.warning("Sin datos para los filtros seleccionados.")
        return

    metric_opts = {
        "Margen Bruto (%)":     "Gross_Margin_Pct",
        "Margen Operativo (%)": "Operating_Margin_Pct",
        "Margen EBITDA (%)":    "EBITDA_Margin_Pct",
    }

    pivot_list = []
    for label, col in metric_opts.items():
        piv = df.groupby(["Period", "Platform"])[col].mean().reset_index()
        piv["Metric"] = label
        piv.rename(columns={col: "Value"}, inplace=True)
        pivot_list.append(piv)
    data_long = pd.concat(pivot_list, ignore_index=True)

    # ── Líneas temporales ─────────────────────
    cols_top = st.columns(3)
    for idx, (label, col) in enumerate(metric_opts.items()):
        grp = df.groupby(["Period", "Platform"])[col].mean().reset_index()
        fig = go.Figure()
        for plat in grp["Platform"].unique():
            sub = grp[grp["Platform"] == plat]
            fig.add_trace(go.Scatter(
                x=sub["Period"], y=sub[col],
                mode="lines+markers",
                name=plat,
                line=dict(color=get_color(plat), width=2.5),
                marker=dict(size=7),
            ))
        fig.update_layout(
            **DARK_LAYOUT,
            title=label,
            yaxis_title="%",
            hovermode="x unified",
        )
        with cols_top[idx]:
            st.plotly_chart(fig, use_container_width=True)

    # ── Barras agrupadas: último período ──────
    latest = df.sort_values(["Year", "Quarter"]).groupby("Platform").last().reset_index()
    fig_bar = go.Figure()
    for metric, col in metric_opts.items():
        fig_bar.add_trace(go.Bar(
            x=latest["Platform"],
            y=latest[col],
            name=metric,
            text=[f"{v:.1f}%" for v in latest[col]],
            textposition="outside",
        ))
    fig_bar.update_layout(
        **DARK_LAYOUT,
        title="Comparación de Márgenes — Último Período Disponible",
        barmode="group",
        yaxis_title="%",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Heatmap ───────────────────────────────
    st.markdown("#### 🔥 Heatmap de Margen EBITDA por Plataforma y Período")
    heat_df = df.groupby(["Platform", "Period"])["EBITDA_Margin_Pct"].mean().reset_index()
    heat_pivot = heat_df.pivot(index="Platform", columns="Period", values="EBITDA_Margin_Pct")
    fig_heat = px.imshow(
        heat_pivot,
        color_continuous_scale="RdYlGn",
        text_auto=".1f",
        aspect="auto",
    )
    fig_heat.update_layout(
        **DARK_LAYOUT,
        title="EBITDA Margin (%) — Heatmap",
        xaxis_title="Período",
        yaxis_title="",
    )
    st.plotly_chart(fig_heat, use_container_width=True)


# ─────────────────────────────────────────────
# PESTAÑA 2: CRECIMIENTO Y TENDENCIAS
# ─────────────────────────────────────────────
def tab_crecimiento(df: pd.DataFrame):
    st.markdown("### 📈 Crecimiento y Tendencias")
    if df.empty:
        st.warning("Sin datos para los filtros seleccionados.")
        return

    cols_top = st.columns(2)

    # ── QoQ Revenue Growth ────────────────────
    with cols_top[0]:
        grp = df.groupby(["Period", "Platform"])["QoQ_Revenue_Growth"].mean().reset_index()
        fig = px.bar(
            grp, x="Period", y="QoQ_Revenue_Growth",
            color="Platform", barmode="group",
            color_discrete_map=BRAND_COLORS,
            title="Crecimiento QoQ de Ingresos (%)",
            labels={"QoQ_Revenue_Growth": "%", "Period": ""},
        )
        fig.update_layout(**DARK_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    # ── Net Income Growth ────────────────────
    with cols_top[1]:
        grp2 = df.groupby(["Period", "Platform"])["Net_Income_Growth"].mean().reset_index()
        fig2 = px.bar(
            grp2, x="Period", y="Net_Income_Growth",
            color="Platform", barmode="group",
            color_discrete_map=BRAND_COLORS,
            title="Crecimiento de Utilidad Neta QoQ (%)",
            labels={"Net_Income_Growth": "%", "Period": ""},
        )
        fig2.update_layout(**DARK_LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Línea de ingresos acumulados ──────────
    rev_ts = df.groupby(["Period", "Platform"])["Total_Revenue"].sum().reset_index()
    fig3 = go.Figure()
    for plat in rev_ts["Platform"].unique():
        sub = rev_ts[rev_ts["Platform"] == plat]
        fig3.add_trace(go.Scatter(
            x=sub["Period"], y=sub["Total_Revenue"],
            mode="lines+markers", name=plat,
            line=dict(color=get_color(plat), width=2.5),
            marker=dict(size=7),
        ))
    fig3.update_layout(
        **DARK_LAYOUT,
        title="Evolución de Ingresos Totales por Plataforma",
        yaxis_title="Revenue",
        hovermode="x unified",
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ── Scatter: Revenue vs Net Income ────────
    agg = df.groupby("Platform").agg(
        Avg_Revenue=("Total_Revenue", "mean"),
        Avg_Net_Income=("Net_Income",  "mean"),
        Avg_EBITDA_Margin=("EBITDA_Margin_Pct", "mean"),
    ).reset_index()
    fig4 = px.scatter(
        agg, x="Avg_Revenue", y="Avg_Net_Income",
        size="Avg_EBITDA_Margin",
        color="Platform",
        color_discrete_map=BRAND_COLORS,
        text="Platform",
        title="Revenue Promedio vs. Utilidad Neta Promedio (tamaño = EBITDA Margin)",
        labels={"Avg_Revenue": "Revenue Promedio", "Avg_Net_Income": "Net Income Promedio"},
    )
    fig4.update_traces(textposition="top center")
    fig4.update_layout(**DARK_LAYOUT)
    st.plotly_chart(fig4, use_container_width=True)


# ─────────────────────────────────────────────
# PESTAÑA 3: ESTRUCTURA Y CONTROL DE COSTOS
# ─────────────────────────────────────────────
def tab_costos(df: pd.DataFrame):
    st.markdown("### 🏗️ Estructura y Control de Costos")
    if df.empty:
        st.warning("Sin datos para los filtros seleccionados.")
        return

    cols_top = st.columns(2)

    # ── OpEx Ratio ───────────────────────────
    with cols_top[0]:
        grp = df.groupby(["Period", "Platform"])["OpEx_Ratio_Pct"].mean().reset_index()
        fig = px.line(
            grp, x="Period", y="OpEx_Ratio_Pct",
            color="Platform",
            color_discrete_map=BRAND_COLORS,
            title="Ratio de Gasto Operativo / Ingresos (%)",
            labels={"OpEx_Ratio_Pct": "%", "Period": ""},
            markers=True,
        )
        fig.update_layout(**DARK_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    # ── Tasa Impositiva Efectiva ──────────────
    with cols_top[1]:
        grp2 = df.groupby(["Period", "Platform"])["Effective_Tax_Rate"].mean().reset_index()
        fig2 = px.area(
            grp2, x="Period", y="Effective_Tax_Rate",
            color="Platform",
            color_discrete_map=BRAND_COLORS,
            title="Tasa Impositiva Efectiva por Período (%)",
            labels={"Effective_Tax_Rate": "%", "Period": ""},
        )
        fig2.update_layout(**DARK_LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Barras apiladas: composición de costos ─
    st.markdown("#### 📊 Composición Financiera — Últimos Datos por Plataforma")
    latest = df.sort_values(["Year", "Quarter"]).groupby("Platform").last().reset_index()
    cost_df = pd.melt(
        latest,
        id_vars="Platform",
        value_vars=["Gross_Profit", "Operating_Expense", "EBITDA", "Net_Income"],
        var_name="Component", value_name="Value",
    )
    component_labels = {
        "Gross_Profit": "Gross Profit",
        "Operating_Expense": "OpEx",
        "EBITDA": "EBITDA",
        "Net_Income": "Net Income",
    }
    cost_df["Component"] = cost_df["Component"].map(component_labels)
    fig3 = px.bar(
        cost_df, x="Platform", y="Value", color="Component",
        barmode="group",
        title="Componentes Financieros Clave — Último Período",
        labels={"Value": "Monto", "Platform": ""},
        color_discrete_sequence=px.colors.qualitative.Vivid,
    )
    fig3.update_layout(**DARK_LAYOUT)
    st.plotly_chart(fig3, use_container_width=True)

    # ── Heatmap OpEx ─────────────────────────
    heat_df = df.groupby(["Platform", "Period"])["OpEx_Ratio_Pct"].mean().reset_index()
    if not heat_df.empty:
        heat_pivot = heat_df.pivot(index="Platform", columns="Period", values="OpEx_Ratio_Pct")
        fig4 = px.imshow(
            heat_pivot, color_continuous_scale="RdYlGn_r",
            text_auto=".1f", aspect="auto",
        )
        fig4.update_layout(
            **DARK_LAYOUT,
            title="Heatmap: Ratio OpEx (%) — menor es mejor",
            xaxis_title="Período", yaxis_title="",
        )
        st.plotly_chart(fig4, use_container_width=True)


# ─────────────────────────────────────────────
# PESTAÑA 4: BENCHMARKING COMPETITIVO
# ─────────────────────────────────────────────
def tab_benchmarking(df: pd.DataFrame):
    st.markdown("### ⚔️ Benchmarking Competitivo")
    if df.empty:
        st.warning("Sin datos para los filtros seleccionados.")
        return

    # ── Revenue Share ─────────────────────────
    share_ts = df.groupby(["Period", "Platform"])["Revenue_Share_Pct"].mean().reset_index()
    fig_share = px.area(
        share_ts, x="Period", y="Revenue_Share_Pct",
        color="Platform",
        color_discrete_map=BRAND_COLORS,
        title="Cuota de Ingresos del Ecosistema por Período (%)",
        labels={"Revenue_Share_Pct": "Share (%)", "Period": ""},
        groupnorm="",
    )
    fig_share.update_layout(**DARK_LAYOUT)
    st.plotly_chart(fig_share, use_container_width=True)

    # ── Basic vs Diluted EPS ──────────────────
    cols_eps = st.columns(2)
    eps_df = df.groupby(["Period", "Platform"])[["Basic_EPS", "Diluted_EPS"]].mean().reset_index()
    for col_w, (eps_col, title) in zip(cols_eps, [
        ("Basic_EPS",   "Basic EPS por Período"),
        ("Diluted_EPS", "Diluted EPS por Período"),
    ]):
        with col_w:
            fig = px.line(
                eps_df, x="Period", y=eps_col,
                color="Platform",
                color_discrete_map=BRAND_COLORS,
                title=title,
                labels={eps_col: "EPS (USD)", "Period": ""},
                markers=True,
            )
            fig.update_layout(**DARK_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

    # ── Radar Chart ───────────────────────────
    st.markdown("#### 🕸️ Radar: Perfil Competitivo por Plataforma")
    radar_metrics = [
        "Gross_Margin_Pct", "Operating_Margin_Pct",
        "EBITDA_Margin_Pct", "Revenue_Share_Pct",
        "Net_Income_Growth",
    ]
    radar_labels = [
        "Gross Margin", "Op. Margin", "EBITDA Margin",
        "Rev. Share", "Net Inc. Growth",
    ]
    radar_agg = df.groupby("Platform")[radar_metrics].mean().reset_index()
    # Normalizar 0-1
    for col in radar_metrics:
        col_min, col_max = radar_agg[col].min(), radar_agg[col].max()
        rng = col_max - col_min if col_max != col_min else 1
        radar_agg[col + "_norm"] = (radar_agg[col] - col_min) / rng

    fig_radar = go.Figure()
    for _, row in radar_agg.iterrows():
        plat = row["Platform"]
        vals = [row[c + "_norm"] for c in radar_metrics]
        vals += [vals[0]]  # cerrar el polígono
        theta = radar_labels + [radar_labels[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=theta,
            fill="toself", name=plat,
            line=dict(color=get_color(plat), width=2),
            fillcolor=get_color(plat).replace("#", "rgba(") + ",0.15)" if False else
                      f"rgba{tuple(int(get_color(plat).lstrip('#')[i:i+2], 16) for i in (0,2,4))}".replace(
                          ")", ",0.15)"),
        ))
    fig_radar.update_layout(
        **DARK_LAYOUT,
        title="Radar Competitivo (valores normalizados 0-1)",
        polar=dict(
            bgcolor="#252836",
            radialaxis=dict(visible=True, range=[0, 1], color="#A0AEC0"),
            angularaxis=dict(color="#A0AEC0"),
        ),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── Tabla comparativa ─────────────────────
    st.markdown("#### 📋 Tabla Comparativa de Métricas Promedio")
    tbl = df.groupby("Platform").agg(
        Revenue_Avg=("Total_Revenue",       "mean"),
        EBITDA_Avg=("EBITDA",               "mean"),
        NetIncome_Avg=("Net_Income",         "mean"),
        GrossMargin_Avg=("Gross_Margin_Pct", "mean"),
        OpMargin_Avg=("Operating_Margin_Pct","mean"),
        EBITDAMargin_Avg=("EBITDA_Margin_Pct","mean"),
        RevShare_Avg=("Revenue_Share_Pct",   "mean"),
        BasicEPS_Avg=("Basic_EPS",           "mean"),
        DilutedEPS_Avg=("Diluted_EPS",       "mean"),
    ).reset_index()
    tbl_display = tbl.copy()
    tbl_display["Revenue_Avg"]     = tbl["Revenue_Avg"].apply(fmt_millions)
    tbl_display["EBITDA_Avg"]      = tbl["EBITDA_Avg"].apply(fmt_millions)
    tbl_display["NetIncome_Avg"]   = tbl["NetIncome_Avg"].apply(fmt_millions)
    tbl_display["GrossMargin_Avg"] = tbl["GrossMargin_Avg"].apply(fmt_pct)
    tbl_display["OpMargin_Avg"]    = tbl["OpMargin_Avg"].apply(fmt_pct)
    tbl_display["EBITDAMargin_Avg"]= tbl["EBITDAMargin_Avg"].apply(fmt_pct)
    tbl_display["RevShare_Avg"]    = tbl["RevShare_Avg"].apply(fmt_pct)
    tbl_display.rename(columns={
        "Platform": "Plataforma",
        "Revenue_Avg": "Ingresos Prom.",
        "EBITDA_Avg": "EBITDA Prom.",
        "NetIncome_Avg": "Net Income Prom.",
        "GrossMargin_Avg": "Gross Margin",
        "OpMargin_Avg": "Op. Margin",
        "EBITDAMargin_Avg": "EBITDA Margin",
        "RevShare_Avg": "Rev. Share",
        "BasicEPS_Avg": "Basic EPS",
        "DilutedEPS_Avg": "Diluted EPS",
    }, inplace=True)
    st.dataframe(tbl_display, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# PESTAÑA 5: PREDICCIÓN Y ANALÍTICA AVANZADA
# ─────────────────────────────────────────────
def run_linear_regression(df_plat: pd.DataFrame, target: str):
    """Entrena regresión lineal y retorna métricas + predicciones."""
    df_sorted = df_plat.sort_values(["Year", "Quarter"]).dropna(subset=[target]).copy()
    if len(df_sorted) < 4:
        return None
    df_sorted["t"] = np.arange(len(df_sorted))
    X = df_sorted[["t"]].values
    y = df_sorted[target].values

    model = LinearRegression()
    model.fit(X, y)
    y_pred_hist = model.predict(X)

    # Proyectar 4 trimestres más
    n = len(df_sorted)
    t_future = np.arange(n, n + 4).reshape(-1, 1)
    y_future = model.predict(t_future)

    metrics = {
        "R2":   r2_score(y, y_pred_hist),
        "MAE":  mean_absolute_error(y, y_pred_hist),
        "RMSE": np.sqrt(mean_squared_error(y, y_pred_hist)),
    }

    # Períodos futuros
    last_year = int(df_sorted["Year"].iloc[-1])
    last_q    = int(df_sorted["Quarter"].iloc[-1])
    future_periods = []
    for _ in range(4):
        last_q += 1
        if last_q > 4:
            last_q = 1
            last_year += 1
        future_periods.append(f"{last_year}-Q{last_q}")

    return {
        "historical_periods": df_sorted["Period"].tolist(),
        "historical_actual":  y.tolist(),
        "historical_fitted":  y_pred_hist.tolist(),
        "future_periods":     future_periods,
        "future_pred":        y_future.tolist(),
        "metrics":            metrics,
        "platform":           df_plat["Platform"].iloc[0] if "Platform" in df_plat.columns else "",
    }


def tab_prediccion(df: pd.DataFrame):
    st.markdown("### 🔮 Predicción y Analítica Avanzada")
    if df.empty:
        st.warning("Sin datos para los filtros seleccionados.")
        return

    target_opts = {
        "Total Revenue":  "Total_Revenue",
        "EBITDA":         "EBITDA",
        "Net Income":     "Net_Income",
    }
    selected_target = st.selectbox(
        "Variable a proyectar",
        options=list(target_opts.keys()),
        index=0,
    )
    target_col = target_opts[selected_target]

    platforms = sorted(df["Platform"].unique())
    results   = {}
    for plat in platforms:
        df_p = df[df["Platform"] == plat].copy()
        res  = run_linear_regression(df_p, target_col)
        if res:
            results[plat] = res

    if not results:
        st.warning("No hay suficientes datos por plataforma para entrenar un modelo (mínimo 4 trimestres).")
        return

    # ── Gráfico unificado de proyecciones ─────
    fig_pred = go.Figure()
    for plat, res in results.items():
        color = get_color(plat)
        all_periods = res["historical_periods"] + res["future_periods"]

        # Histórico real
        fig_pred.add_trace(go.Scatter(
            x=res["historical_periods"],
            y=res["historical_actual"],
            mode="lines+markers", name=f"{plat} — Real",
            line=dict(color=color, width=2),
            marker=dict(size=7, symbol="circle"),
        ))
        # Línea de regresión (histórico)
        fig_pred.add_trace(go.Scatter(
            x=res["historical_periods"],
            y=res["historical_fitted"],
            mode="lines", name=f"{plat} — Regresión",
            line=dict(color=color, width=1.5, dash="dot"),
        ))
        # Proyección futura
        bridge_x = [res["historical_periods"][-1]] + res["future_periods"]
        bridge_y = [res["historical_fitted"][-1]] + res["future_pred"]
        fig_pred.add_trace(go.Scatter(
            x=bridge_x, y=bridge_y,
            mode="lines+markers", name=f"{plat} — Proyección",
            line=dict(color=color, width=2, dash="dash"),
            marker=dict(size=8, symbol="diamond"),
        ))

    # add_vline no funciona con ejes categóricos; usar shape + annotation manualmente
    last_hist_period = list(results.values())[0]["historical_periods"][-1]
    fig_pred.add_shape(
        type="line",
        xref="x", yref="paper",
        x0=last_hist_period, x1=last_hist_period,
        y0=0, y1=1,
        line=dict(dash="dot", color="rgba(255,255,255,0.3)", width=1.5),
    )
    fig_pred.add_annotation(
        x=last_hist_period, y=1,
        xref="x", yref="paper",
        text="→ Proyección",
        showarrow=False,
        font=dict(color="#A0AEC0", size=11),
        xanchor="left",
        yanchor="bottom",
    )
    fig_pred.update_layout(
        **DARK_LAYOUT,
        title=f"Proyección de {selected_target} — Regresión Lineal (4 trimestres futuros)",
        yaxis_title=selected_target,
        hovermode="x unified",
        height=520,
    )
    st.plotly_chart(fig_pred, use_container_width=True)

    # ── Métricas del modelo ───────────────────
    st.markdown("#### 📐 Métricas del Modelo por Plataforma")
    metric_rows = []
    for plat, res in results.items():
        m = res["metrics"]
        metric_rows.append({
            "Plataforma": plat,
            "R²":         f"{m['R2']:.4f}",
            "MAE":        fmt_millions(m["MAE"]),
            "RMSE":       fmt_millions(m["RMSE"]),
        })
    st.dataframe(
        pd.DataFrame(metric_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "ℹ️ **Interpretación:** R² cercano a 1 indica mejor ajuste del modelo. "
        "MAE y RMSE miden el error promedio de las predicciones históricas."
    )

    # ── Valores proyectados ────────────────────
    st.markdown("#### 📅 Valores Proyectados — Próximos 4 Trimestres")
    proj_rows = []
    for plat, res in results.items():
        for period, val in zip(res["future_periods"], res["future_pred"]):
            proj_rows.append({
                "Plataforma": plat,
                "Período":    period,
                selected_target: fmt_millions(val),
            })
    if proj_rows:
        st.dataframe(
            pd.DataFrame(proj_rows),
            use_container_width=True,
            hide_index=True,
        )


# ─────────────────────────────────────────────
# INSIGHTS AUTOMÁTICOS
# ─────────────────────────────────────────────
def render_insights(df: pd.DataFrame):
    """Genera y muestra 4+ insights automáticos."""
    if df.empty:
        return
    st.markdown("---")
    st.markdown("### 💡 Insights")

    agg = df.groupby("Platform").agg(
        EBITDA_Margin=("EBITDA_Margin_Pct",  "mean"),
        QoQ_Growth=("QoQ_Revenue_Growth",    "mean"),
        OpEx_Ratio=("OpEx_Ratio_Pct",        "mean"),
        Rev_Share=("Revenue_Share_Pct",      "mean"),
        Net_Income=("Net_Income",            "mean"),
        Total_Revenue=("Total_Revenue",      "mean"),
    ).reset_index()

    def best(col, df=agg, higher=True):
        if higher:
            idx = df[col].idxmax()
        else:
            idx = df[col].idxmin()
        return df.loc[idx, "Platform"], df.loc[idx, col]

    insights = []

    plat, val = best("EBITDA_Margin")
    insights.append((
        "🏆 Mayor Margen EBITDA",
        f"<b>{plat}</b> lidera con un margen EBITDA promedio de <b>{val:.1f}%</b>, "
        "reflejando la mayor eficiencia operativa del ecosistema.",
    ))

    plat, val = best("QoQ_Growth")
    insights.append((
        "🚀 Mayor Crecimiento Trimestral",
        f"<b>{plat}</b> registra el crecimiento QoQ promedio más alto: <b>{val:.1f}%</b>. "
        "Señal de aceleración del negocio.",
    ))

    plat, val = best("OpEx_Ratio", higher=False)
    insights.append((
        "✂️ Mejor Control de Gastos Operativos",
        f"<b>{plat}</b> tiene el ratio OpEx más bajo ({val:.1f}%), "
        "indicando mayor disciplina en el control de costos operativos.",
    ))

    plat, val = best("Rev_Share")
    insights.append((
        "👑 Dominio de Mercado",
        f"<b>{plat}</b> concentra en promedio el <b>{val:.1f}%</b> de los ingresos totales del ecosistema, "
        "consolidándose como el líder de la industria.",
    ))

    # Extra: plataforma más rentable en términos absolutos
    plat, val = best("Net_Income")
    insights.append((
        "💵 Mayor Utilidad Neta Promedio",
        f"<b>{plat}</b> genera la mayor utilidad neta promedio: <b>{fmt_millions(val)}</b> por trimestre.",
    ))

    cols = st.columns(len(insights))
    for col, (title, text) in zip(cols, insights):
        with col:
            st.markdown(
                f"""
                <div class="insight-card">
                    <div class="insight-title">{title}</div>
                    <div class="insight-text">{text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────
# ENCABEZADO PRINCIPAL
# ─────────────────────────────────────────────
def render_header():
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #0E1117 0%, #1C1E26 100%);
            border-bottom: 1px solid rgba(0,212,255,0.2);
            padding: 24px 0 16px 0;
            margin-bottom: 24px;
        ">
            <h1 style="
                margin: 0;
                font-size: 28px;
                font-weight: 800;
                background: linear-gradient(90deg, #00D4FF, #FFFFFF);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            ">
                📺 Streaming Financial Dashboard
            </h1>
            <p style="color: #A0AEC0; margin: 4px 0 0 0; font-size: 14px;">
                Análisis Comparativo · Netflix · Amazon · Disney · HBO
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    render_header()

    # ── Carga de archivo ──────────────────────
    st.sidebar.markdown("## 📂 Cargar Datos")
    uploaded_file = st.sidebar.file_uploader(
        "Sube tu archivo CSV",
        type=["csv"],
        help="El CSV debe contener las columnas requeridas. Ver plantilla abajo.",
    )
    st.sidebar.markdown("---")

    if uploaded_file is None:
        # Pantalla de bienvenida
        st.markdown(
            """
            <div style="
                background: linear-gradient(135deg, #1C1E26 0%, #252836 100%);
                border: 1px dashed rgba(0,212,255,0.3);
                border-radius: 16px;
                padding: 40px;
                text-align: center;
                margin-top: 40px;
            ">
                <h2 style="color:#00D4FF; margin-bottom:16px;">👋 Bienvenido al Dashboard</h2>
                <p style="color:#A0AEC0; font-size:15px; max-width:600px; margin:0 auto 24px;">
                    Carga un archivo CSV con datos financieros de plataformas de streaming
                    para comenzar el análisis interactivo.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### 📋 Estructura Requerida del CSV")
        sample = {
            "Platform":         ["Netflix", "Amazon", "Disney", "HBO"],
            "Year":             [2023, 2023, 2023, 2023],
            "Quarter":          [1, 1, 1, 1],
            "Total_Revenue":    [8187.8, 21354.0, 5340.3, 2102.4],
            "Gross_Profit":     [3462.7, 9844.7, 2017.3, 672.3],
            "Operating_Income": [1699.4, 4774.0, 1285.3, 210.0],
            "EBITDA":           [2050.0, 5900.0, 1680.0, 420.0],
            "Net_Income":       [1305.1, 3172.7, 1282.9, 188.2],
            "Operating_Expense":[6488.4, 16580.0, 4055.0, 1892.4],
            "Pretax_Income":    [1711.3, 4210.0, 1350.0, 232.5],
            "Tax_Provision":    [406.2, 1037.3, 67.1, 44.3],
            "Basic_EPS":        [2.97, 3.20, 0.64, 0.44],
            "Diluted_EPS":      [2.88, 3.17, 0.62, 0.43],
        }
        st.dataframe(pd.DataFrame(sample), use_container_width=True, hide_index=True)
        st.caption("Los valores numéricos están en millones de USD.")
        return

    df = validate_upload(uploaded_file)
    if df is None:
        return

    # ── Filtros y datos filtrados ──────────────
    df_filtered = apply_sidebar_filters(df)

    # ── KPI Cards ────────────────────────────
    render_kpi_cards(df_filtered)
    st.markdown("---")

    # ── Pestañas ─────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💰 Rentabilidad",
        "📈 Crecimiento",
        "🏗️ Costos",
        "⚔️ Benchmarking",
        "🔮 Predicción",
    ])

    with tab1:
        tab_rentabilidad(df_filtered)
    with tab2:
        tab_crecimiento(df_filtered)
    with tab3:
        tab_costos(df_filtered)
    with tab4:
        tab_benchmarking(df_filtered)
    with tab5:
        tab_prediccion(df_filtered)

    # ── Insights ────────────────────────────
    render_insights(df_filtered)

    # ── Footer ───────────────────────────────
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color:#4A5568; font-size:12px;'>"
        "Streaming Financial Dashboard · Desarrollado con Streamlit & Plotly · "
        "Los datos son responsabilidad del usuario · Uso exclusivo interno"
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()