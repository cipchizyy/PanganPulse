
from __future__ import annotations

import html
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PANGANPULSE
# Food Price Forecasting Dashboard
# Regression    : LightGBM A
# Classification: Logistic Regression
# Forecast      : H+30
# ============================================================


BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

BG = "#F7F2E8"
SURFACE = "#FFFDF8"
GREEN = "#315C49"
GREEN_LIGHT = "#4F8068"
SAGE = "#E6EEE5"
GOLD = "#D8BD8D"
TEXT = "#29372F"
MUTED = "#718076"
BORDER = "#E7DECF"


st.set_page_config(
    page_title="PanganPulse | Prediksi Harga Pangan",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 1. GLOBAL CSS
# ============================================================

st.markdown("""
<style>

:root {
    color-scheme: light;
    --bg: #F7F2E8;
    --surface: #FFFDF8;
    --green: #315C49;
    --green-light: #4F8068;
    --sage: #E6EEE5;
    --gold: #D8BD8D;
    --text: #29372F;
    --muted: #718076;
    --border: #E7DECF;
}

html {
    scroll-behavior: smooth;
    scroll-padding-top: 20px;
}

body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: "Segoe UI", sans-serif;
}

[data-testid="stMainBlockContainer"] {
    max-width: 1280px;
    padding-top: 1.7rem;
    padding-bottom: 4rem;
}

header[data-testid="stHeader"] {
    background: rgba(247,242,232,0.95);
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
    color: var(--text);
}

[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {
    color: var(--green) !important;
    font-weight: 750;
    letter-spacing: -0.025em;
}

[data-testid="stCaptionContainer"] {
    color: var(--muted) !important;
}

/* SIDEBAR */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #F1EBDD 0%,
        #EAE3D4 100%
    ) !important;

    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] > div {
    padding: 1.5rem 1.15rem;
}

section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
    color: var(--green) !important;
}

.sidebar-brand {
    font-size: 1.55rem;
    font-weight: 800;
    color: var(--green);
    letter-spacing: -0.04em;
}

.sidebar-description {
    font-size: 0.82rem;
    color: #788A7C;
    margin-top: 5px;
}

.sidebar-link {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 15px 17px;
    margin-bottom: 11px;
    border-radius: 17px;
    background: rgba(49,92,73,0.075);
    border: 1px solid rgba(49,92,73,0.06);
    color: #315C49 !important;
    text-decoration: none !important;
    font-size: 0.94rem;
    font-weight: 700;
    transition: all 0.22s ease;
}

.sidebar-link:hover {
    background: #DDE9DD;
    border-color: #C6D8C7;
    transform: translateX(4px);
    box-shadow: 0 7px 16px rgba(49,92,73,0.08);
}

/* HERO */

.hero {
    position: relative;
    overflow: hidden;
    padding: 3rem 2.8rem;
    border-radius: 30px;
    background: linear-gradient(
        125deg,
        #254E3D 0%,
        #376D55 55%,
        #59866F 100%
    );
    box-shadow: 0 18px 40px rgba(49,92,73,0.15);
    margin-bottom: 1.5rem;
}

.hero::after {
    content: "";
    position: absolute;
    width: 270px;
    height: 270px;
    right: -75px;
    top: -105px;
    border-radius: 50%;
    background: rgba(255,255,255,0.08);
}

.hero-kicker {
    display: inline-block;
    background: rgba(255,255,255,0.14);
    color: #F1DFB9;
    border-radius: 30px;
    padding: 8px 14px;
    font-size: 0.72rem;
    letter-spacing: 0.13em;
    font-weight: 750;
    margin-bottom: 1.2rem;
}

.hero-title {
    position: relative;
    z-index: 1;
    color: #FFFDF8;
    font-size: clamp(2rem, 4vw, 3.2rem);
    font-weight: 850;
    line-height: 1.13;
    letter-spacing: -0.045em;
    margin-bottom: 1rem;
}

.hero-copy {
    position: relative;
    z-index: 1;
    max-width: 720px;
    color: #E8F0E8;
    font-size: 1.04rem;
    line-height: 1.8;
}

/* SECTION */

.section-title {
    color: var(--green);
    font-size: 1.7rem;
    font-weight: 800;
    letter-spacing: -0.035em;
    margin-top: 1.3rem;
    margin-bottom: 0.3rem;
}

.section-subtitle {
    color: var(--muted);
    font-size: 0.94rem;
    line-height: 1.65;
    margin-bottom: 1.2rem;
}

.section-anchor {
    scroll-margin-top: 20px;
}

/* METRIC CARDS */

[data-testid="stMetric"] {
    background: linear-gradient(
        145deg,
        #FFFDF8 0%,
        #F8F3E9 100%
    ) !important;

    border: 1px solid var(--border) !important;
    border-radius: 22px !important;
    padding: 19px 17px !important;
    min-height: 115px;
    box-shadow: 0 6px 20px rgba(48,65,50,0.055);
}

[data-testid="stMetricLabel"] {
    color: #738176 !important;
    font-size: 0.87rem !important;
    font-weight: 650 !important;
}

[data-testid="stMetricValue"] {
    color: var(--green) !important;
    font-size: 1.8rem !important;
    font-weight: 800 !important;
}

[data-testid="stMetricDelta"] {
    font-weight: 650 !important;
}

/* CHART */

[data-testid="stPlotlyChart"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 22px;
    padding: 9px;
    box-shadow: 0 6px 20px rgba(48,65,50,0.045);
}

/* TABLE */

.table-shell {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 20px;
    overflow: auto;
    box-shadow: 0 6px 20px rgba(48,65,50,0.045);
    margin: 10px 0 20px 0;
}

.pretty-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 0.91rem;
}

.pretty-table thead th {
    position: sticky;
    top: 0;
    background: #315C49;
    color: #FFFFFF;
    text-align: left;
    padding: 15px 16px;
    font-weight: 750;
    white-space: nowrap;
    border-bottom: 2px solid #49765E;
}

.pretty-table tbody td {
    color: #344239;
    padding: 13px 16px;
    border-bottom: 1px solid #EEE7DA;
    white-space: nowrap;
    background: #FFFDF8;
}

.pretty-table tbody tr:nth-child(even) td {
    background: #F7F4EB;
}

.pretty-table tbody tr:hover td {
    background: #EAF0E7;
}

.pretty-table tbody tr:last-child td {
    border-bottom: none;
}

/* STATUS BADGES */

.status-badge {
    display: inline-block;
    padding: 5px 12px;
    border-radius: 30px;
    font-size: 0.81rem;
    font-weight: 750;
}

.badge-up {
    background: #E1EFE4;
    color: #286443;
}

.badge-down {
    background: #F7E4E0;
    color: #A14E45;
}

.badge-stable {
    background: #ECEDE7;
    color: #667264;
}

/* INFO NOTE */

.info-note {
    padding: 17px 20px;
    background: #F4E9D2;
    border: 1px solid #E6D5B0;
    border-radius: 18px;
    color: #594A32;
    line-height: 1.75;
    margin-bottom: 20px;
}

/* INPUTS */

[data-baseweb="select"] > div,
[data-baseweb="input"] > div {
    background: #FFFDF8 !important;
    border-color: #DCD2C1 !important;
    border-radius: 14px !important;
}

[data-baseweb="select"] span,
[data-baseweb="input"] input {
    color: #29372F !important;
}

/* BUTTONS */

.stButton > button,
.stDownloadButton > button {
    border-radius: 14px !important;
    border: none !important;
    padding: 0.65rem 1.2rem !important;
    font-weight: 700 !important;
    transition: all 0.2s ease;
}

.stButton > button {
    background: linear-gradient(
        135deg,
        #315C49,
        #4F8068
    ) !important;
    color: white !important;
}

.stDownloadButton > button {
    background: #D8BD8D !important;
    color: #3C392F !important;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 18px rgba(49,92,73,0.16);
}

/* EXPANDER */

[data-testid="stExpander"] {
    background: #FFFDF8 !important;
    border: 1px solid var(--border) !important;
    border-radius: 17px !important;
}

/* ALERT */

[data-testid="stAlert"] {
    border-radius: 17px;
}

/* DIVIDER */

hr {
    border-color: #E5DCCB !important;
    margin-top: 2rem !important;
    margin-bottom: 2rem !important;
}

/* FOOTER */

.footer {
    text-align: center;
    color: #7D887E;
    font-size: 0.82rem;
    padding: 20px;
}

@media(max-width:768px) {
    .hero {
        padding: 1.8rem;
        border-radius: 23px;
    }

    .hero-title {
        font-size: 2rem;
    }

    .section-title {
        font-size: 1.4rem;
    }
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def format_number(value, decimals=1):

    if value is None or pd.isna(value):
        return "—"

    result = f"{float(value):,.{decimals}f}"

    return result.replace(",", "_").replace(".", ",").replace("_", ".")


def format_date(value):

    if pd.isna(value):
        return "—"

    months = [
        "Januari", "Februari", "Maret", "April",
        "Mei", "Juni", "Juli", "Agustus",
        "September", "Oktober", "November", "Desember"
    ]

    d = pd.Timestamp(value)

    return f"{d.day} {months[d.month - 1]} {d.year}"


def get_class_label(value, labels):

    try:
        key = str(int(value))
    except (ValueError, TypeError):
        key = str(value)

    return labels.get(key, f"Kelas {key}")


def render_header(title, subtitle):

    st.markdown(
        f'<div class="section-title">{html.escape(title)}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="section-subtitle">{html.escape(subtitle)}</div>',
        unsafe_allow_html=True
    )


def render_table(df, badge_column=None, max_height=520):

    if df.empty:
        st.info("Tidak ada data untuk ditampilkan.")
        return

    headers = "".join(
        f"<th>{html.escape(str(col))}</th>"
        for col in df.columns
    )

    rows = []

    for _, row in df.iterrows():

        cells = []

        for col in df.columns:

            value = row[col]

            if pd.isna(value):
                value = "—"

            value = str(value)

            if col == badge_column:

                normalized = value.lower()

                if "naik" in normalized:
                    badge = "badge-up"
                elif "turun" in normalized:
                    badge = "badge-down"
                else:
                    badge = "badge-stable"

                cell = (
                    f'<span class="status-badge {badge}">'
                    f'{html.escape(value)}</span>'
                )

            else:
                cell = html.escape(value)

            cells.append(f"<td>{cell}</td>")

        rows.append("<tr>" + "".join(cells) + "</tr>")

    max_style = (
        f"max-height:{int(max_height)}px;"
        if max_height else ""
    )

    table_html = f"""
<div class="table-shell" style="{max_style}">
<table class="pretty-table">
<thead><tr>{headers}</tr></thead>
<tbody>{"".join(rows)}</tbody>
</table>
</div>
"""

    st.markdown(table_html, unsafe_allow_html=True)


def apply_chart_style(fig, height=390):

    fig.update_layout(
        height=height,
        margin=dict(l=15, r=15, t=45, b=15),
        paper_bgcolor="#FFFDF8",
        plot_bgcolor="#FFFDF8",
        font=dict(
            family="Segoe UI",
            color="#344239",
            size=12
        ),
        title_font=dict(
            color="#315C49",
            size=17
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0
        ),
    )

    fig.update_xaxes(
        showgrid=False,
        linecolor="#E8E1D5",
        tickfont=dict(color="#718076")
    )

    fig.update_yaxes(
        gridcolor="#EAE4D9",
        zeroline=False,
        linecolor="#E8E1D5",
        tickfont=dict(color="#718076")
    )

    return fig


# ============================================================
# 3. LOAD MODELS AND DATA
# ============================================================

@st.cache_resource(show_spinner="Memuat model dan data penelitian...")

def load_assets():

    required_files = [
        "regression_model.joblib",
        "classification_model.joblib",
        "model_metadata.json",
        "eligible_series.csv",
        "latest_history.csv",
        "latest_features.csv",
    ]

    missing = [
        name for name in required_files
        if not (ARTIFACT_DIR / name).exists()
    ]

    if missing:

        raise FileNotFoundError(
            "File artefak tidak ditemukan: "
            + ", ".join(missing)
        )

    metadata = json.loads(
        (ARTIFACT_DIR / "model_metadata.json").read_text(
            encoding="utf-8"
        )
    )

    regression_model = joblib.load(
        ARTIFACT_DIR / "regression_model.joblib"
    )

    classification_model = joblib.load(
        ARTIFACT_DIR / "classification_model.joblib"
    )

    eligible = pd.read_csv(
        ARTIFACT_DIR / "eligible_series.csv"
    )

    history = pd.read_csv(
        ARTIFACT_DIR / "latest_history.csv",
        parse_dates=["Date"]
    )

    latest_features = pd.read_csv(
        ARTIFACT_DIR / "latest_features.csv",
        parse_dates=["Date"]
    )

    reg_features = metadata["regression"]["features"]

    cls_features = metadata["classification"]["features"]

    # Validate feature columns.

    for features in [reg_features, cls_features]:

        missing_columns = [
            col for col in features
            if col not in latest_features.columns
        ]

        if missing_columns:

            raise ValueError(
                f"Fitur tidak ditemukan: {missing_columns}"
            )

    # Validate data columns.

    required_feature_columns = [
        "Date", "Commodity", "Province", "current_price"
    ]

    for col in required_feature_columns:

        if col not in latest_features.columns:
            raise ValueError(f"Kolom {col} tidak tersedia.")

    for col in ["Date", "Commodity", "Province", "Price"]:

        if col not in history.columns:
            raise ValueError(f"Kolom historis {col} tidak tersedia.")

    # Validate duplicate series.

    if latest_features.duplicated(
        ["Commodity", "Province"]
    ).any():

        raise ValueError(
            "Terdapat kombinasi komoditas dan provinsi duplikat."
        )

    # Prediction input.

    reg_input = latest_features[reg_features].copy()

    cls_input = latest_features[cls_features].copy()

    predicted_price = regression_model.predict(reg_input)

    predicted_class = classification_model.predict(cls_input)

    if len(predicted_price) != len(latest_features):

        raise ValueError(
            "Jumlah hasil prediksi regresi tidak sesuai."
        )

    if len(predicted_class) != len(latest_features):

        raise ValueError(
            "Jumlah hasil prediksi klasifikasi tidak sesuai."
        )

    predictions = latest_features[
        ["Date", "Commodity", "Province", "current_price"]
    ].copy()

    horizon = int(metadata["horizon_days"])

    predictions["TargetDate"] = (
        predictions["Date"]
        + pd.to_timedelta(horizon, unit="D")
    )

    predictions["Pred_Harga_H30"] = np.asarray(
        predicted_price,
        dtype=float
    )

    labels = metadata["target_classification"]["labels"]

    predictions["Pred_Kelas_H30"] = [
        get_class_label(value, labels)
        for value in predicted_class
    ]

    current = predictions["current_price"].to_numpy(dtype=float)

    predicted = predictions["Pred_Harga_H30"].to_numpy(dtype=float)

    predictions["Perubahan_Persen"] = np.where(
        current != 0,
        (predicted / current - 1) * 100,
        np.nan
    )

    # Classification probabilities.

    if hasattr(classification_model, "predict_proba"):

        probabilities = classification_model.predict_proba(cls_input)

        classes = getattr(
            classification_model,
            "classes_",
            range(probabilities.shape[1])
        )

        for idx, class_value in enumerate(classes):

            label = get_class_label(class_value, labels)

            predictions[f"Prob_{label}"] = probabilities[:, idx]

    # Normalize dates.

    history["Date"] = pd.to_datetime(history["Date"])

    latest_features["Date"] = pd.to_datetime(
        latest_features["Date"]
    )

    predictions["Date"] = pd.to_datetime(
        predictions["Date"]
    )

    predictions["TargetDate"] = pd.to_datetime(
        predictions["TargetDate"]
    )

    return (
        metadata,
        regression_model,
        classification_model,
        eligible,
        history,
        latest_features,
        predictions
    )


# ============================================================
# 4. DISPLAY HELPERS
# ============================================================

def show_historical_warning(metadata):

    last_date = pd.Timestamp(
        metadata["latest_data_date"]
    )

    target_date = last_date + pd.Timedelta(
        days=int(metadata["horizon_days"])
    )

    st.markdown(
        f"""
<div class="info-note">
<b>Informasi periode data</b><br>
Dataset penelitian berakhir pada <b>{format_date(last_date)}</b>.
Prediksi H+30 merupakan simulasi historis dengan target
<b>{format_date(target_date)}</b>, bukan prakiraan harga terkini tahun 2026.
Satuan harga mengikuti dataset sumber dan belum diverifikasi.
</div>
""",
        unsafe_allow_html=True
    )


def prediction_row(predictions, commodity, province):

    selected = predictions[
        (predictions["Commodity"] == commodity)
        & (predictions["Province"] == province)
    ]

    if selected.empty:
        return None

    return selected.iloc[0]


def make_history_chart(history, forecast_row):

    hist = history[
        (history["Commodity"] == forecast_row["Commodity"])
        & (history["Province"] == forecast_row["Province"])
    ].sort_values("Date")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=hist["Date"],
            y=hist["Price"],
            mode="lines",
            name="Harga historis",
            line=dict(
                color="#367858",
                width=3
            ),
            hovertemplate=(
                "%{x|%d %b %Y}<br>"
                "Harga: %{y:,.1f}<extra></extra>"
            )
        )
    )

    last_date = pd.Timestamp(forecast_row["Date"])

    target_date = pd.Timestamp(forecast_row["TargetDate"])

    last_price = float(forecast_row["current_price"])

    predicted_price = float(forecast_row["Pred_Harga_H30"])

    fig.add_trace(
        go.Scatter(
            x=[last_date, target_date],
            y=[last_price, predicted_price],
            mode="lines",
            name="Penghubung ke H+30",
            line=dict(
                color="#D2A75F",
                width=2.5,
                dash="dash"
            ),
            hoverinfo="skip"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[target_date],
            y=[predicted_price],
            mode="markers",
            name="Prediksi H+30",
            marker=dict(
                color="#D2A75F",
                size=14,
                symbol="diamond"
            ),
            hovertemplate=(
                "%{x|%d %b %Y}<br>"
                "Prediksi: %{y:,.1f}<extra></extra>"
            )
        )
    )

    fig.update_layout(
        title="Pergerakan harga dan titik prediksi",
        hovermode="x unified",
        yaxis_title="Harga (satuan dataset)",
        xaxis_title=None
    )

    return apply_chart_style(fig, 420)


# ============================================================
# 5. LOAD APPLICATION
# ============================================================

try:

    (
        metadata,
        regression_model,
        classification_model,
        eligible,
        history,
        latest_features,
        predictions
    ) = load_assets()

except Exception as exc:

    st.error(
        "Aplikasi belum dapat memuat model atau data."
    )

    st.exception(exc)

    st.info(
        "Pastikan folder artifacts dan seluruh file model tersedia."
    )

    st.stop()


# ============================================================
# 6. SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-brand">🌾 PanganPulse</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-description">'
        'Food Price Forecasting Dashboard'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    navigation = [
        ("🏠", "Dashboard", "#dashboard"),
        ("📈", "Prediksi H+30", "#prediksi"),
        ("📊", "Perbandingan Provinsi", "#perbandingan"),
        ("ℹ️", "Tentang Model", "#tentang"),
    ]

    for icon, label, link in navigation:

        st.markdown(
            f'<a class="sidebar-link" href="{link}">'
            f'<span>{icon}</span><span>{label}</span></a>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    st.markdown("### Ringkasan Data")

    st.markdown(
        f"**{eligible['Commodity'].nunique()}** komoditas"
    )

    st.markdown(
        f"**{eligible['Province'].nunique()}** provinsi"
    )

    st.markdown(
        f"**H+{metadata['horizon_days']}** horizon prediksi"
    )

    st.caption(
        f"Data terakhir: {format_date(metadata['latest_data_date'])}"
    )

    st.markdown("---")

    st.caption("Academic Machine Learning Project")


# ============================================================
# 7. DASHBOARD
# ============================================================

st.markdown(
    '<div id="dashboard" class="section-anchor"></div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="hero">
<div class="hero-kicker">INDONESIAN FOOD PRICE ANALYSIS</div>
<div class="hero-title">Pantau Harga. Pahami Prediksi.</div>
<div class="hero-copy">
Dashboard analisis dan prediksi harga bahan pokok menggunakan
Machine Learning untuk memahami pola harga tiga komoditas utama
di Indonesia.
</div>
</div>
""", unsafe_allow_html=True)


show_historical_warning(metadata)


# ---------- OVERVIEW METRICS ----------

commodities_count = int(eligible["Commodity"].nunique())

provinces_count = int(eligible["Province"].nunique())

series_count = len(eligible)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Komoditas",
    commodities_count,
    help="Jumlah komoditas dalam penelitian."
)

c2.metric(
    "Provinsi",
    provinces_count,
    help="Jumlah provinsi dalam cakupan data."
)

c3.metric(
    "Seri tersedia",
    series_count,
    help="Jumlah kombinasi komoditas dan provinsi."
)

c4.metric(
    "Horizon prediksi",
    f"H+{metadata['horizon_days']}"
)


# ---------- HISTORICAL TREND ----------

st.markdown("### Gambaran Harga Historis")

st.caption(
    "Amati perubahan median harga antarprovinsi untuk setiap komoditas."
)

commodity_options = sorted(
    predictions["Commodity"].unique().tolist()
)

selected_commodity = st.selectbox(
    "Pilih komoditas",
    commodity_options,
    key="dashboard_commodity"
)

selected_history = history[
    history["Commodity"] == selected_commodity
]

median_history = (
    selected_history
    .groupby("Date", as_index=False)["Price"]
    .median()
    .rename(
        columns={
            "Price": "Median harga antarprovinsi"
        }
    )
    .sort_values("Date")
)

fig = px.line(
    median_history,
    x="Date",
    y="Median harga antarprovinsi",
    title=f"Tren harga — {selected_commodity}"
)

fig.update_traces(
    line=dict(
        color="#367858",
        width=3
    ),
    hovertemplate=(
        "%{x|%d %b %Y}<br>"
        "Median: %{y:,.1f}<extra></extra>"
    )
)

fig.update_layout(
    xaxis_title=None,
    yaxis_title="Harga (satuan dataset)"
)

fig = apply_chart_style(fig, 390)

st.plotly_chart(
    fig,
    use_container_width=True,
    config={"displaylogo": False}
)


# ---------- FORECAST SUMMARY ----------

st.markdown("### Ringkasan Prediksi H+30")

summary = (
    predictions
    .groupby("Commodity")
    .agg(
        Seri_tersedia=("Province", "nunique"),
        Median_harga_terakhir=("current_price", "median"),
        Median_prediksi_H30=("Pred_Harga_H30", "median"),
        Median_perubahan_persen=("Perubahan_Persen", "median")
    )
    .reset_index()
)

display_summary = summary.copy()

display_summary["Median_harga_terakhir"] = (
    display_summary["Median_harga_terakhir"]
    .map(lambda x: format_number(x, 1))
)

display_summary["Median_prediksi_H30"] = (
    display_summary["Median_prediksi_H30"]
    .map(lambda x: format_number(x, 1))
)

display_summary["Median_perubahan_persen"] = (
    display_summary["Median_perubahan_persen"]
    .map(lambda x: f"{x:+.2f}%" if pd.notna(x) else "—")
)

display_summary = display_summary.rename(
    columns={
        "Commodity": "Komoditas",
        "Seri_tersedia": "Jumlah provinsi",
        "Median_harga_terakhir": "Median harga terakhir",
        "Median_prediksi_H30": "Median prediksi H+30",
        "Median_perubahan_persen": "Median perubahan"
    }
)

render_table(display_summary, max_height=320)


# ============================================================
# 8. PREDICTION H+30
# ============================================================

st.divider()

st.markdown(
    '<div id="prediksi" class="section-anchor"></div>',
    unsafe_allow_html=True
)

render_header(
    "Prediksi Harga H+30",
    "Pilih komoditas dan provinsi untuk melihat estimasi harga "
    "30 hari setelah tanggal pengamatan terakhir."
)

show_historical_warning(metadata)


commodity = st.selectbox(
    "Pilih komoditas",
    sorted(predictions["Commodity"].unique().tolist()),
    key="prediction_commodity"
)

province_options = sorted(
    predictions.loc[
        predictions["Commodity"] == commodity,
        "Province"
    ].unique().tolist()
)

province = st.selectbox(
    "Pilih provinsi",
    province_options,
    key="prediction_province"
)

row = prediction_row(
    predictions,
    commodity,
    province
)


if row is None:

    st.error("Data prediksi tidak tersedia.")

else:

    st.markdown("### Hasil Prediksi")

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "Harga terakhir",
        format_number(row["current_price"]),
    )

    delta_value = (
        f"{row['Perubahan_Persen']:+.2f}%"
        if pd.notna(row["Perubahan_Persen"])
        else None
    )

    m2.metric(
        "Prediksi harga H+30",
        format_number(row["Pred_Harga_H30"]),
        delta=delta_value
    )

    m3.metric(
        "Klasifikasi perubahan",
        row["Pred_Kelas_H30"]
    )

    st.caption(
        f"Tanggal pengamatan: **{format_date(row['Date'])}** "
        f" · Target prediksi: **{format_date(row['TargetDate'])}**"
    )

    # HISTORY CHART

    st.markdown("### Grafik Historis dan Titik Prediksi")

    st.plotly_chart(
        make_history_chart(history, row),
        use_container_width=True,
        config={"displaylogo": False}
    )

    st.caption(
        "Garis putus-putus hanya menghubungkan harga terakhir "
        "dengan titik target H+30. Garis tersebut bukan prediksi "
        "harga harian di antara kedua tanggal."
    )

    # CLASSIFICATION PROBABILITY

    probability_cols = [
        col for col in predictions.columns
        if col.startswith("Prob_")
    ]

    if probability_cols:

        st.markdown("### Probabilitas Klasifikasi")

        prob_values = []

        for col in probability_cols:

            label = col.replace("Prob_", "")

            prob_values.append({
                "Kelas": label,
                "Probabilitas": float(row[col])
            })

        prob_df = pd.DataFrame(prob_values)

        fig_prob = px.bar(
            prob_df,
            x="Kelas",
            y="Probabilitas",
            color="Kelas",
            text=prob_df["Probabilitas"].map(
                lambda x: f"{x:.1%}"
            ),
            color_discrete_map={
                "Turun": "#C86D63",
                "Stabil": "#A7B09F",
                "Naik": "#4F8068"
            }
        )

        fig_prob.update_layout(
            showlegend=False,
            yaxis=dict(
                tickformat=".0%",
                range=[0, 1]
            ),
            xaxis_title=None,
            yaxis_title="Probabilitas"
        )

        fig_prob.update_traces(
            textposition="outside"
        )

        fig_prob = apply_chart_style(fig_prob, 320)

        st.plotly_chart(
            fig_prob,
            use_container_width=True,
            config={"displaylogo": False}
        )

    # DETAIL DATA

    with st.expander("Lihat detail data historis"):

        series_history = history[
            (history["Commodity"] == commodity)
            & (history["Province"] == province)
        ].sort_values(
            "Date",
            ascending=False
        ).head(30)

        detail_df = series_history.rename(
            columns={
                "Date": "Tanggal",
                "Commodity": "Komoditas",
                "Province": "Provinsi",
                "Price": "Harga"
            }
        ).copy()

        detail_df["Tanggal"] = (
            detail_df["Tanggal"].map(format_date)
        )

        detail_df["Harga"] = (
            detail_df["Harga"].map(
                lambda x: format_number(x, 1)
            )
        )

        render_table(detail_df, max_height=400)


# ============================================================
# 9. PROVINCE COMPARISON
# ============================================================

st.divider()

st.markdown(
    '<div id="perbandingan" class="section-anchor"></div>',
    unsafe_allow_html=True
)

render_header(
    "Perbandingan Antarprovinsi",
    "Bandingkan harga terakhir dan hasil prediksi H+30 "
    "untuk satu komoditas pada seluruh provinsi."
)

show_historical_warning(metadata)


commodity = st.selectbox(
    "Pilih komoditas untuk dibandingkan",
    sorted(predictions["Commodity"].unique().tolist()),
    key="comparison_commodity"
)

comp = predictions[
    predictions["Commodity"] == commodity
].copy()


sort_option = st.selectbox(
    "Urutkan hasil berdasarkan",
    [
        "Prediksi harga tertinggi",
        "Prediksi harga terendah",
        "Persentase kenaikan tertinggi",
        "Persentase penurunan terbesar",
        "Nama provinsi"
    ]
)


if sort_option == "Prediksi harga tertinggi":

    comp = comp.sort_values(
        "Pred_Harga_H30",
        ascending=False
    )

elif sort_option == "Prediksi harga terendah":

    comp = comp.sort_values(
        "Pred_Harga_H30",
        ascending=True
    )

elif sort_option == "Persentase kenaikan tertinggi":

    comp = comp.sort_values(
        "Perubahan_Persen",
        ascending=False
    )

elif sort_option == "Persentase penurunan terbesar":

    comp = comp.sort_values(
        "Perubahan_Persen",
        ascending=True
    )

else:

    comp = comp.sort_values("Province")


st.markdown(f"### {commodity}")

st.caption(
    f"{len(comp)} provinsi tersedia untuk komoditas ini."
)


# COMPARISON CHART

fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=comp["Province"],
        y=comp["current_price"],
        name="Harga terakhir",
        marker_color="#D8BD8D",
        hovertemplate=(
            "%{x}<br>Harga terakhir: %{y:,.1f}<extra></extra>"
        )
    )
)

fig.add_trace(
    go.Bar(
        x=comp["Province"],
        y=comp["Pred_Harga_H30"],
        name="Prediksi H+30",
        marker_color="#4F8068",
        hovertemplate=(
            "%{x}<br>Prediksi H+30: %{y:,.1f}<extra></extra>"
        )
    )
)

fig.update_layout(
    title=f"Perbandingan harga antarprovinsi — {commodity}",
    barmode="group",
    xaxis_title=None,
    yaxis_title="Harga (satuan dataset)"
)

fig = apply_chart_style(fig, 450)

st.plotly_chart(
    fig,
    use_container_width=True,
    config={"displaylogo": False}
)


# COMPARISON TABLE

st.markdown("### Tabel Perbandingan")

table = comp[
    [
        "Province",
        "current_price",
        "Pred_Harga_H30",
        "Perubahan_Persen",
        "Pred_Kelas_H30"
    ]
].rename(
    columns={
        "Province": "Provinsi",
        "current_price": "Harga terakhir",
        "Pred_Harga_H30": "Prediksi H+30",
        "Perubahan_Persen": "Perubahan (%)",
        "Pred_Kelas_H30": "Klasifikasi"
    }
).copy()

table["Harga terakhir"] = (
    table["Harga terakhir"].map(
        lambda x: format_number(x, 1)
    )
)

table["Prediksi H+30"] = (
    table["Prediksi H+30"].map(
        lambda x: format_number(x, 1)
    )
)

table["Perubahan (%)"] = (
    table["Perubahan (%)"].map(
        lambda x: f"{x:+.2f}%" if pd.notna(x) else "—"
    )
)

render_table(
    table,
    badge_column="Klasifikasi",
    max_height=600
)


# DOWNLOAD CSV

csv_data = comp.to_csv(
    index=False
).encode("utf-8-sig")

st.download_button(
    "⬇️ Unduh hasil perbandingan CSV",
    data=csv_data,
    file_name=(
        f"panganpulse_"
        f"{commodity.lower().replace(' ', '_')}_h30.csv"
    ),
    mime="text/csv"
)


# ============================================================
# 10. ABOUT MODEL
# ============================================================

st.divider()

st.markdown(
    '<div id="tentang" class="section-anchor"></div>',
    unsafe_allow_html=True
)

render_header(
    "Tentang Model dan Penelitian",
    "Informasi algoritma, performa validasi, fitur, "
    "cakupan penelitian, dan keterbatasan model."
)

show_historical_warning(metadata)


reg = metadata["regression"]

cls = metadata["classification"]

scope = metadata["scope"]


# REGRESSION

st.markdown("### Model Regresi")

st.subheader(reg["candidate"])

st.write(
    f"Model regresi dipilih berdasarkan "
    f"**{reg['selection_rule']}**."
)

r1, r2, r3 = st.columns(3)

r1.metric(
    "Mean MAE",
    format_number(
        reg["validation_metrics"]["Mean_MAE"],
        2
    )
)

r2.metric(
    "Mean RMSE",
    format_number(
        reg["validation_metrics"]["Mean_RMSE"],
        2
    )
)

r3.metric(
    "Mean MAPE",
    f"{reg['validation_metrics']['Mean_MAPE']:.2f}%"
)


# CLASSIFICATION

st.markdown("### Model Klasifikasi")

st.subheader(cls["candidate"])

st.write(
    f"Model klasifikasi dipilih berdasarkan "
    f"**{cls['selection_rule']}**."
)

c1, c2, c3 = st.columns(3)

c1.metric(
    "Accuracy",
    f"{cls['validation_metrics']['Mean_Accuracy']:.4f}"
)

c2.metric(
    "Macro F1",
    f"{cls['validation_metrics']['Mean_Macro_F1']:.4f}"
)

c3.metric(
    "Balanced Accuracy",
    f"{cls['validation_metrics']['Mean_Balanced_Accuracy']:.4f}"
)


# RESEARCH SCOPE

st.markdown("### Cakupan Penelitian")

a, b, c = st.columns(3)

a.metric(
    "Komoditas",
    len(scope["commodities"])
)

b.metric(
    "Provinsi",
    int(eligible["Province"].nunique())
)

c.metric(
    "Seri eligible",
    int(scope["n_eligible_series"])
)

st.write("**Komoditas yang dianalisis:**")

st.write(", ".join(scope["commodities"]))


# FEATURES

with st.expander("Lihat fitur yang digunakan"):

    st.markdown("**Fitur regresi:**")

    st.code(
        ", ".join(reg["features"])
    )

    st.markdown("**Fitur klasifikasi:**")

    st.code(
        ", ".join(cls["features"])
    )


# MODEL PARAMETERS

with st.expander("Lihat konfigurasi model"):

    st.markdown("**Parameter regresi:**")

    st.json(reg["params"])

    st.markdown("**Parameter klasifikasi:**")

    st.json(cls["params"])


# VALIDATION METHODOLOGY

st.markdown("### Metodologi Evaluasi")

validation = metadata["validation_design"]

folds_value = validation.get("folds", [])

if isinstance(folds_value, (list, tuple, dict)):

    fold_count = len(folds_value)

else:

    fold_count = int(folds_value)

st.write(
    f"Evaluasi menggunakan **{fold_count} fold "
    "expanding rolling-origin**."
)

st.write(
    "Periode validasi berlangsung Januari–Mei 2024."
)

st.write(
    f"Jumlah baris training final: "
    f"**{metadata['training_data']['rows']:,}**."
)


# LIMITATIONS

st.markdown("### Keterbatasan Model")

limitations = metadata.get("limitations", [])

if limitations:

    for limitation in limitations:

        st.markdown(f"- {limitation}")

else:

    st.info("Informasi keterbatasan belum tersedia dalam metadata.")

st.info(
    "Angka performa merupakan hasil validasi rolling-origin "
    "yang digunakan selama proses pemilihan model. "
    "Angka tersebut bukan hasil pengujian pada holdout independen."
)


# ============================================================
# 11. FOOTER
# ============================================================

st.divider()

st.markdown("""
<div class="footer">
🌾 <b>PanganPulse</b><br>
Food Price Forecasting Dashboard<br>
Academic Machine Learning Project<br>
Prediksi historis H+30 · Sumber data mengikuti dataset penelitian
</div>
""", unsafe_allow_html=True)
