from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from textwrap import dedent


# ============================================================
# PanganPulse — Food Price Forecasting Dashboard
# Forecast horizon: H+30
# Regression: LightGBM A
# Classification: Logistic Regression
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

st.set_page_config(
    page_title="PanganPulse | Prediksi Harga Pangan",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# STYLING
# ============================================================

st.markdown("""
<style>
    html, body, section.main, [data-testid="stAppViewContainer"] {
        scroll-behavior: smooth;
        scroll-padding-top: 1.25rem;
    }

    body, .stApp, [data-testid="stAppViewContainer"] {
        background: #FAF7EF;
        color: #2B332D;
        font-family: 'Segoe UI', sans-serif;
    }

    header[data-testid="stHeader"] {
        background: rgba(250, 247, 239, 0.92);
    }

    [data-testid="stToolbar"] {
        right: 1rem;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #F2EBDD 0%, #EAE2D2 100%);
        border-right: 1px solid #DDD2BC;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.5rem;
    }

    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] .stMarkdown p {
        color: #2F5D50;
    }

    .hero {
        background: linear-gradient(135deg, #285747 0%, #4D7A67 100%);
        padding: 2.4rem 2.5rem;
        border-radius: 30px 30px 30px 10px;
        color: #FFFDF8;
        box-shadow: 0 14px 34px rgba(47, 93, 80, 0.16);
        margin: 0.4rem 0 1.6rem 0;
        position: relative;
        overflow: hidden;
    }

    .hero::after {
        content: "";
        position: absolute;
        width: 220px;
        height: 220px;
        right: -65px;
        top: -90px;
        border-radius: 50%;
        background: rgba(255,255,255,0.07);
    }

    .hero-kicker {
        color: #E7D8B9;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 0.7rem;
    }

    .hero-title {
        font-size: clamp(1.8rem, 3vw, 2.6rem);
        line-height: 1.15;
        font-weight: 750;
        margin-bottom: 0.75rem;
        color: #FFFDF8;
    }

    .hero-copy {
        max-width: 760px;
        font-size: 1.02rem;
        line-height: 1.7;
        color: rgba(255,253,248,0.9);
    }

    .hero-subtitle {
        font-size: 1rem;
        opacity: 0.92;
    }

    .custom-card {
        background: #FFFDF8;
        padding: 1.2rem 1.2rem;
        border-radius: 22px;
        border: 1px solid #E7DDCC;
        box-shadow: 0 6px 20px rgba(80, 80, 80, 0.05);
        margin-bottom: 1rem;
    }

    div[data-testid="stMetric"] {
        min-height: 112px;
    }

    div[data-testid="stMetric"] label {
        color: #66746B;
        font-weight: 600;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #285747;
        font-weight: 750;
    }

    h1, h2, h3 {
        color: #285747;
        letter-spacing: -0.02em;
    }

    .stAlert {
        border-radius: 18px;
    }

    div[data-testid="stExpander"] {
        border: 1px solid #E7DDCC;
        border-radius: 18px;
        background: rgba(255,253,248,0.72);
    }

    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2F5D50;
        margin-top: 1.5rem;
        margin-bottom: 0.3rem;
    }

    .section-subtitle {
        color: #6A6A6A;
        margin-bottom: 1.2rem;
    }

    div[data-testid="stMetric"] {
        background: #FFFDF8;
        border: 1px solid #E7DDCC;
        padding: 14px;
        border-radius: 20px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.04);
    }

    .stButton > button {
        background: linear-gradient(135deg, #2F5D50 0%, #567C6B 100%);
        color: white;
        border: none;
        border-radius: 14px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: all 0.25s ease;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #25483E 0%, #476B5C 100%);
        transform: translateY(-1px);
        box-shadow: 0 8px 18px rgba(47, 93, 80, 0.18);
    }

    .stDownloadButton > button {
        background: #D6B98C;
        color: #2F2F2F;
        border: none;
        border-radius: 14px;
        font-weight: 600;
    }

    .stDownloadButton > button:hover {
        background: #CBAE7F;
        color: #1F1F1F;
    }

    .stSelectbox > div > div,
    .stTextInput > div > div,
    .stNumberInput > div > div {
        border-radius: 14px !important;
        border: 1px solid #D8CCB8 !important;
        background-color: #FFFDF8 !important;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 18px;
        overflow: hidden;
        border: 1px solid #E7DDCC;
        background: #FFFDF8;
    }

    .sidebar-link {
        display: flex;
        align-items: center;
        padding: 12px 15px;
        margin: 0 0 9px 0;
        border: 1px solid rgba(47,93,80,0.06);
        border-radius: 16px;
        text-decoration: none !important;
        background: rgba(47,93,80,0.075);
        color: #285747 !important;
        font-weight: 650;
        transition: all 180ms ease;
        box-shadow: 0 2px 7px rgba(45,67,54,0.025);
    }

    .sidebar-link:hover {
        background: #DDE8DC;
        border-color: rgba(47,93,80,0.16);
        color: #21483B !important;
        transform: translateX(3px);
        box-shadow: 0 6px 14px rgba(45,67,54,0.07);
    }

    .sidebar-link:focus-visible {
        outline: 3px solid #D6B98C;
        outline-offset: 2px;
    }

    .section-anchor {
        scroll-margin-top: 1rem;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 18px;
        overflow: hidden;
    }

    .stDownloadButton > button {
        min-height: 2.8rem;
        padding: 0.55rem 1rem;
    }

    @media (max-width: 768px) {
        .hero {
            padding: 1.6rem;
            border-radius: 24px 24px 24px 8px;
        }

        .hero-title {
            font-size: 1.8rem;
        }
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def format_number(value: float | int | None, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "—"

    formatted = f"{float(value):,.{decimals}f}"
    return formatted.replace(",", "_").replace(".", ",").replace("_", ".")


def format_date(value) -> str:
    if pd.isna(value):
        return "—"

    months_id = [
        "Januari", "Februari", "Maret", "April",
        "Mei", "Juni", "Juli", "Agustus",
        "September", "Oktober", "November", "Desember",
    ]

    date_value = pd.Timestamp(value)

    return (
        f"{date_value.day} "
        f"{months_id[date_value.month - 1]} "
        f"{date_value.year}"
    )


def get_class_label(value, labels: dict) -> str:
    try:
        key = str(int(value))
    except (TypeError, ValueError):
        key = str(value)

    return labels.get(key, f"Kelas {key}")


# ============================================================
# LOAD MODELS AND DATA
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
            "Artefak berikut belum ditemukan: "
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
        parse_dates=["Date"],
    )

    latest_features = pd.read_csv(
        ARTIFACT_DIR / "latest_features.csv",
        parse_dates=["Date"],
    )

    # Feature order must follow model metadata.
    reg_features = metadata["regression"]["features"]
    cls_features = metadata["classification"]["features"]

    for frame_name, frame, features in [
        ("latest_features", latest_features, reg_features),
        ("latest_features", latest_features, cls_features),
    ]:

        missing_columns = [
            c for c in features
            if c not in frame.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Kolom fitur {frame_name} tidak lengkap: "
                f"{missing_columns}"
            )

    # Validate commodity-province pairs.
    key_cols = ["Commodity", "Province"]

    if latest_features.duplicated(key_cols).any():
        raise ValueError(
            "latest_features.csv memiliki kombinasi "
            "komoditas-provinsi duplikat."
        )

    # Generate predictions.
    reg_input = latest_features[reg_features].copy()
    cls_input = latest_features[cls_features].copy()

    predicted_price = regression_model.predict(reg_input)

    predicted_class = classification_model.predict(cls_input)

    if len(predicted_price) != len(latest_features):
        raise ValueError(
            "Jumlah prediksi regresi tidak sesuai jumlah seri."
        )

    if len(predicted_class) != len(latest_features):
        raise ValueError(
            "Jumlah prediksi klasifikasi tidak sesuai jumlah seri."
        )

    predictions = latest_features[
        ["Date", "Commodity", "Province", "current_price"]
    ].copy()

    predictions["TargetDate"] = (
        pd.to_datetime(predictions["Date"])
        + pd.to_timedelta(
            int(metadata["horizon_days"]),
            unit="D"
        )
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

    predictions["Perubahan_Persen"] = np.where(
        predictions["current_price"].to_numpy(dtype=float) != 0,
        (
            predictions["Pred_Harga_H30"].to_numpy(dtype=float)
            / predictions["current_price"].to_numpy(dtype=float)
            - 1
        ) * 100,
        np.nan,
    )

    # Optional class probabilities.
    if hasattr(classification_model, "predict_proba"):

        probabilities = classification_model.predict_proba(
            cls_input
        )

        classes = getattr(
            classification_model,
            "classes_",
            range(probabilities.shape[1])
        )

        for idx, class_value in enumerate(classes):

            label = get_class_label(
                class_value,
                labels
            )

            predictions[f"Prob_{label}"] = probabilities[:, idx]

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
        predictions,
    )


# ============================================================
# DISPLAY HELPERS
# ============================================================

def show_historical_warning(metadata: dict):

    last_date = pd.Timestamp(
        metadata["latest_data_date"]
    )

    target_date = last_date + pd.Timedelta(
        days=int(metadata["horizon_days"])
    )

    st.warning(
        f"**Catatan periode data:** dataset berakhir pada "
        f"{format_date(last_date)}. Prediksi H+30 pada aplikasi ini "
        f"merupakan simulasi historis dengan target "
        f"{format_date(target_date)}, bukan prakiraan harga terkini "
        "tahun 2026. Satuan harga mengikuti dataset sumber "
        "dan belum diverifikasi."
    )


def prediction_row(
    predictions: pd.DataFrame,
    commodity: str,
    province: str
):

    selected = predictions[
        (predictions["Commodity"] == commodity)
        & (predictions["Province"] == province)
    ]

    if selected.empty:
        return None

    return selected.iloc[0]


def make_history_chart(
    history: pd.DataFrame,
    forecast_row: pd.Series
):

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
                color="#287A55",
                width=2.5
            ),
            hovertemplate=(
                "%{x|%d %b %Y}<br>"
                "Harga: %{y:,.1f}<extra></extra>"
            ),
        )
    )

    last_date = pd.Timestamp(
        forecast_row["Date"]
    )

    target_date = pd.Timestamp(
        forecast_row["TargetDate"]
    )

    last_price = float(
        forecast_row["current_price"]
    )

    predicted_price = float(
        forecast_row["Pred_Harga_H30"]
    )

    # Connector only, not daily forecast.
    fig.add_trace(
        go.Scatter(
            x=[last_date, target_date],
            y=[last_price, predicted_price],
            mode="lines",
            name="Penghubung ke titik H+30",
            line=dict(
                color="#D59A42",
                width=2,
                dash="dash"
            ),
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[target_date],
            y=[predicted_price],
            mode="markers",
            name="Prediksi H+30",
            marker=dict(
                color="#D59A42",
                size=13,
                symbol="diamond"
            ),
            hovertemplate=(
                "%{x|%d %b %Y}<br>"
                "Prediksi: %{y:,.1f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        height=390,
        margin=dict(
            l=10,
            r=10,
            t=25,
            b=10
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0
        ),
        xaxis_title=None,
        yaxis_title="Nilai harga (satuan dataset)",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    fig.update_xaxes(showgrid=False)

    fig.update_yaxes(
        gridcolor="#E9EFEA",
        zeroline=False
    )

    return fig


def render_header(title: str, subtitle: str):

    st.markdown(f"## {title}")

    st.caption(subtitle)


# ============================================================
# LOAD DATA AND MODELS
# ============================================================

try:

    (
        metadata,
        regression_model,
        classification_model,
        eligible,
        history,
        latest_features,
        predictions,
    ) = load_assets()

except Exception as exc:

    st.error(
        "Aplikasi belum dapat memuat model atau data artefak."
    )

    st.exception(exc)

    st.info(
        "Pastikan folder artifacts tersedia dan install "
        "requirements.txt dengan versi library yang sesuai."
    )

    st.stop()


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:

    st.markdown("## 🌾 PanganPulse")

    st.caption("Food Price Forecasting Dashboard")

    st.markdown("---")

    st.markdown(
        '<a class="sidebar-link" href="#dashboard">'
        '🏠 Dashboard</a>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<a class="sidebar-link" href="#prediksi">'
        '📈 Prediksi H+30</a>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<a class="sidebar-link" href="#perbandingan">'
        '📊 Perbandingan Provinsi</a>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<a class="sidebar-link" href="#tentang">'
        'ℹ️ Tentang Model</a>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("### Ringkasan")

    st.markdown(
        f"- **{eligible['Commodity'].nunique()}** komoditas"
    )

    st.markdown(
        f"- **{eligible['Province'].nunique()}** provinsi"
    )

    st.markdown(
        f"- Horizon **H+{metadata['horizon_days']}**"
    )

    st.markdown(
        f"- Data s.d. **{format_date(metadata['latest_data_date'])}**"
    )


# ============================================================
# 1. DASHBOARD
# ============================================================

st.markdown(
    '<div id="dashboard"></div>',
    unsafe_allow_html=True
)

# ============================================================
# HERO SECTION
# ============================================================

st.markdown("""
<div class="hero">
    <div class="hero-kicker">
        Indonesian Food Price Analysis
    </div>

    <div class="hero-title">
        Pantau Harga. Pahami Prediksi.
    </div>

    <div class="hero-copy">
        Dashboard analisis dan prediksi harga bahan pokok
        menggunakan Machine Learning untuk memahami pola
        harga tiga komoditas utama di Indonesia.
    </div>
</div>
""", unsafe_allow_html=True)


show_historical_warning(metadata)

provinces_count = int(
    eligible["Province"].nunique()
)

commodities_count = int(
    eligible["Commodity"].nunique()
)

last_date = pd.Timestamp(
    metadata["latest_data_date"]
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Komoditas",
    commodities_count
)

c2.metric(
    "Provinsi",
    provinces_count
)

c3.metric(
    "Seri tersedia",
    len(eligible)
)

c4.metric(
    "Horizon prediksi",
    f"H+{metadata['horizon_days']}"
)


# ---------- Historical Price Chart ----------

st.markdown("### Gambaran harga historis")

commodity_options = sorted(
    predictions["Commodity"].unique().tolist()
)

selected_commodity = st.selectbox(
    "Pilih komoditas untuk melihat tren median antarprovinsi",
    commodity_options,
    key="dashboard_commodity",
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
)

fig = px.line(
    median_history,
    x="Date",
    y="Median harga antarprovinsi",
    markers=False,
    title=f"Median harga historis — {selected_commodity}",
)

fig.update_traces(
    line_color="#287A55",
    line_width=2.5
)

fig.update_layout(
    height=370,
    margin=dict(
        l=10,
        r=10,
        t=55,
        b=10
    ),
    plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis_title=None,
    yaxis_title="Nilai harga (satuan dataset)",
)

fig.update_yaxes(
    gridcolor="#E9EFEA"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ---------- Forecast Summary ----------

st.markdown("### Ringkasan prediksi H+30")

summary = (
    predictions
    .groupby("Commodity")
    .agg(
        Seri_tersedia=("Province", "nunique"),
        Median_harga_terakhir=("current_price", "median"),
        Median_prediksi_H30=("Pred_Harga_H30", "median"),
        Median_perubahan_persen=("Perubahan_Persen", "median"),
    )
    .reset_index()
)

display_summary = summary.copy()

for col in [
    "Median_harga_terakhir",
    "Median_prediksi_H30",
]:

    display_summary[col] = display_summary[col].map(
        lambda x: format_number(x, 1)
    )

display_summary["Median_perubahan_persen"] = (
    display_summary["Median_perubahan_persen"]
    .map(
        lambda x: (
            f"{x:+.2f}%"
            if pd.notna(x)
            else "—"
        )
    )
)

display_summary = display_summary.rename(
    columns={
        "Commodity": "Komoditas",
        "Seri_tersedia": "Jumlah provinsi",
        "Median_harga_terakhir": "Median harga terakhir",
        "Median_prediksi_H30": "Median prediksi H+30",
        "Median_perubahan_persen": "Median perubahan",
    }
)

st.dataframe(
    display_summary,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# 2. PREDICTION H+30
# ============================================================

st.divider()

st.markdown(
    '<div id="prediksi" class="section-anchor"></div>',
    unsafe_allow_html=True
)

render_header(
    "Prediksi Harga H+30",
    "Pilih komoditas dan provinsi untuk melihat estimasi "
    "harga 30 hari setelah tanggal data terakhir.",
)

show_historical_warning(metadata)

commodity_options = sorted(
    predictions["Commodity"].unique().tolist()
)

commodity = st.selectbox(
    "Komoditas",
    commodity_options,
    key="prediction_commodity",
)

province_options = sorted(
    predictions.loc[
        predictions["Commodity"] == commodity,
        "Province"
    ].unique().tolist()
)

province = st.selectbox(
    "Provinsi",
    province_options,
    key="prediction_province",
)

row = prediction_row(
    predictions,
    commodity,
    province
)

if row is None:

    st.error(
        "Seri komoditas-provinsi tidak tersedia "
        "pada data penelitian."
    )

else:

    st.markdown("### Hasil prediksi")

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "Harga terakhir",
        format_number(row["current_price"]),
        help="Harga pada tanggal pengamatan terakhir.",
    )

    m2.metric(
        "Prediksi harga H+30",
        format_number(row["Pred_Harga_H30"]),
        delta=(
            f"{row['Perubahan_Persen']:+.2f}% "
            "dari harga terakhir"
        ),
        help="Estimasi harga tepat 30 hari setelah tanggal pengamatan.",
    )

    m3.metric(
        "Klasifikasi perubahan",
        row["Pred_Kelas_H30"],
        help=(
            "Kategori berdasarkan perubahan harga H+30: "
            "Turun, Stabil, atau Naik."
        ),
    )

    st.caption(
        f"Tanggal pengamatan: **{format_date(row['Date'])}**  ·  "
        f"Tanggal target: **{format_date(row['TargetDate'])}**"
    )

    # Historical chart and forecast point.
    st.markdown(
        "### Grafik harga historis dan titik prediksi"
    )

    st.plotly_chart(
        make_history_chart(history, row),
        use_container_width=True,
    )

    st.caption(
        "Garis putus-putus hanya menghubungkan harga terakhir "
        "dengan titik target H+30. Garis tersebut bukan "
        "prediksi harga harian di antara kedua tanggal."
    )

    # Classification probability.
    probability_cols = [
        col for col in predictions.columns
        if col.startswith("Prob_")
    ]

    if probability_cols:

        st.markdown("### Probabilitas klasifikasi")

        prob_values = []

        for col in probability_cols:

            label = col.replace("Prob_", "")

            prob_values.append(
                {
                    "Kelas": label,
                    "Probabilitas": float(row[col])
                }
            )

        prob_df = pd.DataFrame(prob_values)

        fig_prob = px.bar(
            prob_df,
            x="Kelas",
            y="Probabilitas",
            text=prob_df["Probabilitas"].map(
                lambda x: f"{x:.1%}"
            ),
            color="Kelas",
            color_discrete_map={
                "Turun": "#D66B62",
                "Stabil": "#8A9A8D",
                "Naik": "#287A55",
            },
        )

        fig_prob.update_layout(
            showlegend=False,
            height=280,
            yaxis=dict(
                tickformat=".0%",
                range=[0, 1]
            ),
            margin=dict(
                l=10,
                r=10,
                t=20,
                b=10
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        st.plotly_chart(
            fig_prob,
            use_container_width=True
        )

    # Historical data details.
    with st.expander("Lihat detail data seri"):

        series_history = history[
            (history["Commodity"] == commodity)
            & (history["Province"] == province)
        ].sort_values(
            "Date",
            ascending=False
        )

        st.dataframe(
            series_history.rename(
                columns={
                    "Date": "Tanggal",
                    "Commodity": "Komoditas",
                    "Province": "Provinsi",
                    "Price": "Harga",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# 3. PROVINCE COMPARISON
# ============================================================

st.divider()

st.markdown(
    '<div id="perbandingan" class="section-anchor"></div>',
    unsafe_allow_html=True
)

render_header(
    "Perbandingan Antarprovinsi",
    "Bandingkan harga terakhir dan estimasi harga H+30 "
    "untuk satu komoditas pada seluruh provinsi yang tersedia.",
)

show_historical_warning(metadata)

commodity_options = sorted(
    predictions["Commodity"].unique().tolist()
)

commodity = st.selectbox(
    "Pilih komoditas",
    commodity_options,
    key="comparison_commodity",
)

comp = predictions[
    predictions["Commodity"] == commodity
].copy()

sort_option = st.selectbox(
    "Urutkan berdasarkan",
    [
        "Prediksi harga tertinggi",
        "Prediksi harga terendah",
        "Persentase kenaikan tertinggi",
        "Persentase penurunan terbesar",
        "Nama provinsi",
    ],
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


# ---------- Comparison Table ----------

table = comp[
    [
        "Province",
        "current_price",
        "Pred_Harga_H30",
        "Perubahan_Persen",
        "Pred_Kelas_H30",
    ]
].rename(
    columns={
        "Province": "Provinsi",
        "current_price": "Harga terakhir",
        "Pred_Harga_H30": "Prediksi H+30",
        "Perubahan_Persen": "Perubahan (%)",
        "Pred_Kelas_H30": "Klasifikasi",
    }
)

st.markdown(f"### {commodity}")

st.caption(
    f"{len(comp)} seri provinsi tersedia untuk komoditas ini."
)


# ---------- Comparison Chart ----------

fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=comp["Province"],
        y=comp["current_price"],
        name="Harga terakhir",
        marker_color="#9CB7A4",
    )
)

fig.add_trace(
    go.Bar(
        x=comp["Province"],
        y=comp["Pred_Harga_H30"],
        name="Prediksi H+30",
        marker_color="#287A55",
    )
)

fig.update_layout(
    barmode="group",
    height=440,
    margin=dict(
        l=10,
        r=10,
        t=20,
        b=10
    ),
    xaxis_title=None,
    yaxis_title="Nilai harga (satuan dataset)",
    plot_bgcolor="white",
    paper_bgcolor="white",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        x=0
    ),
)

fig.update_yaxes(
    gridcolor="#E9EFEA"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ---------- Comparison Results ----------

st.markdown("### Tabel hasil")

st.dataframe(
    table.round(
        {
            "Harga terakhir": 1,
            "Prediksi H+30": 1,
            "Perubahan (%)": 2
        }
    ),
    use_container_width=True,
    hide_index=True,
)


# ---------- Download CSV ----------

csv_data = comp.to_csv(
    index=False
).encode("utf-8-sig")

st.download_button(
    "⬇️ Unduh hasil perbandingan (CSV)",
    data=csv_data,
    file_name=(
        f"panganpulse_"
        f"{commodity.lower().replace(' ', '_')}_h30.csv"
    ),
    mime="text/csv",
)


# ============================================================
# 4. ABOUT MODEL
# ============================================================

st.divider()

st.markdown(
    '<div id="tentang" class="section-anchor"></div>',
    unsafe_allow_html=True
)

render_header(
    "Tentang Model dan Penelitian",
    "Informasi algoritma, metrik evaluasi, cakupan data, "
    "dan keterbatasan model yang digunakan.",
)

show_historical_warning(metadata)

reg = metadata["regression"]

cls = metadata["classification"]

scope = metadata["scope"]


# ---------- Regression Model ----------

st.markdown("### Model regresi")

st.subheader(reg["candidate"])

st.write(
    f"Model dipilih berdasarkan **{reg['selection_rule']}**."
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


# ---------- Classification Model ----------

st.markdown("### Model klasifikasi")

st.subheader(cls["candidate"])

st.write(
    f"Model dipilih berdasarkan **{cls['selection_rule']}**."
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


# ---------- Research Scope ----------

st.markdown("### Cakupan penelitian")

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

st.write(
    ", ".join(scope["commodities"])
)


# ---------- Features ----------

with st.expander("Lihat fitur yang digunakan"):

    st.write("**Fitur regresi:**")

    st.code(
        ", ".join(reg["features"])
    )

    st.write("**Fitur klasifikasi:**")

    st.code(
        ", ".join(cls["features"])
    )


# ---------- Model Configuration ----------

with st.expander("Lihat konfigurasi model"):

    st.write("**Konfigurasi regresi:**")

    st.json(
        reg["params"]
    )

    st.write("**Konfigurasi klasifikasi:**")

    st.json(
        cls["params"]
    )


# ---------- Evaluation Methodology ----------

st.markdown("### Metodologi evaluasi")

validation = metadata["validation_design"]

folds_value = validation.get("folds", [])

fold_count = (
    len(folds_value)
    if isinstance(folds_value, (list, tuple, dict))
    else int(folds_value)
)

st.write(
    f"Evaluasi menggunakan **{fold_count} fold "
    "expanding rolling-origin**. "
    "Periode validasi berlangsung Januari–Mei 2024."
)

st.write(
    f"Jumlah baris training final: "
    f"**{metadata['training_data']['rows']:,}**."
)


# ---------- Limitations ----------

st.markdown("### Keterbatasan")

for limitation in metadata.get("limitations", []):

    st.markdown(
        f"- {limitation}"
    )

st.info(
    "Angka performa di atas merupakan hasil validasi "
    "rolling-origin yang juga digunakan selama proses "
    "pemilihan model. Angka tersebut bukan hasil "
    "pengujian pada holdout independen."
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "PanganPulse · Academic Machine Learning Project · "
    "Prediksi historis H+30 · "
    "Sumber dan satuan mengikuti dataset penelitian."
)
