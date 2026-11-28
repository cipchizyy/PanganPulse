from __future__ import annotations

import json
import warnings
import html as html_lib
from pathlib import Path
from urllib.request import Request, urlopen

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


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

# ---------- Styling ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    :root {
        --pp-yellow: #F6C445;
        --pp-orange: #F05A28;
        --pp-green: #8BC34A;
        --pp-cream: #FFF3D6;
        --pp-brown: #3B2F2F;
        --pp-ink: #3B2F2F;
        --pp-muted: #78685D;
        --pp-card: #FFFCF4;
        --pp-line: rgba(59, 47, 47, .12);
        --pp-shadow: 0 12px 30px rgba(82, 54, 31, .08);
    }

    html, body, [class*="css"], button, input, textarea, select {
        font-family: 'Poppins', 'Plus Jakarta Sans', sans-serif !important;
    }

    .stApp,
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at 92% 4%, rgba(246,196,69,.18), transparent 23rem),
            radial-gradient(circle at 8% 30%, rgba(139,195,74,.10), transparent 24rem),
            linear-gradient(135deg, #FFF9EC 0%, #FFF3D6 52%, #FFF8E9 100%);
        color: var(--pp-ink);
    }

    [data-testid="stHeader"] {
        background: rgba(255,243,214,.72);
        backdrop-filter: blur(14px);
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1500px;
        padding-top: 1.6rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4, p, label, span, div {
        font-family: 'Poppins', 'Plus Jakarta Sans', sans-serif;
    }

    h1, h2, h3 {
        color: var(--pp-brown);
        letter-spacing: -.035em;
        font-weight: 700;
    }

    h1 { font-size: clamp(1.8rem, 3vw, 2.55rem); }
    h2 { font-size: clamp(1.35rem, 2vw, 1.8rem); }
    h3 { font-size: 1.13rem; }

    p, li, label, .stMarkdown {
        color: var(--pp-ink);
    }

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2.8rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #3B2F2F 0%, #302525 100%);
        border-right: 1px solid rgba(255,243,214,.13);
        box-shadow: 12px 0 34px rgba(59,47,47,.10);
    }
    [data-testid="stSidebar"] > div:first-child {
        background: transparent;
    }
    [data-testid="stSidebar"] * {
        color: #FFF3D6 !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: rgba(255,243,214,.76) !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,243,214,.16);
        margin: 1.05rem 0;
    }
    .brand-wrap {
        padding: .8rem .25rem 1rem;
        margin-bottom: .35rem;
    }
    .brand-mark {
        display: inline-flex;
        width: 48px;
        height: 48px;
        align-items: center;
        justify-content: center;
        border-radius: 16px;
        background: linear-gradient(135deg, var(--pp-yellow), var(--pp-orange));
        color: var(--pp-brown) !important;
        font-size: 1.55rem;
        box-shadow: 0 8px 20px rgba(240,90,40,.24);
        margin-bottom: .7rem;
    }
    .brand {
        font-weight: 800;
        font-size: 1.55rem;
        line-height: 1.15;
        letter-spacing: -.05em;
        color: #FFF3D6 !important;
        margin: 0;
    }
    .brand .pulse { color: var(--pp-yellow) !important; }
    .brand-sub {
        color: rgba(255,243,214,.68) !important;
        font-size: .72rem;
        line-height: 1.55;
        margin-top: .35rem;
        letter-spacing: .02em;
    }
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: rgba(255,243,214,.58) !important;
        font-size: .68rem;
        font-weight: 700;
        letter-spacing: .12em;
        text-transform: uppercase;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
        gap: .42rem;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label {
        border: 1px solid transparent;
        border-radius: 13px;
        padding: .62rem .72rem;
        background: rgba(255,243,214,.045);
        transition: all .22s ease;
        cursor: pointer;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:hover {
        background: rgba(255,243,214,.13);
        border-color: rgba(255,243,214,.18);
        transform: translateX(4px);
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(110deg, #F05A28, #F47A35);
        border-color: rgba(255,243,214,.2);
        box-shadow: 0 7px 18px rgba(240,90,40,.24);
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label p {
        font-weight: 600;
        font-size: .88rem;
    }
    .sidebar-note {
        padding: .85rem .9rem;
        border: 1px solid rgba(255,243,214,.14);
        border-radius: 15px;
        background: rgba(255,243,214,.07);
        margin: .35rem 0;
    }
    .sidebar-note strong {
        color: var(--pp-yellow) !important;
    }

    /* Header and hero */
    .page-heading {
        margin: .2rem 0 1rem;
        animation: ppFadeUp .55s ease both;
    }
    .page-eyebrow, .hero-kicker {
        display: inline-flex;
        align-items: center;
        gap: .4rem;
        color: #A84A25;
        background: rgba(240,90,40,.09);
        border: 1px solid rgba(240,90,40,.14);
        border-radius: 999px;
        padding: .32rem .72rem;
        text-transform: uppercase;
        letter-spacing: .09em;
        font-size: .67rem;
        font-weight: 700;
    }
    .page-heading h1 {
        margin: .55rem 0 .25rem;
        line-height: 1.18;
    }
    .page-heading p {
        margin: 0;
        color: var(--pp-muted);
        font-size: .91rem;
        max-width: 850px;
    }
    .hero {
        position: relative;
        overflow: hidden;
        isolation: isolate;
        background:
            radial-gradient(circle at 90% 15%, rgba(246,196,69,.34), transparent 12rem),
            linear-gradient(115deg, #FFF9EA 0%, #FFE9C0 58%, #FCE0B4 100%);
        color: var(--pp-brown);
        padding: 1.65rem 1.8rem;
        border: 1px solid rgba(240,90,40,.13);
        border-radius: 25px;
        margin-bottom: 1.2rem;
        box-shadow: 0 14px 34px rgba(109,72,39,.09);
        animation: ppFadeUp .65s ease both;
    }
    .hero:after {
        content: "🌾";
        position: absolute;
        right: 3.3%;
        bottom: -1.8rem;
        z-index: -1;
        font-size: 7rem;
        opacity: .17;
        transform: rotate(-13deg);
    }
    .hero-kicker {
        color: #8B4A26;
        background: rgba(246,196,69,.28);
        border-color: rgba(246,196,69,.45);
    }
    .hero-title {
        font-weight: 800;
        color: var(--pp-brown);
        font-size: clamp(1.75rem, 3vw, 2.55rem);
        line-height: 1.15;
        margin: .55rem 0 .45rem;
        letter-spacing: -.05em;
    }
    .hero-copy {
        color: #665248;
        font-size: .92rem;
        line-height: 1.7;
        max-width: 760px;
    }
    .hero-pill {
        display: inline-flex;
        margin-top: .9rem;
        padding: .38rem .75rem;
        border-radius: 999px;
        background: rgba(139,195,74,.17);
        color: #456D22;
        font-size: .72rem;
        font-weight: 700;
    }

    /* Metrics and cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #FFFEF9, #FFF4DE);
        border: 1px solid rgba(59,47,47,.09);
        border-radius: 19px;
        padding: 1rem 1.05rem;
        min-height: 116px;
        box-shadow: var(--pp-shadow);
        transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease;
        animation: ppFadeUp .55s ease both;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        border-color: rgba(240,90,40,.30);
        box-shadow: 0 17px 34px rgba(82,54,31,.14);
    }
    div[data-testid="stMetricLabel"] p {
        color: #76665A !important;
        font-size: .78rem;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] {
        color: var(--pp-brown) !important;
        font-size: clamp(1.45rem, 2.3vw, 2rem);
        font-weight: 800;
        letter-spacing: -.04em;
    }
    div[data-testid="stMetricDelta"] {
        font-size: .75rem;
        font-weight: 600;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255,252,244,.78);
        border: 1px solid rgba(59,47,47,.10) !important;
        border-radius: 20px !important;
        box-shadow: var(--pp-shadow);
        transition: transform .24s ease, box-shadow .24s ease;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 16px 34px rgba(82,54,31,.12);
    }
    [data-testid="stPlotlyChart"] {
        background: rgba(255,252,244,.88);
        border: 1px solid rgba(59,47,47,.09);
        border-radius: 19px;
        padding: .45rem .5rem .2rem;
        box-shadow: 0 8px 24px rgba(82,54,31,.055);
        animation: ppFadeUp .65s ease both;
    }
    .section-kicker {
        color: #A85A2D;
        text-transform: uppercase;
        letter-spacing: .12em;
        font-size: .68rem;
        font-weight: 700;
        margin: .35rem 0 .25rem;
    }
    .section-title {
        color: var(--pp-brown);
        font-size: 1.17rem;
        font-weight: 700;
        margin: 0 0 .25rem;
    }
    .section-subtitle {
        color: var(--pp-muted);
        font-size: .8rem;
        margin: 0 0 .8rem;
    }

    /* Inputs, tabs, buttons — force readable light controls even under dark Streamlit theme */
    [data-testid="stSelectbox"] label,
    [data-testid="stTextInput"] label,
    [data-testid="stSlider"] label {
        color: #3B2F2F !important;
        font-weight: 600;
        font-size: .83rem;
    }

    [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        background: #FFFCF4 !important;
        color: #3B2F2F !important;
        border: 1px solid rgba(59,47,47,.20) !important;
        border-radius: 13px !important;
        min-height: 44px;
        box-shadow: none !important;
        transition: border-color .2s ease, box-shadow .2s ease;
    }

    [data-testid="stSelectbox"] div[data-baseweb="select"] span,
    [data-testid="stSelectbox"] div[data-baseweb="select"] input,
    [data-testid="stSelectbox"] div[data-baseweb="select"] div,
    [data-testid="stSelectbox"] div[data-baseweb="select"] svg {
        color: #3B2F2F !important;
        fill: #3B2F2F !important;
        -webkit-text-fill-color: #3B2F2F !important;
    }

    [data-testid="stSelectbox"] div[data-baseweb="select"] > div:focus-within {
        border-color: #F05A28 !important;
        box-shadow: 0 0 0 3px rgba(240,90,40,.12) !important;
    }

    /* Dropdown menu and options */
    div[data-baseweb="popover"] > div,
    div[data-baseweb="popover"] ul,
    ul[role="listbox"] {
        background: #FFFCF4 !important;
        border: 1px solid rgba(59,47,47,.14) !important;
    }

    [role="option"],
    [role="listbox"] *,
    div[data-baseweb="menu"] * {
        color: #3B2F2F !important;
        -webkit-text-fill-color: #3B2F2F !important;
    }

    [role="option"]:hover,
    [role="option"][aria-selected="true"] {
        background: #FCE8C3 !important;
    }
    [data-testid="stTabs"] [role="tab"] {
        border-radius: 12px 12px 0 0;
        color: #78685D;
        font-weight: 600;
        transition: all .2s ease;
    }
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        color: var(--pp-orange);
        border-bottom-color: var(--pp-orange);
    }
    .stButton > button, .stDownloadButton > button {
        background: linear-gradient(115deg, #F05A28, #F47A32);
        color: #FFFFFF !important;
        border: 1px solid rgba(240,90,40,.25);
        border-radius: 13px;
        min-height: 2.75rem;
        padding: .55rem 1.05rem;
        font-weight: 700;
        box-shadow: 0 7px 17px rgba(240,90,40,.20);
        transition: transform .2s ease, box-shadow .2s ease, filter .2s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        color: #FFFFFF !important;
        transform: translateY(-3px);
        filter: saturate(1.08);
        box-shadow: 0 12px 24px rgba(240,90,40,.28);
        border-color: rgba(240,90,40,.5);
    }
    .stButton > button:active, .stDownloadButton > button:active {
        transform: translateY(-1px) scale(.99);
    }
    .stButton > button:focus, .stDownloadButton > button:focus {
        box-shadow: 0 0 0 3px rgba(240,90,40,.18);
    }

    /* App-owned alert components: consistent cream background and dark readable text */
    .pp-alert,
    .pp-info {
        display: block;
        width: 100%;
        box-sizing: border-box;
        padding: .95rem 1.05rem;
        margin: .45rem 0 1rem;
        border-radius: 15px;
        color: #3B2F2F !important;
        background: #FFF5DE !important;
        border: 1px solid rgba(240,90,40,.24) !important;
        box-shadow: 0 7px 18px rgba(82,54,31,.045);
        font-size: .84rem;
        line-height: 1.65;
    }

    .pp-alert *,
    .pp-info * {
        color: #3B2F2F !important;
        -webkit-text-fill-color: #3B2F2F !important;
        opacity: 1 !important;
    }

    .pp-alert strong,
    .pp-info strong {
        color: #A84A25 !important;
        font-weight: 800;
    }

    /* Alerts — override both the outer wrapper and Streamlit's inner themed content */
    [data-testid="stAlert"],
    [data-testid="stAlert"] > div,
    [data-testid="stAlert"] [data-testid="stMarkdownContainer"] {
        background: #FFF5DE !important;
        color: #3B2F2F !important;
        border-color: rgba(240,90,40,.24) !important;
        border-radius: 15px !important;
    }

    [data-testid="stAlert"] *,
    [data-testid="stAlert"] p,
    [data-testid="stAlert"] span,
    [data-testid="stAlert"] div,
    [data-testid="stAlert"] li {
        color: #3B2F2F !important;
        -webkit-text-fill-color: #3B2F2F !important;
        opacity: 1 !important;
    }

    [data-testid="stAlert"] svg {
        color: #A84A25 !important;
        fill: #A84A25 !important;
    }

    /* Captions and regular text should never inherit low-contrast dark-theme colors */
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] *,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li {
        color: #5F5047;
    }

    div[data-testid="stMetricDelta"],
    div[data-testid="stMetricDelta"] * {
        color: #3B2F2F !important;
        -webkit-text-fill-color: #3B2F2F !important;
    }
    [data-testid="stExpander"] {
        border: 1px solid rgba(59,47,47,.12);
        border-radius: 15px;
        background: rgba(255,252,244,.72);
        overflow: hidden;
    }
    [data-testid="stExpander"] summary {
        font-weight: 600;
    }

    /* Hoverable HTML tables */
    .pp-table-wrap {
        width: 100%;
        overflow-x: auto;
        border: 1px solid rgba(59,47,47,.11);
        border-radius: 17px;
        background: #FFFCF5;
        box-shadow: 0 8px 22px rgba(82,54,31,.055);
    }
    table.pp-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        min-width: 640px;
        font-size: .81rem;
    }
    .pp-table thead th {
        position: sticky;
        top: 0;
        background: #FCE8C3;
        color: #49372F;
        padding: .85rem .9rem;
        text-align: left;
        font-weight: 700;
        border-bottom: 1px solid rgba(59,47,47,.10);
        white-space: nowrap;
    }
    .pp-table tbody td {
        color: #493D36;
        padding: .78rem .9rem;
        border-bottom: 1px solid rgba(59,47,47,.065);
        background: transparent;
        transition: background .18s ease, color .18s ease;
    }
    .pp-table tbody tr:nth-child(even) td {
        background: rgba(246,196,69,.075);
    }
    .pp-table tbody tr:hover td {
        background: rgba(139,195,74,.20);
        color: #30251F;
    }
    .pp-table tbody tr:last-child td { border-bottom: 0; }
    .pp-positive { color: #4E8425 !important; font-weight: 700; }
    .pp-negative { color: #D94F24 !important; font-weight: 700; }
    .pp-neutral { color: #8A6B25 !important; font-weight: 700; }
    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(59,47,47,.12);
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 7px 20px rgba(82,54,31,.05);
    }

    .small-muted {
        color: var(--pp-muted);
        font-size: .8rem;
    }
    .footer-note {
        color: #8A776A;
        font-size: .72rem;
        text-align: center;
        padding: .6rem 0 .2rem;
    }

    @keyframes ppFadeUp {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes ppPulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(240,90,40,0); }
        50% { box-shadow: 0 0 0 7px rgba(240,90,40,.08); }
    }

    @media (max-width: 900px) {
        [data-testid="stMainBlockContainer"] { padding: 1rem .9rem 2rem; }
        .hero { padding: 1.25rem; border-radius: 19px; }
        .hero:after { right: -1rem; font-size: 5rem; }
        div[data-testid="stMetric"] { min-height: 100px; padding: .8rem; }
    }
    @media (prefers-reduced-motion: reduce) {
        *, *:before, *:after {
            animation-duration: .01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: .01ms !important;
            scroll-behavior: auto !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Helpers ----------
def format_number(value: float | int | None, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "—"
    formatted = f"{float(value):,.{decimals}f}"
    return formatted.replace(",", "_").replace(".", ",").replace("_", ".")


def format_date(value) -> str:
    if pd.isna(value):
        return "—"
    return pd.Timestamp(value).strftime("%d %B %Y")


def get_class_label(value, labels: dict) -> str:
    try:
        key = str(int(value))
    except (TypeError, ValueError):
        key = str(value)
    return labels.get(key, f"Kelas {key}")


@st.cache_resource(show_spinner="Memuat model dan data penelitian...")
def load_assets():
    required_files = [
        "regression_model.joblib",
        "classification_model.joblib",
        "model_metadata.json",
        "eligible_series.csv",
        "latest_history.csv",
        "latest_features.csv",
        "external_features_daily.csv",
    ]
    missing = [
        name for name in required_files
        if not (ARTIFACT_DIR / name).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Artefak berikut belum ditemukan: " + ", ".join(missing)
        )

    metadata = json.loads(
        (ARTIFACT_DIR / "model_metadata.json").read_text(encoding="utf-8")
    )

    regression_model = joblib.load(
        ARTIFACT_DIR / "regression_model.joblib"
    )
    classification_model = joblib.load(
        ARTIFACT_DIR / "classification_model.joblib"
    )

    eligible = pd.read_csv(ARTIFACT_DIR / "eligible_series.csv")
    history = pd.read_csv(
        ARTIFACT_DIR / "latest_history.csv",
        parse_dates=["Date"],
    )
    latest_features = pd.read_csv(
        ARTIFACT_DIR / "latest_features.csv",
        parse_dates=["Date"],
    )
    external_daily = pd.read_csv(
        ARTIFACT_DIR / "external_features_daily.csv",
        parse_dates=["Date"],
    )
    external_return_cols = [
        "wheat_return_30",
        "palm_oil_return_30",
        "sugar_return_30",
        "usdidr_return_30",
        "crude_oil_return_30",
    ]
    missing_external = [
        col for col in external_return_cols if col not in external_daily.columns
    ]
    if missing_external:
        raise ValueError(
            "Kolom faktor eksternal tidak lengkap: " + ", ".join(missing_external)
        )
    latest_features = latest_features.merge(
        external_daily[["Date"] + external_return_cols],
        on="Date",
        how="left",
        validate="many_to_one",
    )

    # Keep feature order exactly as recorded in model metadata.
    reg_features = metadata["regression"]["features"]
    cls_features = metadata["classification"]["features"]

    for frame_name, frame, features in [
        ("latest_features", latest_features, reg_features),
        ("latest_features", latest_features, cls_features),
    ]:
        missing_columns = [c for c in features if c not in frame.columns]
        if missing_columns:
            raise ValueError(
                f"Kolom fitur {frame_name} tidak lengkap: {missing_columns}"
            )

    # Validate one record per eligible commodity-province pair.
    key_cols = ["Commodity", "Province"]
    if latest_features.duplicated(key_cols).any():
        raise ValueError(
            "latest_features.csv memiliki kombinasi komoditas-provinsi duplikat."
        )

    # Generate predictions for every eligible series once.
    reg_input = latest_features[reg_features].copy()
    cls_input = latest_features[cls_features].copy()

    predicted_price = regression_model.predict(reg_input)
    predicted_class = classification_model.predict(cls_input)

    if len(predicted_price) != len(latest_features):
        raise ValueError("Jumlah prediksi regresi tidak sesuai jumlah seri.")
    if len(predicted_class) != len(latest_features):
        raise ValueError("Jumlah prediksi klasifikasi tidak sesuai jumlah seri.")

    predictions = latest_features[
        ["Date", "Commodity", "Province", "current_price"] + external_return_cols
    ].copy()

    predictions["TargetDate"] = (
        pd.to_datetime(predictions["Date"])
        + pd.to_timedelta(int(metadata["horizon_days"]), unit="D")
    )
    predictions["Pred_Harga_H30"] = np.asarray(predicted_price, dtype=float)
    labels = metadata["target_classification"]["labels"]
    predictions["Pred_Kelas_H30"] = [
        get_class_label(value, labels) for value in predicted_class
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
        probabilities = classification_model.predict_proba(cls_input)
        classes = getattr(classification_model, "classes_", range(probabilities.shape[1]))
        for idx, class_value in enumerate(classes):
            label = get_class_label(class_value, labels)
            predictions[f"Prob_{label}"] = probabilities[:, idx]

    history["Date"] = pd.to_datetime(history["Date"])
    latest_features["Date"] = pd.to_datetime(latest_features["Date"])
    predictions["Date"] = pd.to_datetime(predictions["Date"])
    predictions["TargetDate"] = pd.to_datetime(predictions["TargetDate"])

    return (
        metadata,
        regression_model,
        classification_model,
        eligible,
        history,
        latest_features,
        predictions,
    )


def show_historical_warning(metadata: dict):
    last_date = pd.Timestamp(metadata["latest_data_date"])
    target_date = last_date + pd.Timedelta(
        days=int(metadata["horizon_days"])
    )

    message = (
        f"<strong>Catatan periode data:</strong> Dataset berakhir pada "
        f"{html_lib.escape(format_date(last_date))}. Prediksi H+30 pada aplikasi ini "
        f"merupakan simulasi historis dengan target "
        f"{html_lib.escape(format_date(target_date))}, bukan prakiraan harga terkini "
        "tahun 2026. Satuan harga mengikuti dataset sumber dan belum diverifikasi."
    )

    render_html_block(f'<div class="pp-alert">{message}</div>')


def prediction_row(predictions: pd.DataFrame, commodity: str, province: str):
    selected = predictions[
        (predictions["Commodity"] == commodity)
        & (predictions["Province"] == province)
    ]
    if selected.empty:
        return None
    return selected.iloc[0]


def make_history_chart(history: pd.DataFrame, forecast_row: pd.Series):
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
            line=dict(color="#F05A28", width=3.2, shape="spline"),
            hovertemplate="%{x|%d %b %Y}<br>Harga: %{y:,.1f}<extra></extra>",
        )
    )

    last_date = pd.Timestamp(forecast_row["Date"])
    target_date = pd.Timestamp(forecast_row["TargetDate"])
    last_price = float(forecast_row["current_price"])
    predicted_price = float(forecast_row["Pred_Harga_H30"])

    # A dashed connector marks only the start and H+30 target;
    # it is not a daily forecast path.
    fig.add_trace(
        go.Scatter(
            x=[last_date, target_date],
            y=[last_price, predicted_price],
            mode="lines",
            name="Penghubung ke titik H+30",
            line=dict(color="#8BC34A", width=2.5, dash="dash"),
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[target_date],
            y=[predicted_price],
            mode="markers",
            name="Prediksi H+30",
            marker=dict(color="#F6C445", line=dict(color="#3B2F2F", width=1.2), size=14, symbol="diamond"),
            hovertemplate="%{x|%d %b %Y}<br>Prediksi: %{y:,.1f}<extra></extra>",
        )
    )

    fig.update_layout(
        height=410,
        margin=dict(l=10, r=10, t=45, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis_title=None,
        yaxis_title="Nilai harga (satuan dataset)",
    )
    style_plotly(fig, 410)
    add_line_replay(fig, hist["Date"], hist["Price"], trace_index=0)
    return fig


FACTOR_TREND_THRESHOLD_PCT = 1.0

EXTERNAL_FACTOR_CONFIG = {
    "Minyak Goreng Kemasan Sederhana": [
        (
            "palm_oil_return_30",
            "Harga minyak sawit global",
            "Bahan baku utama minyak goreng; kenaikan dapat memberi tekanan pada biaya bahan baku.",
            True,
        ),
        (
            "crude_oil_return_30",
            "Harga minyak mentah WTI",
            "Indikator biaya energi dan transportasi; pengaruhnya bersifat tidak langsung.",
            True,
        ),
        (
            "usdidr_return_30",
            "Kurs USD/IDR",
            "Kenaikan USD/IDR menunjukkan pelemahan rupiah dan dapat meningkatkan biaya input impor.",
            True,
        ),
    ],
    "Beras Medium": [
        (
            "usdidr_return_30",
            "Kurs USD/IDR",
            "Dapat memengaruhi biaya input impor dan distribusi; bukan penentu tunggal harga beras.",
            True,
        ),
        (
            "crude_oil_return_30",
            "Harga minyak mentah WTI",
            "Indikator biaya energi dan transportasi yang dapat memengaruhi distribusi.",
            True,
        ),
        (
            "wheat_return_30",
            "Harga gandum global",
            "Hanya indikator pasar serealia global; bukan harga beras atau bahan baku beras secara langsung.",
            False,
        ),
    ],
    "Cabai Rawit Merah": [
        (
            "usdidr_return_30",
            "Kurs USD/IDR",
            "Dapat memengaruhi biaya input pertanian yang berasal dari impor.",
            True,
        ),
        (
            "crude_oil_return_30",
            "Harga minyak mentah WTI",
            "Indikator biaya energi dan transportasi hasil panen.",
            True,
        ),
    ],
}


def build_external_factor_explanation(
    commodity: str,
    predicted_class: str,
    row: pd.Series,
):
    definitions = EXTERNAL_FACTOR_CONFIG.get(commodity, [])
    factor_rows = []
    upward_pressure = 0
    downward_pressure = 0

    for feature, factor_name, influence, counts_as_pressure in definitions:
        value = row.get(feature, np.nan)
        if pd.isna(value):
            status = "Tidak tersedia"
            detail = "Nilai faktor eksternal tidak tersedia pada tanggal pengamatan."
        else:
            value = float(value)
            if feature == "wheat_return_30":
                if value >= FACTOR_TREND_THRESHOLD_PCT:
                    status = "Naik"
                elif value <= -FACTOR_TREND_THRESHOLD_PCT:
                    status = "Turun"
                else:
                    status = "Relatif stabil"
                detail = (
                    "Perubahan ini hanya menggambarkan pasar gandum global; "
                    "tidak dapat dibaca sebagai perubahan harga beras secara langsung."
                )
            elif value >= FACTOR_TREND_THRESHOLD_PCT:
                status = "Naik"
                if feature == "usdidr_return_30":
                    detail = "Rupiah melemah; berpotensi menambah tekanan biaya terkait impor."
                else:
                    detail = "Kenaikan dapat menambah tekanan biaya pada jalur yang relevan."
                if counts_as_pressure:
                    upward_pressure += 1
            elif value <= -FACTOR_TREND_THRESHOLD_PCT:
                status = "Turun"
                if feature == "usdidr_return_30":
                    detail = "Rupiah menguat; berpotensi mengurangi tekanan biaya terkait impor."
                else:
                    detail = "Penurunan dapat mengurangi tekanan biaya pada jalur yang relevan."
                if counts_as_pressure:
                    downward_pressure += 1
            else:
                status = "Relatif stabil"
                detail = "Perubahannya belum melewati ambang interpretasi ±1%."

        factor_rows.append(
            {
                "Faktor eksternal": factor_name,
                "Perubahan 30 hari": (
                    f"{float(value):+.2f}%" if pd.notna(value) else "—"
                ),
                "Kondisi": status,
                "Potensi pengaruh": influence,
                "Keterangan": detail,
            }
        )

    if predicted_class == "Naik":
        if upward_pressure > downward_pressure:
            summary = (
                "Prediksi klasifikasi menunjukkan kenaikan. Beberapa indikator biaya "
                "eksternal juga bergerak naik sehingga berpotensi memberi tekanan "
                "kenaikan harga. Hubungan ini merupakan indikasi, bukan bukti sebab-akibat."
            )
        elif upward_pressure == 0:
            summary = (
                "Prediksi klasifikasi menunjukkan kenaikan, tetapi faktor eksternal "
                "yang tersedia belum menunjukkan tekanan naik yang dominan. Perubahan "
                "dapat berkaitan dengan pola harga lokal atau faktor lain yang tidak "
                "tercakup dalam dataset."
            )
        else:
            summary = (
                "Prediksi klasifikasi menunjukkan kenaikan, tetapi indikator eksternal "
                "bergerak beragam. Belum ada satu arah tekanan biaya yang dominan."
            )
    elif predicted_class == "Turun":
        if downward_pressure > upward_pressure:
            summary = (
                "Prediksi klasifikasi menunjukkan penurunan. Beberapa indikator biaya "
                "eksternal juga bergerak turun sehingga berpotensi mengurangi tekanan "
                "biaya. Hubungan ini bukan bukti bahwa faktor tersebut menyebabkan penurunan."
            )
        elif downward_pressure == 0:
            summary = (
                "Prediksi klasifikasi menunjukkan penurunan, tetapi faktor eksternal "
                "yang tersedia belum menunjukkan penurunan biaya yang dominan. "
                "Perubahan dapat dipengaruhi pola harga lokal atau faktor lain."
            )
        else:
            summary = (
                "Prediksi klasifikasi menunjukkan penurunan, tetapi indikator eksternal "
                "bergerak beragam sehingga pengaruhnya belum menunjukkan arah yang jelas."
            )
    else:
        summary = (
            "Prediksi klasifikasi menunjukkan kondisi relatif stabil. Indikator eksternal "
            "yang ditampilkan merupakan konteks tambahan dan belum menunjukkan tekanan "
            "yang cukup konsisten untuk menjelaskan perubahan harga."
        )

    if commodity == "Cabai Rawit Merah":
        limitation = (
            "Catatan: dataset faktor eksternal tidak memuat curah hujan, cuaca, luas panen, "
            "produksi, stok, maupun distribusi cabai. Karena itu, faktor yang ditampilkan "
            "hanya proksi biaya input dan transportasi, bukan penjelasan langsung atas harga cabai."
        )
    elif commodity == "Beras Medium":
        limitation = (
            "Catatan: dataset tidak memuat stok, produksi, panen, maupun harga gabah/beras "
            "global. Gandum hanya indikator pasar serealia yang lebih luas, bukan pengganti "
            "langsung harga beras."
        )
    else:
        limitation = (
            "Catatan: harga sawit global merupakan indikator yang paling dekat dengan bahan "
            "baku minyak goreng dalam dataset ini. Harga eceran lokal tetap dapat dipengaruhi "
            "kebijakan, pasokan domestik, distribusi, dan margin perdagangan yang tidak tercakup."
        )

    return pd.DataFrame(factor_rows), summary, limitation


def render_html_block(content: str):
    """Render trusted UI HTML directly; do not pass HTML fragments through Markdown."""
    cleaned_html = "\n".join(
        line.lstrip()
        for line in content.splitlines()
    ).strip()

    # st.html renders HTML as HTML. st.markdown(..., unsafe_allow_html=True)
    # can still send some indented fragments through Markdown code-block parsing.
    if not hasattr(st, "html"):
        raise RuntimeError(
            "Versi Streamlit terlalu lama. Gunakan streamlit>=1.33.0 "
            "agar komponen HTML dapat dirender dengan benar."
        )

    st.html(cleaned_html)


def render_header(title: str, subtitle: str):
    render_page_heading(title, subtitle)


def render_page_heading(
    title: str,
    subtitle: str,
    eyebrow: str = "PANGANPULSE ANALYTICS",
):
    safe_title = html_lib.escape(title)
    safe_subtitle = html_lib.escape(subtitle)
    safe_eyebrow = html_lib.escape(eyebrow)

    render_html_block(f"""
        <div class="page-heading">
            <span class="page-eyebrow">✦ {safe_eyebrow}</span>
            <h1>{safe_title}</h1>
            <p>{safe_subtitle}</p>
        </div>
    """)


def render_html_table(frame: pd.DataFrame, max_rows: int | None = None):
    """Render a compact HTML table with row-hover feedback."""
    view = frame.head(max_rows).copy() if max_rows else frame.copy()
    headers = "".join(
        f"<th>{html_lib.escape(str(column))}</th>" for column in view.columns
    )
    body_rows = []
    for _, row_data in view.iterrows():
        cells = []
        for value in row_data.tolist():
            text = "—" if pd.isna(value) else str(value)
            safe_value = html_lib.escape(text)
            cell_class = ""
            stripped = text.strip()
            if stripped.startswith("+"):
                cell_class = "pp-positive"
            elif stripped.startswith("-"):
                cell_class = "pp-negative"
            elif stripped.lower() in {"naik", "tinggi"}:
                cell_class = "pp-negative"
            elif stripped.lower() in {"turun", "rendah"}:
                cell_class = "pp-positive"
            elif stripped.lower() in {"stabil", "sedang", "relatif stabil"}:
                cell_class = "pp-neutral"
            cells.append(f'<td class="{cell_class}">{safe_value}</td>')
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    markup = (
        '<div class="pp-table-wrap"><table class="pp-table">'
        f"<thead><tr>{headers}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table></div>"
    )
    st.html(markup)


def style_plotly(fig, height: int | None = None):
    fig.update_layout(
        font=dict(family="Poppins, Plus Jakarta Sans, sans-serif", color="#3B2F2F", size=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(
            bgcolor="#3B2F2F",
            bordercolor="#F6C445",
            font=dict(color="#FFF3D6", family="Poppins, sans-serif", size=12),
        ),
        legend=dict(
            bgcolor="rgba(255,252,244,.72)",
            bordercolor="rgba(59,47,47,.08)",
            borderwidth=1,
            font=dict(color="#3B2F2F", size=11),
        ),
        transition=dict(duration=550, easing="cubic-in-out"),
        colorway=["#F05A28", "#8BC34A", "#F6C445", "#3B2F2F"],
    )
    if height:
        fig.update_layout(height=height)
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor="rgba(59,47,47,.14)",
        tickfont=dict(color="#78685D", size=10),
        title_font=dict(color="#3B2F2F"),
    )
    fig.update_yaxes(
        gridcolor="rgba(59,47,47,.09)",
        zeroline=False,
        tickfont=dict(color="#78685D", size=10),
        title_font=dict(color="#3B2F2F"),
    )
    return fig


def add_line_replay(fig, x_values, y_values, trace_index: int = 0):
    """Add a user-triggered line-draw replay to a Plotly chart."""
    x_values = list(x_values)
    y_values = list(y_values)
    if len(x_values) < 3:
        return fig

    frame_count = min(24, len(x_values) - 1)
    steps = np.unique(np.linspace(2, len(x_values), frame_count, dtype=int))
    frames = []
    for step in steps:
        frames.append(
            go.Frame(
                name=f"line-{step}",
                data=[go.Scatter(x=x_values[:step], y=y_values[:step])],
                traces=[trace_index],
            )
        )
    fig.frames = frames
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                showactive=False,
                x=1,
                xanchor="right",
                y=1.18,
                yanchor="top",
                bgcolor="#FFF3D6",
                bordercolor="rgba(240,90,40,.24)",
                borderwidth=1,
                font=dict(color="#3B2F2F", size=10),
                buttons=[
                    dict(
                        label="▶ Putar grafik",
                        method="animate",
                        args=[
                            None,
                            {
                                "fromcurrent": False,
                                "frame": {"duration": 42, "redraw": True},
                                "transition": {"duration": 35},
                            },
                        ],
                    ),
                    dict(
                        label="Ⅱ Jeda",
                        method="animate",
                        args=[
                            [None],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": False},
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                ],
            )
        ]
    )
    return fig


def add_grouped_bar_replay(fig, categories, first_values, second_values):
    """Add a replay control that grows grouped bars across categories."""
    categories = list(categories)
    first_values = list(first_values)
    second_values = list(second_values)
    if len(categories) < 3:
        return fig

    frame_count = min(24, len(categories) - 1)
    steps = np.unique(np.linspace(2, len(categories), frame_count, dtype=int))
    frames = []
    for step in steps:
        frames.append(
            go.Frame(
                name=f"bars-{step}",
                data=[
                    go.Bar(x=categories[:step], y=first_values[:step]),
                    go.Bar(x=categories[:step], y=second_values[:step]),
                ],
                traces=[0, 1],
            )
        )
    fig.frames = frames
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                showactive=False,
                x=1,
                xanchor="right",
                y=1.18,
                yanchor="top",
                bgcolor="#FFF3D6",
                bordercolor="rgba(240,90,40,.24)",
                borderwidth=1,
                font=dict(color="#3B2F2F", size=10),
                buttons=[
                    dict(
                        label="▶ Putar grafik",
                        method="animate",
                        args=[
                            None,
                            {
                                "fromcurrent": False,
                                "frame": {"duration": 55, "redraw": True},
                                "transition": {"duration": 45},
                            },
                        ],
                    ),
                    dict(
                        label="Ⅱ Jeda",
                        method="animate",
                        args=[
                            [None],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": False},
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                ],
            )
        ]
    )
    return fig


# ---------- Load data and models ----------
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
    st.error("Aplikasi belum dapat memuat model atau data artefak.")
    st.exception(exc)
    st.info(
        "Pastikan folder artifacts tersedia dan install requirements.txt "
        "dengan versi library yang sesuai."
    )
    st.stop()


# ---------- Sidebar ----------
with st.sidebar:
    render_html_block("""
        <div class="brand-wrap">
            <div class="brand-mark">🌾</div>
            <div class="brand">Pangan<span class="pulse">Pulse</span></div>
            <div class="brand-sub">PEMANTAUAN &amp; PREDIKSI<br>HARGA PANGAN NASIONAL</div>
        </div>
    """)
    st.markdown("**MENU UTAMA**")
    nav_labels = {
        "🏡  Dashboard": "Dashboard",
        "📈  Prediksi Harga H+30": "Prediksi H+30",
        "🧭  Perbandingan Provinsi": "Perbandingan Provinsi",
        "🧠  Tentang Model": "Tentang Model",
    }
    selected_nav = st.radio(
        "NAVIGASI",
        list(nav_labels.keys()),
        label_visibility="collapsed",
        key="pp_navigation",
    )
    page = nav_labels[selected_nav]

    st.divider()
    st.markdown("**RUANG LINGKUP**")
    render_html_block(f"""
        <div class="sidebar-note">
            <div>🌱 <strong>3 komoditas</strong></div>
            <div>🗺️ <strong>{int(eligible['Province'].nunique())} provinsi</strong></div>
            <div>📊 <strong>{len(predictions)} seri data</strong></div>
        </div>
    """)
    st.markdown("**PERIODE DATA**")
    st.caption(format_date(metadata["latest_data_date"]))
    st.caption("Academic project · Machine Learning")


# ---------- Indonesia province map helpers ----------
# GeoJSON ini memakai koordinat WGS84 (longitude, latitude) dan
# geometri provinsi yang disederhanakan agar peta ringan dirender.
INDONESIA_PROVINCE_GEOJSON_URLS = (
    "https://raw.githubusercontent.com/stoikal/indonesia-geodata/"
    "main/json/indonesiaLow.json",
    "https://cdn.jsdelivr.net/gh/stoikal/indonesia-geodata@main/"
    "json/indonesiaLow.json",
)

STATUS_COLORS = {
    "Turun": "#F6C445",       # kuning
    "Naik": "#E74C3C",        # merah
    "Stabil": "#8BC34A",      # hijau
    "Tidak tersedia": "#D8D2C8",
}

STATUS_CODES = {
    "Turun": 0,
    "Naik": 1,
    "Stabil": 2,
    "Tidak tersedia": 3,
}

# Nama pada GeoJSON diselaraskan dengan nama provinsi di dataset penelitian.
# Data penelitian memakai pembagian 34 provinsi, sedangkan GeoJSON ini
# memakai pembagian 38 provinsi. Provinsi hasil pemekaran Papua mengikuti
# status provinsi induk pada dataset 34 provinsi.
GEOJSON_TO_DATA_PROVINCE = {
    "Jakarta Raya": "DKI Jakarta",
    "Yogyakarta": "DI Yogyakarta",
    "Papua Pegunungan": "Papua",
    "Papua Selatan": "Papua",
    "Papua Tengah": "Papua",
    "Papua Barat Daya": "Papua Barat",
}

GEOJSON_DISPLAY_NAMES = {
    "Jakarta Raya": "DKI Jakarta",
    "Yogyakarta": "DI Yogyakarta",
}


@st.cache_data(ttl=86400, show_spinner="Memuat batas wilayah provinsi Indonesia...")
def load_indonesia_province_geojson():
    """Load simplified Indonesia province GeoJSON in WGS84 coordinates."""
    local_candidates = [
        BASE_DIR / "indonesiaLow.json",
        BASE_DIR / "assets" / "indonesiaLow.json",
    ]

    geojson = None
    for local_path in local_candidates:
        if local_path.exists():
            with local_path.open("r", encoding="utf-8") as file:
                geojson = json.load(file)
            break

    if geojson is None:
        download_errors = []
        for geojson_url in INDONESIA_PROVINCE_GEOJSON_URLS:
            try:
                request = Request(
                    geojson_url,
                    headers={"User-Agent": "PanganPulse/1.0"},
                )
                with urlopen(request, timeout=30) as response:
                    geojson = json.loads(response.read().decode("utf-8"))
                break
            except Exception as exc:
                download_errors.append(f"{geojson_url}: {exc}")

        if geojson is None:
            raise RuntimeError(
                "Tidak dapat mengunduh GeoJSON dari sumber utama maupun CDN. "
                + " | ".join(download_errors)
            )

    if geojson.get("type") != "FeatureCollection":
        raise ValueError("Format GeoJSON batas provinsi tidak valid.")

    features = geojson.get("features", [])
    if len(features) < 30:
        raise ValueError(
            f"GeoJSON hanya memuat {len(features)} fitur; "
            "batas provinsi Indonesia tidak lengkap."
        )

    # Pastikan properti nama dan koordinat benar-benar tersedia.
    for feature in features:
        properties = feature.get("properties") or {}
        if not properties.get("name"):
            raise ValueError(
                "Properti 'name' tidak ditemukan pada salah satu fitur GeoJSON."
            )
        if not feature.get("geometry"):
            raise ValueError("Ada fitur GeoJSON tanpa geometri.")

    # Cek koordinat WGS84 secara rekursif. Ini mencegah file berproyeksi
    # meter (misalnya EPSG:3857) dibaca seolah-olah longitude/latitude.
    def walk_coordinates(value):
        if (
            isinstance(value, (list, tuple))
            and len(value) >= 2
            and isinstance(value[0], (int, float))
            and isinstance(value[1], (int, float))
        ):
            yield float(value[0]), float(value[1])
        elif isinstance(value, (list, tuple)):
            for child in value:
                yield from walk_coordinates(child)

    sample = []
    for feature in features:
        coords = feature["geometry"].get("coordinates", [])
        for point in walk_coordinates(coords):
            sample.append(point)
            if len(sample) >= 500:
                break
        if len(sample) >= 500:
            break

    if not sample:
        raise ValueError("Koordinat GeoJSON tidak dapat dibaca.")

    lon_values = [point[0] for point in sample]
    lat_values = [point[1] for point in sample]

    if (
        min(lon_values) < 90 or max(lon_values) > 150
        or min(lat_values) < -15 or max(lat_values) > 12
    ):
        raise ValueError(
            "Koordinat GeoJSON bukan rentang WGS84 Indonesia. "
            "Periksa sumber atau sistem koordinat GeoJSON."
        )

    return geojson


def normalize_prediction_status(value) -> str:
    """Normalize model labels into the three requested map categories."""
    text = str(value).strip().lower()

    if "turun" in text:
        return "Turun"
    if "naik" in text:
        return "Naik"
    if "stabil" in text or "relatif" in text:
        return "Stabil"

    return "Tidak tersedia"


def build_province_status_map(geojson: dict, commodity_data: pd.DataFrame):
    """Build a province choropleth with hover limited to name and status."""
    status_by_data_province = {}

    for _, row in commodity_data.iterrows():
        province_name = str(row["Province"]).strip()
        status_by_data_province[province_name] = normalize_prediction_status(
            row["Pred_Kelas_H30"]
        )

    map_locations = []
    map_status_codes = []
    map_customdata = []

    for feature in geojson["features"]:
        geo_name = str(feature.get("properties", {}).get("name", "")).strip()
        if not geo_name:
            continue

        data_province = GEOJSON_TO_DATA_PROVINCE.get(geo_name, geo_name)
        status = status_by_data_province.get(
            data_province,
            "Tidak tersedia",
        )
        display_name = GEOJSON_DISPLAY_NAMES.get(geo_name, geo_name)

        map_locations.append(geo_name)
        map_status_codes.append(STATUS_CODES[status])
        map_customdata.append([display_name, status, data_province])

    if len(map_locations) < 30:
        raise ValueError(
            f"Hanya {len(map_locations)} provinsi yang berhasil dipetakan."
        )

    # Batas warna dibuat diskrit agar tiap kelas selalu memakai satu warna.
    colorscale = [
        [0.0000, STATUS_COLORS["Turun"]],
        [0.1666, STATUS_COLORS["Turun"]],
        [0.1667, STATUS_COLORS["Naik"]],
        [0.5000, STATUS_COLORS["Naik"]],
        [0.5001, STATUS_COLORS["Stabil"]],
        [0.8333, STATUS_COLORS["Stabil"]],
        [0.8334, STATUS_COLORS["Tidak tersedia"]],
        [1.0000, STATUS_COLORS["Tidak tersedia"]],
    ]

    fig = go.Figure(
        go.Choropleth(
            geojson=geojson,
            featureidkey="properties.name",
            locations=map_locations,
            z=map_status_codes,
            zmin=0,
            zmax=3,
            colorscale=colorscale,
            showscale=False,
            marker=dict(
                line=dict(
                    color="#FFF9EC",
                    width=0.8,
                )
            ),
            customdata=np.asarray(map_customdata, dtype=object),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Status prediksi: <b>%{customdata[1]}</b>"
                "<extra></extra>"
            ),
            name="Status prediksi",
        )
    )

    fig.update_geos(
        fitbounds="locations",
        visible=False,
        projection_type="mercator",
        bgcolor="rgba(0,0,0,0)",
        showframe=False,
        showcountries=False,
        showcoastlines=False,
        showland=False,
        domain=dict(x=[0.02, 0.98], y=[0.02, 0.98]),
    )

    fig.update_layout(
        height=590,
        autosize=True,
        margin=dict(l=5, r=5, t=5, b=5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(
            bgcolor="#3B2F2F",
            bordercolor="#F6C445",
            font=dict(
                color="#FFF3D6",
                family="Poppins, sans-serif",
                size=13,
            ),
        ),
        dragmode=False,
        uirevision="panganpulse-indonesia-province-map",
    )

    return fig, map_customdata


# ---------- Dashboard ----------
if page == "Dashboard":
    render_html_block("""
        <div class="hero">
            <div class="hero-kicker">✦ Indonesian food price intelligence</div>
            <div class="hero-title">Pantau Harga. Pahami Prediksi. 🌾</div>
            <div class="hero-copy">
                Kenali pergerakan harga pangan melalui data historis, prediksi H+30,
                dan konteks faktor eksternal untuk tiga komoditas penelitian.
            </div>
            <div class="hero-pill">● Data penelitian · 3 komoditas · Prediksi H+30</div>
        </div>
    """)
    show_historical_warning(metadata)

    provinces_count = int(eligible["Province"].nunique())
    commodities_count = int(eligible["Commodity"].nunique())
    last_date = pd.Timestamp(metadata["latest_data_date"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Komoditas", commodities_count)
    c2.metric("Provinsi", provinces_count)
    c3.metric("Seri tersedia", len(eligible))
    c4.metric("Horizon prediksi", f"H+{metadata['horizon_days']}")

    render_html_block("""
        <div class="section-kicker">Market overview</div>
        <div class="section-title">Gambaran harga historis</div>
        <div class="section-subtitle">Pilih komoditas untuk melihat pergerakan median harga antarprovinsi.</div>
    """)
    commodity_options = sorted(predictions["Commodity"].unique().tolist())
    selected_commodity = st.selectbox(
        "Pilih komoditas untuk melihat tren median antarprovinsi",
        commodity_options,
        key="dashboard_commodity",
    )

    selected_history = history[
        history["Commodity"] == selected_commodity
    ]
    median_history = (
        selected_history.groupby("Date", as_index=False)["Price"]
        .median()
        .rename(columns={"Price": "Median harga antarprovinsi"})
    )

    fig = px.line(
        median_history,
        x="Date",
        y="Median harga antarprovinsi",
        markers=False,
        title=f"Median harga historis — {selected_commodity}",
    )
    fig.update_traces(line_color="#F05A28", line_width=3.2, line_shape="spline", fill="tozeroy", fillcolor="rgba(240,90,40,.10)", marker=dict(size=5, color="#F6C445", line=dict(color="#F05A28", width=1)))
    fig.update_layout(
        height=390,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_title=None,
        yaxis_title="Nilai harga (satuan dataset)",
    )
    style_plotly(fig, 390)
    add_line_replay(
        fig,
        median_history["Date"],
        median_history["Median harga antarprovinsi"],
        trace_index=0,
    )
    st.plotly_chart(fig, use_container_width=True)

    render_html_block("""
        <div class="section-kicker">Forecast snapshot</div>
        <div class="section-title">Ringkasan prediksi H+30</div>
    """)
    summary = (
        predictions.groupby("Commodity")
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
        .map(lambda x: f"{x:+.2f}%" if pd.notna(x) else "—")
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
    render_html_table(display_summary)


# ---------- Prediction page ----------
elif page == "Prediksi H+30":
    render_header(
        "Prediksi Harga H+30",
        "Pilih komoditas dan provinsi untuk melihat estimasi harga 30 hari setelah tanggal data terakhir.",
    )
    show_historical_warning(metadata)

    commodity_options = sorted(predictions["Commodity"].unique().tolist())
    commodity = st.selectbox(
        "Komoditas",
        commodity_options,
        key="prediction_commodity",
    )
    province_options = sorted(
        predictions.loc[
            predictions["Commodity"] == commodity, "Province"
        ].unique().tolist()
    )
    province = st.selectbox(
        "Provinsi",
        province_options,
        key="prediction_province",
    )

    row = prediction_row(predictions, commodity, province)
    if row is None:
        st.error("Seri komoditas-provinsi tidak tersedia pada data penelitian.")
        st.stop()

    render_html_block("""
        <div class="section-kicker">Prediction result</div>
        <div class="section-title">Hasil prediksi</div>
    """)
    m1, m2, m3 = st.columns(3)
    m1.metric(
        "Harga terakhir",
        format_number(row["current_price"]),
        help="Harga pada tanggal pengamatan terakhir.",
    )
    m2.metric(
        "Prediksi harga H+30",
        format_number(row["Pred_Harga_H30"]),
        delta=f"{row['Perubahan_Persen']:+.2f}% dari harga terakhir",
        help="Estimasi harga tepat 30 hari setelah tanggal pengamatan.",
    )
    m3.metric(
        "Klasifikasi perubahan",
        row["Pred_Kelas_H30"],
        help="Kategori berdasarkan perubahan harga H+30: Turun, Stabil, atau Naik.",
    )

    st.caption(
        f"Tanggal pengamatan: **{format_date(row['Date'])}**  ·  "
        f"Tanggal target: **{format_date(row['TargetDate'])}**"
    )

    st.markdown("### Grafik harga historis dan titik prediksi")
    st.plotly_chart(
        make_history_chart(history, row),
        use_container_width=True,
    )
    st.caption(
        "Garis putus-putus hanya menghubungkan harga terakhir dengan titik target H+30. "
        "Garis tersebut bukan prediksi harga harian di antara kedua tanggal."
    )

    probability_cols = [
        col for col in predictions.columns if col.startswith("Prob_")
    ]
    if probability_cols:
        st.markdown("### Probabilitas klasifikasi")
        prob_values = []
        for col in probability_cols:
            label = col.replace("Prob_", "")
            prob_values.append(
                {"Kelas": label, "Probabilitas": float(row[col])}
            )
        prob_df = pd.DataFrame(prob_values)
        fig_prob = px.bar(
            prob_df,
            x="Kelas",
            y="Probabilitas",
            text=prob_df["Probabilitas"].map(lambda x: f"{x:.1%}"),
            color="Kelas",
            color_discrete_map={
                "Turun": "#F05A28",
                "Stabil": "#F6C445",
                "Naik": "#8BC34A",
            },
        )
        fig_prob.update_traces(
            marker_line_color="#FFF3D6",
            marker_line_width=1.5,
            marker_cornerradius=9,
            textposition="outside",
        )
        fig_prob.update_layout(
            showlegend=False,
            height=300,
            yaxis=dict(tickformat=".0%", range=[0, 1]),
            margin=dict(l=10, r=10, t=25, b=10),
        )
        style_plotly(fig_prob, 300)
        st.plotly_chart(fig_prob, use_container_width=True)

    render_html_block("""
        <div class="section-kicker">Market context</div>
        <div class="section-title">Konteks faktor eksternal</div>
    """)
    factor_table, factor_summary, factor_limitation = build_external_factor_explanation(
        commodity,
        str(row["Pred_Kelas_H30"]),
        row,
    )
    render_html_block(
        f'<div class="pp-info"><strong>Ringkasan interpretasi:</strong> '
        f'{html_lib.escape(factor_summary)}</div>'
    )
    if not factor_table.empty:
        render_html_table(factor_table)
    st.caption(
        "Perubahan faktor dihitung dari return 30 hari terakhir sebelum tanggal pengamatan. "
        "Faktor eksternal ini ditampilkan sebagai konteks interpretasi dan tidak menjadi "
        "input classifier final."
    )
    st.caption(factor_limitation)

    with st.expander("Lihat detail data seri"):
        series_history = history[
            (history["Commodity"] == commodity)
            & (history["Province"] == province)
        ].sort_values("Date", ascending=False)
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


# ---------- Province comparison ----------
elif page == "Perbandingan Provinsi":
    render_header(
        "Peta Status Prediksi Provinsi",
        "Lihat arah prediksi harga H+30 untuk setiap provinsi berdasarkan komoditas yang dipilih.",
    )
    show_historical_warning(metadata)

    commodity_options = sorted(predictions["Commodity"].unique().tolist())
    commodity = st.selectbox(
        "Pilih komoditas",
        commodity_options,
        key="comparison_commodity",
    )

    comp = predictions[
        predictions["Commodity"] == commodity
    ].copy()

    render_html_block("""
        <div class="section-kicker">Indonesia map overview</div>
        <div class="section-title">Status prediksi per provinsi</div>
        <div class="section-subtitle">
            Arahkan kursor ke salah satu provinsi untuk melihat nama wilayah dan status prediksinya.
        </div>
    """)

    legend_items = [
        ("Turun", STATUS_COLORS["Turun"]),
        ("Naik", STATUS_COLORS["Naik"]),
        ("Stabil", STATUS_COLORS["Stabil"]),
    ]
    if comp["Province"].nunique() < 34:
        legend_items.append(("Tidak tersedia", STATUS_COLORS["Tidak tersedia"]))

    legend_html = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:8px;'
        f'padding:7px 12px;border:1px solid rgba(59,47,47,.10);'
        f'border-radius:999px;background:#FFFCF4;color:#3B2F2F;'
        f'font-size:13px;font-weight:600;">'
        f'<span style="width:11px;height:11px;border-radius:4px;'
        f'background:{color};display:inline-block;"></span>{label}</span>'
        for label, color in legend_items
    )

    render_html_block(
        '<div style="display:flex;flex-wrap:wrap;gap:9px;margin:0 0 14px;">'
        + legend_html
        + '</div>'
    )

    try:
        indonesia_geojson = load_indonesia_province_geojson()
        map_fig, map_customdata = build_province_status_map(
            indonesia_geojson,
            comp,
        )

        st.plotly_chart(
            map_fig,
            use_container_width=True,
            config={
                "displayModeBar": False,
                "scrollZoom": False,
                "responsive": True,
            },
        )

        unavailable_count = sum(
            1 for item in map_customdata if item[1] == "Tidak tersedia"
        )
        if unavailable_count:
            st.caption(
                f"{unavailable_count} provinsi tidak memiliki hasil prediksi "
                "untuk komoditas ini dan ditampilkan dengan warna abu-abu."
            )

    except Exception as exc:
        st.error(
            "Peta belum dapat dimuat. Pastikan aplikasi memiliki akses internet "
            "untuk mengambil batas wilayah GeoJSON Indonesia."
        )
        st.caption(f"Detail teknis: {exc}")

    st.caption(
        "Warna menunjukkan hasil klasifikasi prediksi H+30: "
        "kuning = turun, merah = naik, dan hijau = stabil. "
        "Batas peta menggunakan pembagian provinsi terbaru (38 wilayah). "
        "Karena dataset penelitian menggunakan 34 provinsi, wilayah hasil "
        "pemekaran Papua mengikuti status provinsi induknya. "
        "Sumber geometri: stoikal/indonesia-geodata (indonesiaLow)."
    )


# ---------- About model ----------
elif page == "Tentang Model":
    render_header(
        "Tentang Model dan Penelitian",
        "Informasi algoritma, metrik evaluasi, cakupan data, dan keterbatasan model yang digunakan.",
    )
    show_historical_warning(metadata)

    reg = metadata["regression"]
    cls = metadata["classification"]
    scope = metadata["scope"]

    st.markdown("### Model regresi")
    st.subheader(reg["candidate"])
    st.write(
        f"Model dipilih berdasarkan **{reg['selection_rule']}**."
    )
    r1, r2, r3 = st.columns(3)
    r1.metric("Mean MAE", format_number(reg["validation_metrics"]["Mean_MAE"], 2))
    r2.metric("Mean RMSE", format_number(reg["validation_metrics"]["Mean_RMSE"], 2))
    r3.metric("Mean MAPE", f"{reg['validation_metrics']['Mean_MAPE']:.2f}%")

    st.markdown("### Model klasifikasi")
    st.subheader(cls["candidate"])
    st.write(
        f"Model dipilih berdasarkan **{cls['selection_rule']}**."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{cls['validation_metrics']['Mean_Accuracy']:.4f}")
    c2.metric("Macro F1", f"{cls['validation_metrics']['Mean_Macro_F1']:.4f}")
    c3.metric(
        "Balanced Accuracy",
        f"{cls['validation_metrics']['Mean_Balanced_Accuracy']:.4f}",
    )
    st.caption(
        "Classifier final menggunakan fitur historis Skenario A. Faktor eksternal "
        "ditampilkan pada halaman prediksi sebagai konteks interpretasi, bukan "
        "sebagai penyebab yang telah dibuktikan atau input classifier final."
    )

    st.markdown("### Cakupan penelitian")
    a, b, c = st.columns(3)
    a.metric("Komoditas", len(scope["commodities"]))
    b.metric("Provinsi", int(eligible["Province"].nunique()))
    c.metric("Seri eligible", int(scope["n_eligible_series"]))

    st.write("**Komoditas yang dianalisis:**")
    st.write(", ".join(scope["commodities"]))

    with st.expander("Lihat fitur yang digunakan"):
        st.write("**Fitur regresi:**")
        st.code(", ".join(reg["features"]))
        st.write("**Fitur klasifikasi:**")
        st.code(", ".join(cls["features"]))

    with st.expander("Lihat konfigurasi model"):
        st.write("**Konfigurasi regresi:**")
        st.json(reg["params"])
        st.write("**Konfigurasi klasifikasi:**")
        st.json(cls["params"])

    st.markdown("### Metodologi evaluasi")
    validation = metadata["validation_design"]
    st.write(
        f"Evaluasi menggunakan **{validation['folds'].__len__()} fold "
        "expanding rolling-origin**. Periode validasi berlangsung Januari–Mei 2024."
    )
    st.write(
        f"Jumlah baris training final: **{metadata['training_data']['rows']:,}**."
    )

    st.markdown("### Keterbatasan")
    for limitation in metadata.get("limitations", []):
        st.markdown(f"- {limitation}")

    render_html_block("""
        <div class="pp-info">
            <strong>Catatan evaluasi:</strong>
            Angka performa di atas merupakan hasil validasi rolling-origin yang
            juga digunakan selama proses pemilihan model. Angka tersebut bukan
            hasil pengujian pada holdout independen.
        </div>
    """)

st.divider()
render_html_block("""
    <div class="footer-note">
        PanganPulse ✦ Academic Machine Learning Project ·
        Prediksi historis H+30 · Sumber dan satuan mengikuti dataset penelitian.
    </div>
""")
