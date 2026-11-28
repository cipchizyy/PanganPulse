from __future__ import annotations

import json
import warnings
import html as html_lib
from pathlib import Path

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


# ============================================================
# UI/UX STYLING
# Color Palette:
# Yellow : #F6C445
# Orange : #F05A28
# Green  : #8BC34A
# Cream  : #FFF3D6
# Brown  : #3B2F2F
# ============================================================

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


    /* SIDEBAR */

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

    .brand .pulse {
        color: var(--pp-yellow) !important;
    }

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


    /* HEADER AND HERO */

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


    /* METRICS AND CARDS */

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


    /* INPUTS, TABS, BUTTONS */

    [data-testid="stSelectbox"] label,
    [data-testid="stTextInput"] label,
    [data-testid="stSlider"] label {
        color: var(--pp-brown) !important;
        font-weight: 600;
        font-size: .83rem;
    }

    [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        background: #FFFCF4;
        border: 1px solid rgba(59,47,47,.16);
        border-radius: 13px;
        min-height: 44px;
        transition: border-color .2s ease, box-shadow .2s ease;
    }

    [data-testid="stSelectbox"] div[data-baseweb="select"] > div:focus-within {
        border-color: var(--pp-orange);
        box-shadow: 0 0 0 3px rgba(240,90,40,.12);
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


    /* ALERTS AND EXPANDERS */

    [data-testid="stAlert"] {
        border-radius: 15px;
        border: 1px solid rgba(246,196,69,.35);
        background: rgba(255,248,225,.9);
        color: var(--pp-brown);
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


    /* INTERACTIVE HTML TABLE */

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

    .pp-table tbody tr:last-child td {
        border-bottom: 0;
    }

    .pp-positive {
        color: #4E8425 !important;
        font-weight: 700;
    }

    .pp-negative {
        color: #D94F24 !important;
        font-weight: 700;
    }

    .pp-neutral {
        color: #8A6B25 !important;
        font-weight: 700;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(59,47,47,.12);
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 7px 20px rgba(82,54,31,.05);
    }


    /* SMALL DETAILS */

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


    /* ANIMATIONS */

    @keyframes ppFadeUp {
        from {
            opacity: 0;
            transform: translateY(12px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes ppPulse {
        0%, 100% {
            box-shadow: 0 0 0 0 rgba(240,90,40,0);
        }
        50% {
            box-shadow: 0 0 0 7px rgba(240,90,40,.08);
        }
    }


    /* RESPONSIVE */

    @media (max-width: 900px) {
        [data-testid="stMainBlockContainer"] {
            padding: 1rem .9rem 2rem;
        }

        .hero {
            padding: 1.25rem;
            border-radius: 19px;
        }

        .hero:after {
            right: -1rem;
            font-size: 5rem;
        }

        div[data-testid="stMetric"] {
            min-height: 100px;
            padding: .8rem;
        }
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


# ============================================================
# GENERAL HELPERS
# ============================================================

def format_number(value: float | int | None, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "—"

    formatted = f"{float(value):,.{decimals}f}"

    return (
        formatted
        .replace(",", "_")
        .replace(".", ",")
        .replace("_", ".")
    )


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
