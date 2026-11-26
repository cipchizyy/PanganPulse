from __future__ import annotations

import json
import warnings
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

# ---------- Styling ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'Manrope', sans-serif;
        letter-spacing: -0.025em;
    }
    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2.5rem;
        max-width: 1440px;
    }
    [data-testid="stSidebar"] {
        background: #F3F7F3;
        border-right: 1px solid #E1EAE2;
    }
    .brand {
        font-family: 'Manrope', sans-serif;
        font-weight: 800;
        font-size: 1.55rem;
        color: #174B3A;
        margin-bottom: 0;
    }
    .brand-sub {
        color: #6B7D70;
        font-size: .78rem;
        margin-top: -.25rem;
    }
    .hero {
        background: linear-gradient(120deg, #174B3A 0%, #28664D 100%);
        color: white;
        padding: 1.55rem 1.7rem;
        border-radius: 18px;
        margin-bottom: 1.1rem;
    }
    .hero-kicker {
        text-transform: uppercase;
        letter-spacing: .11em;
        font-size: .72rem;
        font-weight: 700;
        color: #C8E0D0;
    }
    .hero-title {
        font-family: 'Manrope', sans-serif;
        font-weight: 800;
        font-size: clamp(1.7rem, 3vw, 2.45rem);
        line-height: 1.15;
        margin: .4rem 0;
    }
    .hero-copy {
        color: #E1EEE5;
        font-size: .95rem;
        max-width: 760px;
    }
    .section-label {
        color: #648071;
        font-size: .76rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .08em;
    }
    .small-muted {
        color: #738176;
        font-size: .82rem;
    }
    .stMetric {
        background: #F5F8F5;
        border: 1px solid #E4ECE5;
        border-radius: 14px;
        padding: 12px 14px;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid #E5ECE6;
        border-radius: 12px;
        overflow: hidden;
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
        ["Date", "Commodity", "Province", "current_price"]
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
    st.warning(
        f"**Catatan periode data:** dataset berakhir pada "
        f"{format_date(last_date)}. Prediksi H+30 pada aplikasi ini "
        f"merupakan simulasi historis dengan target {format_date(target_date)}, "
        "bukan prakiraan harga terkini tahun 2026. Satuan harga mengikuti "
        "dataset sumber dan belum diverifikasi."
    )


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
            line=dict(color="#287A55", width=2.5),
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
            line=dict(color="#D59A42", width=2, dash="dash"),
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[target_date],
            y=[predicted_price],
            mode="markers",
            name="Prediksi H+30",
            marker=dict(color="#D59A42", size=13, symbol="diamond"),
            hovertemplate="%{x|%d %b %Y}<br>Prediksi: %{y:,.1f}<extra></extra>",
        )
    )

    fig.update_layout(
        height=390,
        margin=dict(l=10, r=10, t=25, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis_title=None,
        yaxis_title="Nilai harga (satuan dataset)",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#E9EFEA", zeroline=False)
    return fig


def render_header(title: str, subtitle: str):
    st.markdown(f"## {title}")
    st.caption(subtitle)


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
    st.markdown(
        '<div class="brand">🌾 PanganPulse</div>'
        '<div class="brand-sub">Food Price Forecasting Dashboard</div>',
        unsafe_allow_html=True,
    )
    st.divider()
    page = st.radio(
        "NAVIGASI",
        [
            "Dashboard",
            "Prediksi H+30",
            "Perbandingan Provinsi",
            "Tentang Model",
        ],
        label_visibility="visible",
    )
    st.divider()
    st.caption("RUANG LINGKUP PENELITIAN")
    st.markdown("**3 komoditas**")
    st.markdown("**34 provinsi**")
    st.markdown(f"**{len(predictions)} seri eligible**")
    st.divider()
    st.caption("Periode data terakhir")
    st.markdown(
        f"**{format_date(metadata['latest_data_date'])}**"
    )
    st.caption("Academic project · Machine Learning")


# ---------- Dashboard ----------
if page == "Dashboard":
    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Indonesian food price analysis</div>
            <div class="hero-title">Pantau Harga. Pahami Prediksi.</div>
            <div class="hero-copy">
                Dashboard analisis dan prediksi harga bahan pokok menggunakan
                Machine Learning, dengan fokus pada tiga komoditas penelitian
                dan seri provinsi yang memenuhi kriteria seleksi.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    show_historical_warning(metadata)

    provinces_count = int(eligible["Province"].nunique())
    commodities_count = int(eligible["Commodity"].nunique())
    last_date = pd.Timestamp(metadata["latest_data_date"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Komoditas", commodities_count)
    c2.metric("Provinsi", provinces_count)
    c3.metric("Seri tersedia", len(eligible))
    c4.metric("Horizon prediksi", f"H+{metadata['horizon_days']}")

    st.markdown("### Gambaran harga historis")
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
    fig.update_traces(line_color="#287A55", line_width=2.5)
    fig.update_layout(
        height=370,
        margin=dict(l=10, r=10, t=55, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis_title=None,
        yaxis_title="Nilai harga (satuan dataset)",
    )
    fig.update_yaxes(gridcolor="#E9EFEA")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Ringkasan prediksi H+30")
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
    st.dataframe(
        display_summary,
        use_container_width=True,
        hide_index=True,
    )


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
                "Turun": "#D66B62",
                "Stabil": "#8A9A8D",
                "Naik": "#287A55",
            },
        )
        fig_prob.update_layout(
            showlegend=False,
            height=280,
            yaxis=dict(tickformat=".0%", range=[0, 1]),
            margin=dict(l=10, r=10, t=20, b=10),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        st.plotly_chart(fig_prob, use_container_width=True)

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
        "Perbandingan Antarprovinsi",
        "Bandingkan harga terakhir dan estimasi harga H+30 untuk satu komoditas pada seluruh provinsi yang tersedia.",
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
        comp = comp.sort_values("Pred_Harga_H30", ascending=False)
    elif sort_option == "Prediksi harga terendah":
        comp = comp.sort_values("Pred_Harga_H30", ascending=True)
    elif sort_option == "Persentase kenaikan tertinggi":
        comp = comp.sort_values("Perubahan_Persen", ascending=False)
    elif sort_option == "Persentase penurunan terbesar":
        comp = comp.sort_values("Perubahan_Persen", ascending=True)
    else:
        comp = comp.sort_values("Province")

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
    st.caption(f"{len(comp)} seri provinsi tersedia untuk komoditas ini.")

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
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title=None,
        yaxis_title="Nilai harga (satuan dataset)",
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_yaxes(gridcolor="#E9EFEA")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Tabel hasil")
    st.dataframe(
        table.round({"Harga terakhir": 1, "Prediksi H+30": 1, "Perubahan (%)": 2}),
        use_container_width=True,
        hide_index=True,
    )

    csv_data = comp.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇️ Unduh hasil perbandingan (CSV)",
        data=csv_data,
        file_name=f"panganpulse_{commodity.lower().replace(' ', '_')}_h30.csv",
        mime="text/csv",
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

    st.info(
        "Angka performa di atas merupakan hasil validasi rolling-origin yang "
        "juga digunakan selama proses pemilihan model. Angka tersebut bukan "
        "hasil pengujian pada holdout independen."
    )

st.divider()
st.caption(
    "PanganPulse · Academic Machine Learning Project · "
    "Prediksi historis H+30 · Sumber dan satuan mengikuti dataset penelitian."
)
