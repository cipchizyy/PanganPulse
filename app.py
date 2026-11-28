        comp["Province"],
        comp["current_price"],
        comp["Pred_Harga_H30"],
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Tabel hasil")
    display_table = table.round(
        {"Harga terakhir": 1, "Prediksi H+30": 1, "Perubahan (%)": 2}
    ).copy()
    display_table["Harga terakhir"] = display_table["Harga terakhir"].map(lambda v: format_number(v, 1))
    display_table["Prediksi H+30"] = display_table["Prediksi H+30"].map(lambda v: format_number(v, 1))
    display_table["Perubahan (%)"] = display_table["Perubahan (%)"].map(lambda v: f"{v:+.2f}%")
    render_html_table(display_table)

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
