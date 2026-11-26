# PanganPulse — Food Price Forecasting Dashboard

Aplikasi Streamlit untuk menampilkan hasil model penelitian:
- Regresi: LightGBM A
- Klasifikasi: Logistic Regression
- Horizon: H+30
- Cakupan: 3 komoditas, 34 provinsi, 101 seri eligible

## Catatan penting tentang data
Dataset terakhir berakhir pada 30 September 2024. Karena itu, aplikasi ini
menampilkan simulasi prediksi historis H+30 dengan target 30 Oktober 2024,
bukan prakiraan harga terkini tahun 2026.

Satuan harga pada dataset sumber belum diverifikasi. Aplikasi sengaja tidak
menambahkan simbol Rp agar tidak mengasumsikan satuan yang belum dipastikan.

## Struktur proyek
```text
PanganPulse/
├── app.py
├── requirements.txt
├── README.md
└── artifacts/
    ├── regression_model.joblib
    ├── classification_model.joblib
    ├── model_metadata.json
    ├── eligible_series.csv
    ├── latest_history.csv
    ├── latest_features.csv
    ├── external_features_daily.csv
    └── external_medians.json
```

## Menjalankan secara lokal
Gunakan Python 3.13. Buat virtual environment, lalu:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Penting: model diserialisasi dengan scikit-learn 1.6.1. Jangan mengganti
versi scikit-learn secara sembarang karena model pipeline dapat gagal dimuat
atau menimbulkan ketidakcocokan saat deserialisasi.

## Deployment ke Streamlit Community Cloud
1. Upload seluruh isi folder proyek ini ke repository GitHub.
2. Pastikan folder `artifacts/` ikut terunggah.
3. Buka Streamlit Community Cloud.
4. Pilih repository, branch, dan file utama `app.py`.
5. Deploy.

Jangan mengunggah dataset pribadi atau kredensial ke repository publik.
Artefak ini berasal dari dataset penelitian dan harus diperlakukan sesuai
ketentuan penggunaan dataset sumber.

## Fitur aplikasi
- Dashboard ringkasan cakupan penelitian dan tren historis.
- Prediksi harga H+30 untuk komoditas-provinsi yang eligible.
- Klasifikasi perubahan harga: Turun, Stabil, atau Naik.
- Probabilitas kelas jika tersedia dari model.
- Perbandingan hasil prediksi antarprovinsi.
- Unduh tabel perbandingan sebagai CSV.
- Informasi model, metrik, dan keterbatasan penelitian.

Aplikasi tidak menyediakan rekomendasi AI dan tidak membuat prediksi H+7.
