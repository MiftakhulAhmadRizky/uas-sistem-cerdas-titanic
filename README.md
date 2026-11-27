# Sistem Prediksi Keselamatan Penumpang Titanic Berbasis Web (Streamlit)

## Masalah yang diselesaikan
Memprediksi peluang keselamatan penumpang kapal Titanic berdasarkan 4 atribut utama penumpang (kelas tiket, jenis kelamin, usia, dan tarif tiket) menggunakan model klasifikasi *machine learning* untuk mengidentifikasi pola prioritas evakuasi.

## Dataset
- **Sumber:** Titanic - Machine Learning from Disaster (Kaggle: https://www.kaggle.com/c/titanic)
- **Lisensi:** Public / Open Competition Data
- **Jumlah Data:** 891 baris data penumpang (split 80% data latih / 712 baris dan 20% data uji / 179 baris)

## Metode
- **Preprocessing:** Encoding kategorikal pada kolom `Sex` (`female=0`, `male=1`) serta imputasi nilai kosong menggunakan *median* pada kolom `Age` dan `Fare`.
- **Algoritma:** Random Forest Classifier (`n_estimators=100`, `random_state=42`) dengan akurasi pengujian sebesar **80.45%**.
- **Antarmuka Web:** Python Streamlit (`app.py`).

## Struktur Folder
- `dataset/titanic.csv`: Dataset historis penumpang Titanic
- `model/model_rf.pkl`: Model Random Forest terlatih
- `model/evaluasi.json`: Metrik evaluasi model (akurasi, presisi, recall)
- `train_model.py`: Skrip pelatihan dan evaluasi model
- `app.py`: Aplikasi utama antarmuka web Streamlit
- `requirements.txt`: Daftar pustaka Python yang dibutuhkan

## Cara menjalankan
1. Aktifkan virtual environment dan instal pustaka:
   ```bash
   pip install -r requirements.txt