# Sistem Prediksi Keselamatan Penumpang Titanic Berbasis Web (Streamlit)

## Masalah yang Diselesaikan
Memprediksi peluang keselamatan penumpang kapal Titanic berdasarkan 4 atribut utama penumpang (kelas tiket, jenis kelamin, usia, dan tarif tiket) menggunakan model klasifikasi *machine learning* untuk mengidentifikasi pola prioritas evakuasi.

## Dataset & Sumber
- **Sumber Dataset:** Titanic - Machine Learning from Disaster (Kaggle: [https://www.kaggle.com/c/titanic](https://www.kaggle.com/c/titanic))
- **Lisensi:** Public / Open Competition Data
- **Jumlah Data:** 891 baris data penumpang (dibagi menjadi **70% data latih / 623 baris** dan **30% data uji / 268 baris**).

## Metode
- **Preprocessing:** *Encoding* kategorikal pada kolom `Sex` (`female = 0`, `male = 1`) serta *imputasi* nilai kosong (`NaN`) menggunakan nilai tengah (*median*) pada kolom `Age` dan `Fare`.
- **Algoritma:** Random Forest Classifier (`n_estimators=100`, `random_state=42`).
- **Antarmuka Web:** Python Streamlit (`app.py`).

## Hasil Evaluasi
Evaluasi dilakukan menggunakan 30% data uji (268 baris) dengan metrik berikut (sesuaikan angka di bawah dengan isi file `model/evaluasi.json` setelah menjalankan ulang `train_model.py` dengan `test_size=0.3`):
- **Accuracy:** `79.85%` *(ganti sesuai nilai di `evaluasi.json`)*
- **Precision:** `80.0%` *(ganti sesuai nilai di `evaluasi.json`)*
- **Recall:** `68.47%` *(ganti sesuai nilai di `evaluasi.json`)*

## Struktur Folder
- `dataset/titanic.csv`: Dataset historis penumpang Titanic
- `model/model_rf.pkl`: File biner model Random Forest terlatih
- `model/evaluasi.json`: File penyimpanan metrik evaluasi model (akurasi, presisi, recall)
- `train_model.py`: Skrip pelatihan dan evaluasi model
- `app.py`: Aplikasi utama antarmuka web Streamlit
- `requirements.txt`: Daftar pustaka Python yang dibutuhkan

## Cara Menjalankan
1. Aktifkan *virtual environment* dan instal pustaka yang dibutuhkan:
   ```bash
   .\venv\Scripts\pip.exe install -r requirements.txt
   ```
2. Latih ulang model dan perbarui metrik evaluasi (opsional, hasil model sudah tersedia di folder `model/`):
   ```bash
   .\venv\Scripts\python.exe train_model.py
   ```
3. Jalankan aplikasi web Streamlit secara lokal:
   ```bash
   .\venv\Scripts\streamlit.exe run app.py
   ```
4. Akses aplikasi secara online (publik): 
```
https://uas-sistem-cerdas-titanic-miftakhulahmadrizky.streamlit.app/
```

## Refleksi Singkat
Melalui pembuatan proyek ini, saya mempelajari alur penerapan *machine learning* secara menyeluruh mulai dari pembersihan data (*handling missing values* pada kolom usia dan tarif Titanic), pembagian data latih dan uji, penyimpanan model ke format `.pkl`, hingga membangun antarmuka web interaktif berbasis Streamlit yang terhubung ke GitHub dan Streamlit Community Cloud. Tantangan utama berada pada penyesuaian format tipe data input pengguna agar selaras dengan fitur saat pelatihan model.