# Sistem Prediksi Keselamatan Penumpang Titanic Berbasis Web (Stremlit)

## Masalah yang diselesaikan
Memprediksi peluang keselamatan penumpang kapal Titanic berdasarkan 4 atribut utama penumpang (kelas tiket, jenis kelamin, usia, dan tarif tiket) melalui antarmuka web interaktif.

## Dataset
- Sumber: Titanic - Machine Learning from Disaster — Kaggle (`https://www.kaggle.com/c/titanic`)
- Lisensi: Open / Public Dataset (Kaggle Competition Data)
- Jumlah data: 891 baris data penumpang dengan 4 fitur input terpilih (`Pclass`, `Sex`, `Age`, `Fare`) dan 1 label target biner (`Survived`: 0 = Tidak Selamat, 1 = Selamat).

## Metode
- Preprocessing: Encoding variabel kategorikal `Sex` menjadi numerik (0/1) serta imputasi nilai median pada kolom `Age` dan `Fare` yang memiliki nilai kosong.
- Teknik: Machine Learning Supervised Classification menggunakan **Random Forest Classifier** (100 decision trees, `random_state=42`).
- Kerangka Web: Django (Python) untuk menerima input pengguna, memuat model tersimpan (`joblib`), dan menampilkan hasil prediksi.

## Hasil & evaluasi
- Pembagian data: 80% data latih (712 baris) dan 20% data uji (179 baris).
- Metrik pengujian: Akurasi ~80%, Presisi, dan Recall yang dimuat langsung dari `model/evaluasi.json` pada halaman utama aplikasi.

## Cara menjalankan
1. Instal dependensi: `pip install -r requirements.txt`
2. (Opsional) Latih ulang model: `python train_model.py`
3. Jalankan server web: `python manage.py runserver`
4. Buka browser dan akses `http://127.0.0.1:8000/`
5. Akses langsung secara online (publik): `https://miftakhulahmadrizky.pythonanywhere.com`

## Refleksi singkat
- **Yang berhasil:** Tahap preprocessing (encoding dan imputasi median) berhasil meningkatkan akurasi model hingga ~80% dan terintegrasi lancar dengan web Django.
- **Keterbatasan:** Fitur keluarga (`SibSp`, `Parch`) dan pelabuhan keberangkatan (`Embarked`) belum dilibatkan sebagai fitur prediksi.
- **Rencana perbaikan:** Menambahkan rekayasa fitur ukuran keluarga (*FamilySize*) dan perbandingan performa dengan algoritma Gradient Boosting.