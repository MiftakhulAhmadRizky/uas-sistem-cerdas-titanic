import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Prediksi Keselamatan Titanic", page_icon="🚢", layout="centered")

# 1. Muat model dan metrik evaluasi yang sudah dilatih oleh train_model.py
@st.cache_resource
def load_assets():
    model = joblib.load("model/model_rf.pkl")
    with open("model/evaluasi.json", "r") as f:
        evaluasi = json.load(f)
    return model, evaluasi

model, evaluasi = load_assets()

st.title("Aplikasi Sistem Cerdas: Prediksi Keselamatan Penumpang Titanic")
st.caption("Metode: Machine Learning (Random Forest Classifier)")

# 2. Bagian Input Data Penumpang
with st.container(border=True):
    st.subheader("1. Input Data Penumpang")

    pclass_label = st.selectbox(
        "Kelas Tiket (Pclass):",
        options=[1, 2, 3],
        format_func=lambda x: {
            1: "Kelas 1 (First Class / VIP)",
            2: "Kelas 2 (Second Class)",
            3: "Kelas 3 (Economy Class)",
        }[x],
    )

    sex_label = st.selectbox("Jenis Kelamin:", ["Perempuan", "Laki-laki"])
    sex_num = 0 if sex_label == "Perempuan" else 1

    age = st.number_input("Usia (Tahun):", min_value=0.0, max_value=100.0, value=28.0, step=1.0)
    fare = st.number_input("Tarif Tiket / Fare (USD):", min_value=0.0, max_value=600.0, value=15.5, step=0.5)

    if st.button("Proses Prediksi", type="primary"):
        data_baru = pd.DataFrame([[pclass_label, sex_num, age, fare]], columns=["Pclass", "Sex", "Age", "Fare"])
        prediksi = model.predict(data_baru)[0]
        prob_selamat = model.predict_proba(data_baru)[0][1] * 100

        if prediksi == 1:
            st.success(f"**Hasil Prediksi Sistem:** Selamat (Survived) — **Probabilitas Selamat:** {prob_selamat:.2f}%")
        else:
            st.error(f"**Hasil Prediksi Sistem:** Tidak Selamat (Not Survived) — **Probabilitas Selamat:** {prob_selamat:.2f}%")

# 3. Bagian Informasi Dataset & Evaluasi Model
with st.container(border=True):
    st.subheader("2. Informasi Dataset & Evaluasi Model")
    st.markdown("**Sumber Dataset:** Titanic - Machine Learning from Disaster (Kaggle) — Lisensi: Public / Open Competition Data")

    tabel_evaluasi = pd.DataFrame([{
        "Total Data": f"{evaluasi['total_data']} baris",
        "Split Latih / Uji": f"{evaluasi['data_latih']} / {evaluasi['data_uji']} (70:30)",
        "Akurasi Uji": f"{evaluasi['akurasi']}%",
        "Presisi": f"{evaluasi['presisi']}%",
        "Recall": f"{evaluasi['recall']}%",
    }])
    st.dataframe(tabel_evaluasi, hide_index=True, use_container_width=True)