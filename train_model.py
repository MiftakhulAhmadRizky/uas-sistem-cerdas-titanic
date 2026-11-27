import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# 1. Membaca dataset Titanic dari folder dataset
df = pd.read_csv("dataset/titanic.csv")

# 2. Preprocessing: ubah teks Sex menjadi angka & isi data Age/Fare yang kosong dengan median
df["Sex"] = df["Sex"].map({"female": 0, "male": 1})
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

# 3. Pilih 4 fitur input (X) dan 1 label target (y: 0 = Tidak Selamat, 1 = Selamat)
fitur = ["Pclass", "Sex", "Age", "Fare"]
X = df[fitur]
y = df["Survived"]

# 4. Split 80% data latih dan 20% data uji
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Latih model Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluasi model pada data uji
y_pred = model.predict(X_test)
hasil_evaluasi = {
    "total_data": len(df),
    "data_latih": len(X_train),
    "data_uji": len(X_test),
    "akurasi": round(accuracy_score(y_test, y_pred) * 100, 2),
    "presisi": round(precision_score(y_test, y_pred) * 100, 2),
    "recall": round(recall_score(y_test, y_pred) * 100, 2),
}

# 7. Simpan model baru dan hasil evaluasinya
joblib.dump(model, "model/model_rf.pkl")
with open("model/evaluasi.json", "w") as f:
    json.dump(hasil_evaluasi, f, indent=2)

print("Model Titanic berhasil dilatih! Metrik evaluasi:", hasil_evaluasi)