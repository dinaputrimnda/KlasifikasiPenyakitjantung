import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ==========================================================
# 1. KONFIGURASI HALAMAN
# ==========================================================
st.set_page_config(
    page_title="CardioCare | Klasifikasi Penyakit Jantung",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================================
# 2. KONFIGURASI (HARUS SAMA DENGAN NOTEBOOK COLAB)
# ==========================================================
# 13 fitur awal, urutan sama dengan variabel FITUR di notebook (scaler dilatih pada 13 fitur ini)
FITUR = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
         'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# 9 fitur hasil seleksi (FITUR_RFECV di notebook). Dibuang: sex, fbs, restecg, slope
FITUR_TERPILIH = ['age', 'cp', 'trestbps', 'chol', 'thalch', 'exang', 'oldpeak', 'ca', 'thal']
IDX_TERPILIH = [FITUR.index(f) for f in FITUR_TERPILIH]

# Batas IQR Capping dari data latih (08_batas_outlier.csv di notebook)
BATAS = {
    'age':      (27.50, 79.50),
    'trestbps': (90.00, 170.00),
    'chol':     (31.62, 406.62),
    'thalch':   (67.50, 207.50),
    'oldpeak':  (-2.25, 3.75),
}

# Label Encoding (urut alfabet, sama dengan LabelEncoder di notebook)
MAP_CP    = {"Asymptomatic (tidak ada gejala)": 0, "Atypical Angina (tidak khas)": 1,
             "Non-anginal (bukan angina)": 2, "Typical Angina (khas)": 3}
MAP_EXANG = {"Tidak": 0, "Ya": 1}
MAP_THAL  = {"Fixed Defect": 0, "Normal": 1, "Reversable Defect": 2}

# Hasil pengujian Skenario 1 (SMOTE) pada data uji 92 data (classification report di notebook)
AKURASI_UJI = 60.87
PRESISI_UJI = {0: 77, 1: 55, 2: 57, 3: 36, 4: 33}   # % prediksi kelas tsb yang benar pada data uji

KETERANGAN_KELAS = {
    0: "Kelas 0 - tidak terindikasi penyakit jantung",
    1: "Kelas 1 - ringan",
    2: "Kelas 2 - sedang",
    3: "Kelas 3 - berat",
    4: "Kelas 4 - kritis",
}


# ==========================================================
# 3. LOAD MODEL & SCALER
# ==========================================================
@st.cache_resource
def load_components():
    folder = os.path.dirname(os.path.abspath(__file__))
    try:
        scaler = joblib.load(os.path.join(folder, '09_scaler_minmax.pkl'))
        model = joblib.load(os.path.join(folder, 'A_model_final.pkl'))   # Skenario 1 (RF + RFECV + SMOTE)
        return scaler, model, "Success"
    except Exception as e:
        return None, None, (f"Gagal memuat '09_scaler_minmax.pkl' / 'A_model_final.pkl' "
                            f"(pastikan ada di folder yang sama dengan app.py): {e}")


scaler, model, status = load_components()


# ==========================================================
# 4. PREPROCESSING INFERENSI (urutan sama dengan pelatihan)
#    encoding -> IQR capping -> MinMaxScaler (13 fitur) -> ambil 9 fitur
# ==========================================================
def preprocess(nilai: dict):
    """nilai: dict 9 fitur (sudah di-encoding). Return (X 9 fitur, daftar fitur yang di-capping)."""
    # Fitur yang tidak dipakai model diisi 0 (hanya agar scaler mendapat 13 kolom; dibuang setelah scaling)
    x = np.array([[nilai.get(f, 0) for f in FITUR]], dtype=float)

    dicapping = []
    for f, (bawah, atas) in BATAS.items():
        i = FITUR.index(f)
        asli = x[0, i]
        x[0, i] = np.clip(asli, bawah, atas)
        if x[0, i] != asli:
            dicapping.append((f, asli, x[0, i]))

    return scaler.transform(x)[:, IDX_TERPILIH], dicapping


# ==========================================================
# 5. SIDEBAR
# ==========================================================
with st.sidebar:
    st.header("🧭 Navigasi Menu")
    menu = st.radio("Pilih Halaman:", ["Halaman Prediksi", "Informasi Fitur Medis"])

    st.divider()
    st.header("🏥 Panduan Sistem")
    st.info("Klasifikasi tingkat keparahan penyakit jantung (5 kelas) menggunakan "
            "Random Forest + RFECV (9 fitur) + SMOTE.")
    st.markdown("### 📝 Instruksi:")
    st.markdown("1. Buka **Informasi Fitur Medis** jika bingung dengan istilah medis.")
    st.markdown("2. Buka **Halaman Prediksi** untuk memasukkan data pasien.")
    st.markdown("3. Klik tombol **Analisis Klasifikasi**.")

# ==========================================================
# 6. HALAMAN: INFORMASI FITUR MEDIS
# ==========================================================
if menu == "Informasi Fitur Medis":
    st.title("📚 Glosarium & Informasi Fitur Medis")
    st.markdown("Panduan ini membantu memahami arti setiap parameter medis pada form prediksi.")
    st.divider()

    with st.expander("👤 Demografi & Gejala Dasar", expanded=True):
        st.markdown("""
        * **Umur**: Usia pasien dalam tahun.
        * **Tipe Nyeri Dada (cp)**:
            * *Typical Angina*: nyeri dada khas karena jantung kekurangan oksigen.
            * *Atypical Angina*: nyeri dada tidak khas.
            * *Non-anginal*: nyeri dada bukan karena masalah jantung (misalnya otot atau asam lambung).
            * *Asymptomatic*: tidak merasakan nyeri sama sekali.
        * **Tekanan Darah Istirahat (trestbps)**: tekanan darah sistolik saat istirahat (mmHg).
        """)

    with st.expander("🩸 Tes Darah & EKG", expanded=True):
        st.markdown("""
        * **Kolesterol Serum (chol)**: kadar kolesterol total (mg/dl). Batas normal biasanya di bawah 200 mg/dl.
        * **Detak Jantung Maksimum (thalch)**: detak jantung tertinggi yang dicapai saat uji latihan (bpm).
        * *Catatan: jenis kelamin (sex), gula darah puasa (fbs), hasil EKG istirahat (restecg), dan slope ST (slope) tidak digunakan model karena tidak terpilih oleh seleksi fitur RFECV.*
        """)

    with st.expander("🏃 Hasil Stress Test (Tes Aktivitas)", expanded=True):
        st.markdown("""
        * **Angina Akibat Olahraga (exang)**: apakah nyeri dada muncul saat aktivitas fisik.
        * **Depresi ST (oldpeak)**: penurunan segmen ST pada EKG akibat olahraga dibandingkan saat istirahat.
        * **Jumlah Pembuluh Darah Utama (ca)**: jumlah pembuluh darah utama (0-3) yang terdeteksi melalui fluoroskopi.
        * **Hasil Thalassemia (thal)**: *Normal*, *Fixed Defect* (cacat tetap), *Reversable Defect* (cacat yang dapat pulih).
        """)

# ==========================================================
# 7. HALAMAN: FORM PREDIKSI
# ==========================================================
else:
    st.title("🩺 CardioCare: Form Klasifikasi")
    st.markdown("Masukkan data klinis pasien (9 fitur hasil seleksi RFECV).")

    if status != "Success":
        st.error(f"⚠️ {status}")
        st.stop()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👤 Demografi & Gejala")
        age = st.number_input("Umur (tahun)", min_value=20, max_value=100, value=50, step=1)
        cp = st.selectbox("Tipe Nyeri Dada (Chest Pain)", list(MAP_CP.keys()))
        trestbps = st.number_input("Tekanan Darah Istirahat (mmHg)", min_value=80, max_value=200,
                                   value=120, step=1)

    with col2:
        st.subheader("🩸 Tes Darah & EKG")
        chol = st.number_input("Kolesterol Serum (mg/dl)", min_value=100, max_value=600,
                               value=200, step=1)
        thalch = st.number_input("Detak Jantung Maksimum (bpm)", min_value=60, max_value=220,
                                 value=150, step=1)

    with col3:
        st.subheader("🏃 Hasil Stress Test")
        exang = st.selectbox("Angina Akibat Olahraga?", list(MAP_EXANG.keys()))
        oldpeak = st.number_input("Depresi ST (Oldpeak)", min_value=-3.0, max_value=6.5,
                                  value=0.0, step=0.1, format="%.1f")
        ca = st.selectbox("Jumlah Pembuluh Darah Utama Terlihat", [0, 1, 2, 3])
        thal = st.selectbox("Hasil Thalassemia", list(MAP_THAL.keys()), index=1)

    st.divider()

    if st.button("🔬 Analisis Klasifikasi", type="primary"):
        nilai = {
            'age': age, 'cp': MAP_CP[cp], 'trestbps': trestbps, 'chol': chol,
            'thalch': thalch, 'exang': MAP_EXANG[exang], 'oldpeak': oldpeak,
            'ca': ca, 'thal': MAP_THAL[thal],
        }

        with st.spinner("Memproses data klinis..."):
            try:
                X_input, dicapping = preprocess(nilai)
                pred = int(model.predict(X_input)[0])
                proba = model.predict_proba(X_input)[0]
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses data: {e}")
                st.stop()

        if dicapping:
            teks = ", ".join(f"{f} ({a:g} → {b:g})" for f, a, b in dicapping)
            st.info("Nilai di luar rentang data latih disesuaikan ke batas IQR, "
                    f"sama seperti pada tahap pelatihan: {teks}.")

        st.subheader("📋 Hasil Klasifikasi")
        if pred == 0:
            st.success(f"**{KETERANGAN_KELAS[0]}**")
        else:
            st.error(f"**Terindikasi penyakit jantung - {KETERANGAN_KELAS[pred]}**")

        k1, k2, k3 = st.columns(3)
        k1.metric("Keyakinan model untuk hasil ini", f"{proba[list(model.classes_).index(pred)] * 100:.1f}%")
        k2.metric("Akurasi model (data uji)", f"{AKURASI_UJI:.2f}%")
        k3.metric(f"Ketepatan prediksi Kelas {pred} (data uji)", f"{PRESISI_UJI[pred]}%")
        st.caption("Keyakinan = probabilitas kelas terpilih dari model. Ketepatan = persentase prediksi "
                   f"Kelas {pred} yang benar pada data uji (presisi).")

        df_p = pd.DataFrame({
            "Kelas": [f"Kelas {int(c)}" for c in model.classes_],
            "Probabilitas": proba,
        }).set_index("Kelas")
        st.caption("Probabilitas prediksi model per kelas")
        st.bar_chart(df_p)

        st.warning(
            "Aplikasi ini merupakan prototipe penelitian, bukan alat diagnosis medis. "
            "Akurasi model pada data uji 60,87% (92 data); pengenalan kelas 2, 3, dan "
            "terutama kelas 3 dan 4 masih lemah karena data minoritas sangat sedikit. "
            "Hasil harus dikonfirmasi oleh tenaga medis.")