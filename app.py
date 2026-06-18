import streamlit as st
import pandas as pd
import joblib

# Konfigurasi Halaman
st.set_page_config(page_title="Prediksi Status TPS", page_icon="♻️", layout="centered")

st.title("♻️ Prediksi Status TPS")
st.markdown("""
Aplikasi web sederhana menggunakan **Random Forest Pipeline** untuk memprediksi status TPS. 
Masukkan data fitur mentah (tanpa perlu melakukan standarisasi manual) pada panel di sebelah kiri untuk melihat hasil prediksi.
""")

# Load Pipeline Model
@st.cache_resource
def load_model():
    # File pipeline sudah mencakup scaler, encoder, dan classifier
    return joblib.load('rf_pipeline.pkl')

try:
    pipeline = load_model()
    st.success("Pipeline model berhasil dimuat! Siap untuk melakukan prediksi.")
except FileNotFoundError:
    st.error("Model 'rf_pipeline.pkl' tidak ditemukan. Harap jalankan file modeling.ipynb terlebih dahulu untuk memproses dan menyimpan pipeline.")
    st.stop()

# Form Input di Sidebar
st.sidebar.header("Input Parameter Fitur")
st.sidebar.info("Masukkan data sesuai dengan nilai aslinya. Sistem akan melakukan standarisasi dan encoding secara otomatis berdasarkan Pipeline.")

# Input numerik
volume_sampah_kg = st.sidebar.number_input("Volume Sampah (kg)", min_value=0.0, value=250.0)
kapasitas_tps_kg = st.sidebar.number_input("Kapasitas TPS (kg)", min_value=0.0, value=500.0)
tahun = st.sidebar.selectbox("Tahun", [2019, 2020, 2021, 2022, 2023, 2024])
alamat_missing_flag = st.sidebar.selectbox("Alamat Bank Sampah Tidak Diketahui?", [0, 1], format_func=lambda x: "Ya (1)" if x == 1 else "Tidak (0)")

# Input kategorik (Akan di-OneHotEncode secara otomatis oleh pipeline)
st.sidebar.markdown("---")
st.sidebar.subheader("Informasi Lokasi")

# Pengguna dapat menginput nama desa/kelurahan dan nama bank sampah
bps_desa_kelurahan = st.sidebar.text_input("Nama Desa/Kelurahan (BPS)", value="CICADAS")
nama_unit_bank_sampah = st.sidebar.text_input("Nama Unit Bank Sampah", value="BANK SAMPAH TERATAI INDAH")

# Menggabungkan data menjadi dataframe sesuai format saat training (nama kolom harus presisi)
input_data = pd.DataFrame({
    'volume_sampah_kg': [volume_sampah_kg],
    'kapasitas_tps_kg': [kapasitas_tps_kg],
    'tahun': [tahun],
    'alamat_missing_flag': [alamat_missing_flag],
    'bps_desa_kelurahan': [bps_desa_kelurahan],
    'nama_unit_bank_sampah': [nama_unit_bank_sampah]
})

# Tampilkan preview input data
st.subheader("Preview Data Input")
st.dataframe(input_data)

# Tombol Prediksi
if st.button("Lakukan Prediksi"):
    # Lakukan prediksi langsung menggunakan pipeline
    prediction = pipeline.predict(input_data)
    
    st.subheader("Hasil Prediksi")
    
    # Berdasarkan dataset, 1 = Penuh, 0 = Tidak Penuh
    if prediction[0] == 1:
        st.error("🤖 **Prediksi:** TPS berstatus **PENUH** (Kelas 1)")
    else:
        st.success("🤖 **Prediksi:** TPS berstatus **TIDAK PENUH** (Kelas 0)")
