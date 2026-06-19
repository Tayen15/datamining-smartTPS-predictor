import streamlit as st
import pandas as pd
import joblib

# =========================================================
# KONFIGURASI FILE
# =========================================================
MODEL_PATH = "rf_pipeline.pkl"
DATASET_PATH = "dataset_cleaned_lengkap.csv"

# Isi sesuai hasil evaluasi dari modeling.ipynb kamu
MODEL_METRICS = {
    "Accuracy": None,
    "Precision": None,
    "Recall": None,
    "F1-Score": None,
    "ROC-AUC": None
}

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="Prediksi Status TPS",
    page_icon="♻️",
    layout="centered"
)

st.title("♻️ Prediksi Status TPS Berbasis Data Bank Sampah")

st.markdown("""
Aplikasi ini merupakan **prototype prediksi status TPS** menggunakan model **Random Forest Pipeline**.  
Prediksi dilakukan berdasarkan data bank sampah dan fitur simulasi operasional yang merepresentasikan kondisi TPS pada wilayah terkait.
""")

st.info("""
📌 **Catatan Penting:**  
Data **volume sampah** yang ditampilkan pada aplikasi ini merupakan data simulasi/feature engineering.  
Sementara itu, informasi kapasitas TPS digunakan secara internal oleh sistem berdasarkan data referensi, 
bukan ditampilkan sebagai input pengguna.
""")

# =========================================================
# LOAD MODEL DAN DATASET
# =========================================================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_reference_data():
    return pd.read_csv(DATASET_PATH)


try:
    pipeline = load_model()
    st.success("✅ Pipeline model berhasil dimuat! Siap melakukan prediksi.")
except FileNotFoundError:
    st.error(f"❌ Model '{MODEL_PATH}' tidak ditemukan. Pastikan file model berada di folder yang sama dengan app.py.")
    st.stop()
except Exception as e:
    st.error(f"❌ Terjadi error saat memuat model: {e}")
    st.stop()

try:
    df_ref = load_reference_data()
except FileNotFoundError:
    st.error(f"❌ Dataset '{DATASET_PATH}' tidak ditemukan. Pastikan file dataset berada di folder yang sama dengan app.py.")
    st.stop()
except Exception as e:
    st.error(f"❌ Terjadi error saat membaca dataset: {e}")
    st.stop()

# =========================================================
# VALIDASI KOLOM DATASET
# =========================================================
required_columns = [
    "bps_desa_kelurahan",
    "nama_unit_bank_sampah",
    "tahun",
    "alamat_missing_flag",
    "volume_sampah_kg",
    "kapasitas_tps_kg"
]

missing_columns = [col for col in required_columns if col not in df_ref.columns]

if missing_columns:
    st.error(f"❌ Kolom berikut tidak ditemukan di dataset: {missing_columns}")
    st.stop()

# Rapikan isi data
df_ref["bps_desa_kelurahan"] = df_ref["bps_desa_kelurahan"].astype(str).str.strip()
df_ref["nama_unit_bank_sampah"] = df_ref["nama_unit_bank_sampah"].astype(str).str.strip()

df_ref["tahun"] = pd.to_numeric(df_ref["tahun"], errors="coerce")
df_ref["alamat_missing_flag"] = pd.to_numeric(df_ref["alamat_missing_flag"], errors="coerce").fillna(0).astype(int)
df_ref["volume_sampah_kg"] = pd.to_numeric(df_ref["volume_sampah_kg"], errors="coerce")
df_ref["kapasitas_tps_kg"] = pd.to_numeric(df_ref["kapasitas_tps_kg"], errors="coerce")

df_ref = df_ref.dropna(subset=[
    "bps_desa_kelurahan",
    "nama_unit_bank_sampah",
    "tahun",
    "volume_sampah_kg",
    "kapasitas_tps_kg"
])

df_ref["tahun"] = df_ref["tahun"].astype(int)

# =========================================================
# SIDEBAR INPUT
# =========================================================
st.sidebar.header("Input Parameter Prediksi")
st.sidebar.info("""
Pilih lokasi bank sampah dan tahun data. Sistem akan menggunakan data referensi dari dataset untuk melakukan prediksi status TPS.
""")

# =========================================================
# INPUT LOKASI
# =========================================================
st.sidebar.markdown("---")
st.sidebar.subheader("Informasi Lokasi")

desa_placeholder = "-- Pilih Desa/Kelurahan --"
bank_placeholder = "-- Pilih Unit Bank Sampah --"

desa_options = [desa_placeholder] + sorted(
    df_ref["bps_desa_kelurahan"].dropna().unique().tolist()
)

bps_desa_kelurahan = st.sidebar.selectbox(
    "Nama Desa/Kelurahan (BPS)",
    desa_options
)

if bps_desa_kelurahan != desa_placeholder:
    filtered_desa = df_ref[df_ref["bps_desa_kelurahan"] == bps_desa_kelurahan]

    bank_sampah_options = [bank_placeholder] + sorted(
        filtered_desa["nama_unit_bank_sampah"].dropna().unique().tolist()
    )
else:
    filtered_desa = pd.DataFrame()
    bank_sampah_options = [bank_placeholder]

nama_unit_bank_sampah = st.sidebar.selectbox(
    "Nama Unit Bank Sampah",
    bank_sampah_options
)

# =========================================================
# INPUT TAHUN BERDASARKAN LOKASI
# =========================================================
if nama_unit_bank_sampah != bank_placeholder and not filtered_desa.empty:
    filtered_bank = filtered_desa[
        filtered_desa["nama_unit_bank_sampah"] == nama_unit_bank_sampah
    ]

    tahun_options = sorted(filtered_bank["tahun"].dropna().unique().tolist())
else:
    filtered_bank = pd.DataFrame()
    tahun_options = sorted(df_ref["tahun"].dropna().unique().tolist())

if len(tahun_options) == 0:
    tahun_options = [2022]

tahun = st.sidebar.selectbox(
    "Tahun",
    tahun_options,
    index=len(tahun_options) - 1
)

# =========================================================
# AMBIL DATA REFERENSI BERDASARKAN PILIHAN
# =========================================================
selected_ref = pd.DataFrame()

if (
    bps_desa_kelurahan != desa_placeholder and
    nama_unit_bank_sampah != bank_placeholder and
    tahun is not None
):
    selected_ref = df_ref[
        (df_ref["bps_desa_kelurahan"] == bps_desa_kelurahan) &
        (df_ref["nama_unit_bank_sampah"] == nama_unit_bank_sampah) &
        (df_ref["tahun"] == tahun)
    ]

if not selected_ref.empty:
    ref_row = selected_ref.iloc[0]

    default_volume = float(ref_row["volume_sampah_kg"])
    default_kapasitas = float(ref_row["kapasitas_tps_kg"])
    default_alamat_missing = int(ref_row["alamat_missing_flag"])

    if "alamat_bank_sampah" in df_ref.columns:
        alamat_ref = ref_row["alamat_bank_sampah"]
        st.sidebar.caption(f"Alamat referensi: {alamat_ref}")
else:
    default_volume = 250.0
    default_kapasitas = 500.0
    default_alamat_missing = 0

# =========================================================
# INPUT DATA SIMULASI OPERASIONAL
# =========================================================
st.sidebar.markdown("---")
st.sidebar.subheader("Data Simulasi Operasional")

widget_key_suffix = f"{bps_desa_kelurahan}_{nama_unit_bank_sampah}_{tahun}"

volume_sampah_kg = st.sidebar.number_input(
    "Volume Sampah (kg)",
    min_value=0.0,
    value=default_volume,
    step=10.0,
    key=f"volume_{widget_key_suffix}"
)

# Kapasitas TPS tidak ditampilkan kepada user.
# Nilainya tetap dipakai model secara internal berdasarkan dataset.
kapasitas_tps_kg = default_kapasitas

# Alamat missing flag tetap ditampilkan karena tidak langsung membocorkan status penuh/tidak penuh.
alamat_missing_flag = st.sidebar.selectbox(
    "Alamat Bank Sampah Tidak Diketahui?",
    [0, 1],
    index=default_alamat_missing if default_alamat_missing in [0, 1] else 0,
    format_func=lambda x: "Ya (1)" if x == 1 else "Tidak (0)",
    key=f"alamat_missing_{widget_key_suffix}"
)

# =========================================================
# VALIDASI INPUT
# =========================================================
error_messages = []

if bps_desa_kelurahan == desa_placeholder:
    error_messages.append("Mohon pilih nama desa/kelurahan terlebih dahulu.")

if nama_unit_bank_sampah == bank_placeholder:
    error_messages.append("Mohon pilih nama unit bank sampah terlebih dahulu.")

if volume_sampah_kg <= 0:
    error_messages.append("Volume sampah tidak boleh 0 atau negatif.")

if kapasitas_tps_kg <= 0:
    error_messages.append("Data kapasitas referensi tidak valid. Silakan pilih lokasi atau tahun lain.")

if kapasitas_tps_kg > 0:
    rasio_kapasitas = volume_sampah_kg / kapasitas_tps_kg
    persentase_kapasitas = rasio_kapasitas * 100
else:
    rasio_kapasitas = 0
    persentase_kapasitas = 0

# =========================================================
# DATA INPUT UNTUK MODEL
# =========================================================
# input_data lengkap untuk model
input_data = pd.DataFrame({
    "volume_sampah_kg": [volume_sampah_kg],
    "kapasitas_tps_kg": [kapasitas_tps_kg],
    "tahun": [tahun],
    "alamat_missing_flag": [alamat_missing_flag],
    "bps_desa_kelurahan": [
        bps_desa_kelurahan if bps_desa_kelurahan != desa_placeholder else ""
    ],
    "nama_unit_bank_sampah": [
        nama_unit_bank_sampah if nama_unit_bank_sampah != bank_placeholder else ""
    ]
})

# preview_data khusus untuk ditampilkan ke user
# kapasitas_tps_kg sengaja tidak ditampilkan agar user tidak langsung menebak hasil prediksi
preview_data = pd.DataFrame({
    "volume_sampah_kg": [volume_sampah_kg],
    "tahun": [tahun],
    "alamat_missing_flag": [alamat_missing_flag],
    "bps_desa_kelurahan": [
        bps_desa_kelurahan if bps_desa_kelurahan != desa_placeholder else ""
    ],
    "nama_unit_bank_sampah": [
        nama_unit_bank_sampah if nama_unit_bank_sampah != bank_placeholder else ""
    ]
})

# =========================================================
# PREVIEW DATA INPUT
# =========================================================
st.subheader("Preview Data Input")

st.markdown("""
Data berikut adalah input yang ditampilkan kepada pengguna sebelum dilakukan prediksi.
""")

st.dataframe(
    preview_data,
    use_container_width=True,
    hide_index=True
)

st.caption("""
Catatan: Beberapa fitur referensi digunakan secara internal oleh sistem agar model tetap sesuai dengan data training.
""")

# =========================================================
# RINGKASAN LOKASI
# =========================================================
st.subheader("Ringkasan Lokasi")

col_lokasi1, col_lokasi2 = st.columns(2)

with col_lokasi1:
    st.metric(
        label="Desa/Kelurahan",
        value=bps_desa_kelurahan if bps_desa_kelurahan != desa_placeholder else "-"
    )

with col_lokasi2:
    st.metric(
        label="Tahun",
        value=str(tahun) if tahun is not None else "-"
    )

st.markdown(f"""
**Nama Unit Bank Sampah:**  
{nama_unit_bank_sampah if nama_unit_bank_sampah != bank_placeholder else "-"}
""")

st.metric(
    label="Volume Sampah",
    value=f"{volume_sampah_kg:.0f} kg"
)

# =========================================================
# STATUS DATA INPUT
# =========================================================
st.subheader("Status Data Input")

if error_messages:
    for error in error_messages:
        st.warning(f"⚠️ {error}")
else:
    st.success("✅ Data input sudah lengkap dan siap diprediksi.")

if volume_sampah_kg > 0 and kapasitas_tps_kg > 0:
    if rasio_kapasitas > 1.5:
        st.warning("⚠️ Volume sampah yang dimasukkan tergolong sangat tinggi berdasarkan data referensi internal.")

# =========================================================
# INFORMASI MODEL
# =========================================================
st.subheader("Informasi Model")

st.markdown("""
Model yang digunakan adalah **Random Forest Pipeline**.  
Pipeline ini memungkinkan sistem menerima data mentah dari pengguna, kemudian melakukan preprocessing dan klasifikasi secara otomatis.
""")

with st.expander("Lihat Performa Model"):
    if all(value is not None for value in MODEL_METRICS.values()):
        metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)

        with metric_col1:
            st.metric("Accuracy", MODEL_METRICS["Accuracy"])

        with metric_col2:
            st.metric("Precision", MODEL_METRICS["Precision"])

        with metric_col3:
            st.metric("Recall", MODEL_METRICS["Recall"])

        with metric_col4:
            st.metric("F1-Score", MODEL_METRICS["F1-Score"])

        with metric_col5:
            st.metric("ROC-AUC", MODEL_METRICS["ROC-AUC"])
    else:
        st.info("""
        Nilai performa model belum diisi pada kode aplikasi.

        Silakan isi bagian `MODEL_METRICS` di atas sesuai hasil evaluasi dari `modeling.ipynb`.

        Contoh:
        - Accuracy: 0.95
        - Precision: 0.96
        - Recall: 0.95
        - F1-Score: 0.95
        - ROC-AUC: 0.98
        """)

with st.expander("Lihat Detail Internal untuk Keperluan Demo Dosen"):
    st.markdown("""
    Bagian ini hanya untuk kebutuhan penjelasan teknis kepada dosen, bukan untuk tampilan utama pengguna.
    """)

    detail_internal_df = pd.DataFrame({
        "Keterangan": [
            "Volume Sampah",
            "Kapasitas TPS Referensi",
            "Rasio Kapasitas",
            "Alamat Missing Flag"
        ],
        "Nilai": [
            f"{volume_sampah_kg:.0f} kg",
            f"{kapasitas_tps_kg:.0f} kg",
            f"{persentase_kapasitas:.2f}%",
            alamat_missing_flag
        ]
    })

    st.dataframe(
        detail_internal_df,
        use_container_width=True,
        hide_index=True
    )

# =========================================================
# FUNGSI BANTUAN PREDIKSI
# =========================================================
def is_prediction_penuh(pred_value):
    pred_text = str(pred_value).strip().lower()

    if pred_text in ["1", "penuh", "full"]:
        return True

    return False


def get_prediction_probabilities(model, data):
    if not hasattr(model, "predict_proba"):
        return None, None, None

    proba = model.predict_proba(data)[0]

    if hasattr(model, "classes_"):
        classes = list(model.classes_)
    else:
        classes = [0, 1]

    prob_tidak_penuh = None
    prob_penuh = None

    for cls, prob in zip(classes, proba):
        cls_text = str(cls).strip().lower()

        if cls_text in ["0", "tidak penuh", "tidak_penuh", "not full"]:
            prob_tidak_penuh = prob * 100

        elif cls_text in ["1", "penuh", "full"]:
            prob_penuh = prob * 100

    if prob_tidak_penuh is None and len(proba) > 0:
        prob_tidak_penuh = proba[0] * 100

    if prob_penuh is None and len(proba) > 1:
        prob_penuh = proba[1] * 100

    confidence = max(proba) * 100

    return prob_tidak_penuh, prob_penuh, confidence

# =========================================================
# TOMBOL PREDIKSI
# =========================================================
st.subheader("Prediksi Status TPS")

if st.button("Lakukan Prediksi"):
    if error_messages:
        for error in error_messages:
            st.error(f"❌ {error}")

    else:
        try:
            prediction = pipeline.predict(input_data)
            prediction_value = prediction[0]

            prob_tidak_penuh, prob_penuh, confidence = get_prediction_probabilities(
                pipeline,
                input_data
            )

            st.subheader("Hasil Prediksi")

            if is_prediction_penuh(prediction_value):
                st.error("🤖 **Prediksi: TPS berstatus PENUH**")

                st.markdown(f"""
                Berdasarkan data lokasi, tahun, dan volume sampah sebesar **{volume_sampah_kg:.0f} kg**, 
                sistem memprediksi bahwa TPS pada lokasi tersebut berada dalam kondisi **penuh**.
                """)

                st.warning("""
                📌 **Rekomendasi Tindakan:**  
                TPS perlu diprioritaskan untuk pengangkutan agar tidak terjadi penumpukan sampah.
                """)

            else:
                st.success("🤖 **Prediksi: TPS berstatus TIDAK PENUH**")

                st.markdown(f"""
                Berdasarkan data lokasi, tahun, dan volume sampah sebesar **{volume_sampah_kg:.0f} kg**, 
                sistem memprediksi bahwa TPS pada lokasi tersebut masih dalam kondisi **tidak penuh**.
                """)

                st.info("""
                📌 **Rekomendasi Tindakan:**  
                TPS masih dalam kondisi aman, sehingga pengangkutan dapat dilakukan sesuai jadwal normal.
                """)

            # Probabilitas Prediksi
            if confidence is not None:
                st.subheader("Probabilitas Prediksi")

                col_prob1, col_prob2, col_prob3 = st.columns(3)

                with col_prob1:
                    st.metric(
                        "Probabilitas Tidak Penuh",
                        f"{prob_tidak_penuh:.2f}%" if prob_tidak_penuh is not None else "-"
                    )

                with col_prob2:
                    st.metric(
                        "Probabilitas Penuh",
                        f"{prob_penuh:.2f}%" if prob_penuh is not None else "-"
                    )

                with col_prob3:
                    st.metric(
                        "Confidence Model",
                        f"{confidence:.2f}%"
                    )
            else:
                st.info("Model ini belum mendukung tampilan probabilitas karena tidak tersedia fungsi `predict_proba`.")

            # Ringkasan Hasil
            st.subheader("Ringkasan Input dan Hasil")

            hasil_label = "Penuh" if is_prediction_penuh(prediction_value) else "Tidak Penuh"

            summary_df = pd.DataFrame({
                "Keterangan": [
                    "Desa/Kelurahan",
                    "Nama Unit Bank Sampah",
                    "Tahun",
                    "Volume Sampah",
                    "Alamat Missing Flag",
                    "Hasil Prediksi"
                ],
                "Nilai": [
                    bps_desa_kelurahan,
                    nama_unit_bank_sampah,
                    tahun,
                    f"{volume_sampah_kg:.0f} kg",
                    alamat_missing_flag,
                    hasil_label
                ]
            })

            st.dataframe(
                summary_df,
                use_container_width=True,
                hide_index=True
            )

        except Exception as e:
            st.error(f"❌ Terjadi error saat melakukan prediksi: {e}")
            st.info("""
            Kemungkinan penyebab:
            1. Nama kolom input tidak sama dengan kolom saat training.
            2. Pipeline model belum mencakup preprocessing untuk data kategorik.
            3. Ada kategori baru yang tidak dikenali model.
            4. File model `rf_pipeline.pkl` belum dibuat ulang setelah perubahan dataset.
            """)