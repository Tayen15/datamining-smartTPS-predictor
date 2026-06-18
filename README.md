# ♻️ Prediksi Status Kapasitas TPS

Proyek ini merupakan implementasi pemodelan **Data Mining** untuk memprediksi status Tempat Pembuangan Sementara (TPS) apakah berada dalam kondisi **Penuh (Kelas 1)** atau **Tidak Penuh (Kelas 0)**. Proyek ini dibangun sebagai bagian dari tugas akhir (UAS) mata kuliah Data Mining.

## 📌 Deskripsi Singkat
Dalam proyek ini, dilakukan perbandingan performa antara algoritma **Decision Tree** dan **Random Forest**. Model dibangun menggunakan `Pipeline` dari Scikit-Learn yang mengintegrasikan proses *preprocessing* (`StandardScaler` untuk data numerik & `OneHotEncoder` untuk data teks/kategorikal) dengan algoritma klasifikasi. 

Model Random Forest terbaik diekspor dalam format `.pkl` dan kemudian diintegrasikan ke dalam antarmuka aplikasi web berbasis **Streamlit** untuk *deployment*.

## 📊 Dataset
Dataset memuat informasi kapasitas, volume sampah, tahun observasi, dan lokasi (seperti kelurahan dan nama bank sampah).
- File data: `data/dataset_cleaned_lengkap.csv`

## 🛠️ Teknologi yang Digunakan
- **Bahasa Pemrograman**: Python
- **Eksplorasi & Pemodelan**: Jupyter Notebook (`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`)
- **Deployment**: Streamlit
- **Serialisasi Model**: `joblib`

## 🚀 Cara Menjalankan Aplikasi Secara Lokal

1. **Buka Terminal/Command Prompt** di direktori proyek ini.
2. **Install semua dependensi** yang dibutuhkan dengan menjalankan perintah berikut:
   ```bash
   pip install -r requirements.txt
   ```
3. *(Opsional)* **Melatih Ulang Model:**
   Buka dan jalankan file `modeling.ipynb` dari awal hingga akhir. Proses ini akan melatih ulang model dan menghasilkan file `rf_pipeline.pkl` yang baru.
4. **Jalankan Aplikasi Web:**
   Ketik perintah berikut di terminal:
   ```bash
   streamlit run app.py
   ```
5. Akses antarmuka yang muncul secara otomatis di browser lokal Anda.

## 📁 Struktur File & Folder Utama
```text
.
├── data/
│   └── dataset_cleaned_lengkap.csv   # Dataset yang digunakan untuk training
├── app.py                            # File utama aplikasi Streamlit
├── modeling.ipynb                    # Notebook berisi tahap EDA, Modeling, Evaluasi, dan Export Pipeline
├── requirements.txt                  # Daftar library yang dibutuhkan
├── rf_pipeline.pkl                   # Model Pipeline yang sudah dilatih (Generate dari notebook)
└── README.md                         # Dokumentasi repository ini
```
