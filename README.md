# Analisis Sentimen Twitter Mengenai Pengaruh Tokoh Politik Dengan Metode Distance-Weighted K-Nearest Neighbor (DWKNN)

Aplikasi ini merupakan sistem klasifikasi sentimen berbasis teks yang dirancang untuk menganalisis opini masyarakat terhadap tokoh politik melalui media sosial Twitter. Opini tersebut diklasifikasikan ke dalam tiga kategori sentimen: **positif**, **netral**, dan **negatif**.

Sistem ini menggunakan algoritma **Distance-Weighted K-Nearest Neighbor (DWKNN)** yang memberikan bobot lebih besar pada data latih yang paling dekat. Pendekatan ini diharapkan mampu meningkatkan akurasi klasifikasi dan memberikan gambaran yang lebih representatif mengenai persepsi publik terhadap tokoh politik tertentu.

---

## 📌 Fitur Utama

- Klasifikasi sentimen teks menjadi positif, netral, atau negatif.
- Antarmuka pengguna berbasis web menggunakan Streamlit.
- Kemampuan unggah dataset dan uji prediksi secara interaktif.
- Dukungan preprocessing teks bahasa Indonesia (tokenisasi, stopword removal, stemming, dsb.).
- Implementasi algoritma DWKNN berbasis scikit-learn dan numpy.

---

## 🛠️ Instalasi dan Penggunaan

### 1. Instalasi Python dan Git

Silakan unduh dan instal dari situs resmi berikut:

- [Python](https://www.python.org)
- [Git](https://git-scm.com)

### 2. Clone Repository

Buka Command Prompt atau terminal, lalu jalankan:

```bash
git clone https://github.com/SuryaAdiPalguna/Analisis_Sentimen_Tokoh_DWKNN.git
```

Masuk ke direktori proyek:

```bash
cd Analisis_Sentimen_Tokoh_DWKNN
```

### 3. Instalasi Library

Instal semua dependensi dengan menjalankan:

```bash
pip install streamlit
pip install pandas
pip install numpy
pip install nltk
pip install regex
pip install git+https://github.com/ariaghora/mpstemmer.git
pip install scikit-learn
pip install imbalanced-learn
pip install openpyxl
pip install Levenshtein
```

### 4. Jalankan Aplikasi

Masuk ke folder aplikasi dan jalankan:

```bash
cd Application
streamlit run app.py
```

Jika aplikasi tidak terbuka otomatis di browser, akses manual melalui:

- `http://localhost:8501`
- atau `http://<alamat-IP-lokal>:8501`

---

## 📄 Panduan Penggunaan

### Halaman `Home`

- Unggah dataset `.xlsx` (format contoh tersedia di repositori).
- Contoh dataset dapat diunduh di sini:
  [train.xlsx – Dataset Contoh](https://github.com/SuryaAdiPalguna/Analisis_Sentimen_Tokoh_DWKNN/blob/main/Data/1_Pengumpulan_Data/train.xlsx)
- Setelah mengunggah file, klik tombol **Upload** dan tunggu proses selesai.
- Masukkan teks yang ingin diuji ke kolom yang tersedia dan klik **Proses**.
- Sistem akan menampilkan hasil sentimen: **positif**, **netral**, atau **negatif**.

### Halaman `About`

- Menyediakan informasi tentang aplikasi dan pengembangannya.

---

## 📚 Teknologi dan Library

- Python
- Streamlit
- scikit-learn
- pandas, numpy, nltk, regex
- imbalanced-learn
- mpstemmer (Bahasa Indonesia stemmer)
- Levenshtein distance

---

## 🧠 Tentang DWKNN

**Distance-Weighted K-Nearest Neighbor (DWKNN)** adalah varian dari algoritma KNN, di mana kontribusi tiap tetangga terhadap klasifikasi tidak dianggap sama rata, tetapi diberi bobot berdasarkan jaraknya terhadap titik data uji. Semakin dekat sebuah tetangga, semakin besar pengaruhnya dalam keputusan klasifikasi.

---

## 📬 Kontribusi

Kontribusi sangat diterima!
Silakan kirimkan **issue** atau **pull request** untuk menambahkan fitur, memperbaiki bug, atau meningkatkan dokumentasi.

---

## 📜 Lisensi

Proyek ini berada di bawah lisensi MIT. Lihat file [LICENSE](LICENSE) untuk informasi lebih lanjut.

---

## 👤 Pengembang

**I Made Surya Adi Palguna**
_Analisis Sentimen Tokoh Politik dengan Metode DWKNN_

---
