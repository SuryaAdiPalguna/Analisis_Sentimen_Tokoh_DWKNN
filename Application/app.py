import streamlit as st
import pandas as pd
from utils import train_dwknn_model, predict_sentiment
from Modul._5_implementasi_algoritma import DWKNN

# model = DWKNN(k=5)
# X_train = None
# y_train = None

st.set_page_config(page_title="DISTANCE-WEIGHTED K-NEAREST NEIGHBOR", layout="centered")

page = st.sidebar.radio("", ["Home", "About"])

if "data_uploaded" not in st.session_state:
  st.session_state.data_uploaded = False

if page == "Home":
  user_input = st.text_input("Masukkan Ulasan")
  button_input = st.button("Proses")

  if button_input:
    if not st.session_state.data_uploaded:
      st.warning("Silakan upload dataset terlebih dahulu")
    elif user_input.strip() == "":
      st.warning("Mohon masukkan kalimat terlebih dahulu.")
    else:
      prediction = predict_sentiment(user_input)
      prediction = 'Positif' if prediction == 1 else 'Negatif' if prediction == -1 else 'Netral'
      st.success(f"Prediksi sentimen: **{prediction}**")

  uploaded_file = st.file_uploader("Pilih Dataset", type=["xlsx"])
  uploaded_button = st.button("Upload")

  if uploaded_button and uploaded_file is not None:
    try:
      df = pd.read_excel(uploaded_file, usecols=['full_text', 'sentiment'])
      if "full_text" in df.columns and "sentiment" in df.columns:
        st.session_state.data_uploaded = True
        train_dwknn_model(df)
        st.success("✅ Model berhasil dilatih dengan data yang diunggah!")
        st.write("Contoh data:", df.head())
        st.write(df.describe())
        st.write(df['sentiment'].value_counts())
      else:
        st.error("❌ Kolom harus bernama `text` dan `label`.")
    except Exception as e:
      st.error(f"Gagal memproses file: {e}")

elif page == "About":
  st.title("ANALISIS SENTIMEN TWITTER MENGENAI PENGARUH TOKOH POLITIK DENGAN METODE DISTANCE-WEIGHTED K-NEAREST NEIGHBOR")
  st.write("**Oleh: I Made Surya Adi Palguna (2108561067)**")
  st.write("Platform media sosial seperti Twitter menjadi tempat untuk menggali opini publik mengenai tokoh-tokoh politik dan isu-isu terkait, diskusi, hingga sentimen politik. Salah satu metode yang digunakan dengan beberapa metode pemungutan suara berbobot, salah satunya dengan menggunakan DWKNN. Distance-Weighted K-Nearest Neighbor (DWKNN) adalah algoritma yang didasarkan pada KNN yang setiap training set ditetapkan nilai kedekatannya terhadap query set, lalu diambil k training set terdekatnya, lalu setiap k training set tersebut ditetapkan nilai bobot yang berbeda pada k tetangga terdekat sesuai dengan jaraknya, hingga kemudian hasilnya ditentukan oleh label atau kelas yang mempunyai total bobot terbesar dari k tetangga terdekat. Dari penelitian ini, algoritma Distance-Weighted K-Nearest Neighbor pada klasifikasi sentimen optimal pada perlakuan baseline, fitur TF – IDF, dan nilai k = 35, dengan hasil akurasi sebesar 70%, precision sebesar 54%, recall sebesar 50%, dan f1-score tertinggi sebesar 50%.")
  st.write("Kata kunci: Analisis Sentimen, Twitter, Tokoh Politik, Distance-Weighted K-Nearest Neighbor, K-Nearest Neighbor")
