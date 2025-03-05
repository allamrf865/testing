# ========================= 🔥 TAHAP 1: KONFIGURASI AWAL & SETUP ========================= #

import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import networkx as nx
import h3
from textblob import TextBlob
from fpdf import FPDF
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import tensorflow as tf
from scipy.fft import fft

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="Dashboard Kuesioner", layout="wide")

# Menampilkan judul aplikasi
st.markdown("<h1 style='text-align: center; color: #FF5733;'>📊 Dashboard Kuesioner</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #33FFBD;'>Dapatkan insight terbaik dari data kuesioner secara otomatis</h3>", unsafe_allow_html=True)

st.sidebar.title("🔍 Navigasi Aplikasi")
menu = st.sidebar.radio("Pilih Halaman:", ["🏠 Home", "📝 Isi Form", "📊 Dashboard"])

# File dataset
data_file = "data/Sample_Data_Kuesioner__1000_Data_.csv"

# Cek apakah data tersedia
if os.path.exists(data_file):
    df = pd.read_csv(data_file)
    st.success("✅ Data terbaru telah dimuat!")
else:
    df = pd.DataFrame()  # Jika tidak ada data, inisialisasi DataFrame kosong
    st.warning("⚠️ Data belum tersedia. Silakan isi kuesioner terlebih dahulu.")

st.markdown("---")
# ========================= 🔥 TAHAP 2: NAVIGASI & HALAMAN APLIKASI ========================= #

# 🎨 Kustomisasi gaya CSS untuk tampilan lebih menarik
st.markdown("""
    <style>
        .big-font { font-size:30px !important; text-align: center; color: #FF5733; }
        .medium-font { font-size:20px !important; text-align: center; color: #33FFBD; }
        .stButton>button { background-color: #FF5733; color: white; font-size: 18px; }
        .stMetric { text-align: center; font-size: 18px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ========================= 🏠 HOME PAGE ========================= #
if menu == "🏠 Home":
    st.markdown("<h1 class='big-font'>🏠 Selamat Datang di Dashboard Kuesioner</h1>", unsafe_allow_html=True)
    st.markdown("<h3 class='medium-font'>Dapatkan insight terbaik dari data kuesioner secara otomatis</h3>", unsafe_allow_html=True)
    
    # Animasi jika data tersedia
    if not df.empty:
        st.success("✅ Data terbaru telah dimuat!")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("📊 Jumlah Responden", len(df))
            st.metric("💰 Rata-rata Penghasilan", f"${df['Penghasilan'].mean():,.2f}")

        with col2:
            st.metric("⭐ Rata-rata Rating", f"{df['Rating'].mean():.2f} / 5.0")
            st.metric("🛍️ Frekuensi Belanja", f"{df['Frekuensi'].mean():.1f} kali/bulan")

    else:
        st.warning("⚠️ Data belum tersedia. Silakan isi kuesioner terlebih dahulu.")

# ========================= 📝 FORM KUESIONER ========================= #
if menu == "📝 Isi Form":
    st.markdown("<h1 class='big-font'>📝 Form Kuesioner</h1>", unsafe_allow_html=True)

    # 📌 Input Form dengan Komponen Interaktif
    usia = st.number_input("🧑 Usia", min_value=18, max_value=100, step=1)
    penghasilan = st.number_input("💰 Penghasilan (USD)", min_value=1000, max_value=100000, step=500)
    rating = st.slider("⭐ Rating Pelayanan (1 - 5)", min_value=1, max_value=5, step=1)
    review = st.text_area("📝 Review Singkat tentang Layanan", placeholder="Tulis pendapat Anda di sini...")
    frekuensi = st.slider("🛍️ Frekuensi Belanja dalam Sebulan", min_value=1, max_value=30, step=1)
    jenis_kelamin = st.radio("🚻 Jenis Kelamin", ["Pria", "Wanita"], index=0)

    # 🎯 Simpan Data ke CSV
    if st.button("💾 Simpan Data"):
        new_data = pd.DataFrame([{ 
            "Usia": usia, "Penghasilan": penghasilan, "Rating": rating, 
            "Review": review, "Frekuensi": frekuensi, "Jenis_Kelamin": jenis_kelamin 
        }])
        
        if os.path.exists(data_file):
            existing_data = pd.read_csv(data_file)
            df = pd.concat([existing_data, new_data], ignore_index=True)
        else:
            df = new_data
        
        df.to_csv(data_file, index=False)
        st.success("✅ Data berhasil disimpan! Klik ke Dashboard untuk melihat analisis terbaru.")
# ========================= 🔥 TAHAP 3: ANALISIS & VISUALISASI DATA ========================= #
if menu == "📊 Dashboard":
    st.markdown("<h1 class='big-font'>📊 Dashboard Analitik</h1>", unsafe_allow_html=True)

    if df.empty:
        st.warning("⚠️ Tidak ada data! Silakan isi kuesioner di halaman Form.")
    else:
        # 🔥 STATISTIK DESKRIPTIF
        st.subheader("📌 Statistik Deskriptif")
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("📊 Responden", len(df))
        with col2: st.metric("💰 Rata-rata Penghasilan", f"${df['Penghasilan'].mean():,.2f}")
        with col3: st.metric("⭐ Rata-rata Rating", f"{df['Rating'].mean():.2f} / 5.0")
        with col4: st.metric("🛍️ Frekuensi Belanja", f"{df['Frekuensi'].mean():.1f} kali/bulan")

        st.markdown("---")

        # 🔥 GEOSPATIAL ANALYSIS DENGAN H3 HEATMAP
st.subheader("🌍 Geospatial Heatmap")

# Pastikan Latitude & Longitude tersedia
if "Latitude" not in df.columns or "Longitude" not in df.columns:
    st.warning("⚠️ Data tidak memiliki koordinat, menggunakan nilai acak sebagai contoh.")
    df["Latitude"] = np.random.uniform(-90, 90, len(df))
    df["Longitude"] = np.random.uniform(-180, 180, len(df))

# Validasi nilai koordinat agar tidak ada NaN
df = df.dropna(subset=["Latitude", "Longitude"])
df["Latitude"] = df["Latitude"].astype(float)
df["Longitude"] = df["Longitude"].astype(float)

# Pastikan data tidak kosong setelah cleaning
if not df.empty:
    try:
        df["H3_Index"] = df.apply(
            lambda row: h3.geo_to_h3(row["Latitude"], row["Longitude"], 7), axis=1
        )
        
        fig_geo = px.density_mapbox(df, lat="Latitude", lon="Longitude", z="Rating",
                                    radius=10, mapbox_style="open-street-map",
                                    title="Geospatial Heatmap berdasarkan Rating")
        st.plotly_chart(fig_geo, use_container_width=True)
    
    except Exception as e:
        st.error(f"⚠️ Terjadi kesalahan dalam pemrosesan H3 Heatmap: {e}")

else:
    st.warning("⚠️ Tidak ada data yang tersedia untuk dianalisis.")


        # 🔥 DBSCAN CLUSTERING UNTUK SEGMENTASI PELANGGAN
        st.subheader("📊 DBSCAN Clustering")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[['Usia', 'Penghasilan', 'Frekuensi']])
        dbscan = DBSCAN(eps=0.5, min_samples=5).fit(X_scaled)
        df['Cluster'] = dbscan.labels_
        fig_cluster = px.scatter(df, x="Usia", y="Penghasilan", color=df["Cluster"].astype(str),
                                 title="Hasil Clustering DBSCAN", color_discrete_sequence=px.colors.qualitative.Set1)
        st.plotly_chart(fig_cluster, use_container_width=True)

        # 🔥 XGBOOST PREDICTION UNTUK ANALISIS TREN
        st.subheader("📈 XGBoost Prediction")
        df['Target'] = df['Frekuensi'].shift(-1).fillna(df['Frekuensi'].mean())
        X_train, y_train = df[['Usia', 'Penghasilan', 'Rating']], df['Target']
        model_xgb = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
        model_xgb.fit(X_train, y_train)
        df['Predicted_Frekuensi'] = model_xgb.predict(X_train)
        fig_pred = px.line(df, x=df.index, y=["Frekuensi", "Predicted_Frekuensi"],
                           title="Prediksi Frekuensi Belanja dengan XGBoost")
        st.plotly_chart(fig_pred, use_container_width=True)

        # 🔥 PCA + DBSCAN UNTUK CLUSTERING LEBIH AKURAT
        st.subheader("🎯 PCA + DBSCAN Clustering")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        df["PCA1"], df["PCA2"] = X_pca[:, 0], X_pca[:, 1]
        dbscan_pca = DBSCAN(eps=0.3, min_samples=5).fit(X_pca)
        df["PCA_Cluster"] = dbscan_pca.labels_
        fig_pca = px.scatter(df, x="PCA1", y="PCA2", color=df["PCA_Cluster"].astype(str),
                             title="PCA + DBSCAN Clustering", color_discrete_sequence=px.colors.qualitative.Dark2)
        st.plotly_chart(fig_pca, use_container_width=True)

        # 🔥 AUTOENCODER ANOMALY DETECTION
        st.subheader("🚨 Anomaly Detection dengan Autoencoder")
        autoencoder = tf.keras.models.Sequential([
            tf.keras.layers.Dense(16, activation="relu", input_shape=(X_scaled.shape[1],)),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(X_scaled.shape[1], activation="linear")
        ])
        autoencoder.compile(optimizer="adam", loss="mse")
        autoencoder.fit(X_scaled, X_scaled, epochs=10, batch_size=16, verbose=0)
        df["Reconstruction_Error"] = np.mean((X_scaled - autoencoder.predict(X_scaled))**2, axis=1)
        fig_anomaly = px.histogram(df, x="Reconstruction_Error", title="Distribusi Anomali")
        st.plotly_chart(fig_anomaly, use_container_width=True)

        # 🔥 FOURIER TRANSFORM TIME SERIES
        st.subheader("⏳ Fourier Transform Time Series")
        signal_fft = fft(df["Frekuensi"])
        fig_fft = px.line(y=np.abs(signal_fft), title="FFT Transform dari Frekuensi Belanja")
        st.plotly_chart(fig_fft, use_container_width=True)

        # 🔥 NETWORK ANALYSIS - GRAPH CENTRALITY
        st.subheader("🌐 Network Science - Graph Centrality")
        G = nx.erdos_renyi_graph(50, 0.2)
        centrality = nx.betweenness_centrality(G)
        fig_network = px.bar(x=list(centrality.keys()), y=list(centrality.values()), title="Network Centrality")
        st.plotly_chart(fig_network, use_container_width=True)

        # 🔥 GENERATE PDF REPORT
        st.subheader("📄 Generate Report PDF")
        def generate_pdf():
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="🚀 Laporan Analisis Kuesioner", ln=True, align='C')
            pdf.cell(200, 10, txt=f"Rata-rata Penghasilan: ${df['Penghasilan'].mean():.2f}", ln=True)
            pdf.cell(200, 10, txt=f"Quantum Measurement Result: {quantum_counts}", ln=True)
            pdf.output("Laporan_Analisis.pdf")
            st.success("✅ Laporan PDF Berhasil Dibuat! 📄")

        if st.button("📥 Generate PDF Report"):
            generate_pdf()
            st.download_button(label="⬇️ Download Report PDF", data=open("Laporan_Analisis.pdf", "rb"), file_name="Laporan_Analisis.pdf", mime="application/pdf")

st.markdown("---")
