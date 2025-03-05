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

# Tambahkan opsi untuk menampilkan analisis setelah input atau tanpa input
if menu == "📊 Dashboard":
    show_analysis = st.sidebar.checkbox("🔍 Tampilkan Analisis Data", value=False)

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

if st.button("💾 Simpan Data"):
    new_data = pd.DataFrame([{ 
        "Usia": usia, 
        "Penghasilan": penghasilan, 
        "Rating": rating, 
        "Review": review, 
        "Frekuensi": frekuensi, 
        "Jenis_Kelamin": jenis_kelamin 
    }])

    if os.path.exists(data_file):
        existing_data = pd.read_csv(data_file)
        df = pd.concat([existing_data, new_data], ignore_index=True)
    else:
        df = new_data

    df.to_csv(data_file, index=False)

    # Aktifkan analisis setelah menyimpan data
    st.session_state["show_analysis"] = True  
    st.success("✅ Data berhasil disimpan! Pergi ke Dashboard dan aktifkan 'Tampilkan Analisis Data' untuk melihat hasil analisis.")

# ========================= 🔥 TAHAP 3: ANALISIS & VISUALISASI DATA ========================= #
if menu == "📊 Dashboard":
    st.markdown("<h1 class='big-font'>📊 Dashboard Analitik</h1>", unsafe_allow_html=True)
    
    if not show_analysis:
        st.warning("⚠️ Pilih 'Tampilkan Analisis Data' di sidebar untuk melihat hasil analisis.")
        st.stop()  # Hentikan eksekusi jika analisis tidak diaktifkan

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

       # 🔥 MULTI-LINE CHART (PENGGANTI GEOSPATIAL HEATMAP)
st.subheader("📈 Multi-Line Chart untuk Tren Data")

# Pastikan ada data untuk membuat banyak garis
if 'Frekuensi' not in df.columns:
    df['Frekuensi'] = np.random.randint(1, 30, len(df))

# Buat dummy data untuk tren
multi_line_df = df[['Frekuensi']].copy()
for i in range(5):  # Tambah 5 garis berbeda
    multi_line_df[f"Tren_{i+1}"] = df["Frekuensi"] + np.random.randint(-5, 5, len(df))

fig_multiline = px.line(multi_line_df, x=multi_line_df.index, y=multi_line_df.columns,
                        title="Multi-Line Chart")
st.plotly_chart(fig_multiline, use_container_width=True)

        # 🔥 DBSCAN CLUSTERING UNTUK SEGMENTASI PELANGGAN
st.subheader("📊 DBSCAN Clustering")

# Pastikan dataset memiliki kolom yang dibutuhkan, jika tidak ada, buat data random
required_columns = ['Usia', 'Penghasilan', 'Frekuensi']
for col in required_columns:
    if col not in df.columns:
        st.warning(f"⚠️ Kolom '{col}' tidak ditemukan, menggunakan nilai acak sebagai contoh.")
        df[col] = np.random.randint(18, 65, len(df)) if col == "Usia" else np.random.randint(1000, 10000, len(df))

# Hapus baris dengan NaN jika masih ada setelah inisialisasi
df_cluster = df.dropna(subset=required_columns).copy()

if not df_cluster.empty:
    # Standarisasi data sebelum clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_cluster[required_columns])

    # DBSCAN Clustering
    dbscan = DBSCAN(eps=0.5, min_samples=5).fit(X_scaled)
    df_cluster['Cluster'] = dbscan.labels_

    # Visualisasi hasil clustering
    fig_cluster = px.scatter(df_cluster, x="Usia", y="Penghasilan", 
                             color=df_cluster["Cluster"].astype(str),
                             title="Hasil Clustering DBSCAN",
                             color_discrete_sequence=px.colors.qualitative.Set1)
    st.plotly_chart(fig_cluster, use_container_width=True)

else:
    st.warning("⚠️ Tidak cukup data untuk clustering.")


    # 🔥 XGBOOST PREDICTION UNTUK ANALISIS TREN
st.subheader("📈 XGBoost Prediction")

# Pastikan data tersedia, jika tidak buat random
if 'Frekuensi' not in df.columns:
    df['Frekuensi'] = np.random.randint(1, 30, len(df))
if 'Usia' not in df.columns:
    df['Usia'] = np.random.randint(18, 65, len(df))
if 'Penghasilan' not in df.columns:
    df['Penghasilan'] = np.random.randint(1000, 10000, len(df))
if 'Rating' not in df.columns:
    df['Rating'] = np.random.randint(1, 5, len(df))

df['Target'] = df['Frekuensi'].shift(-1).fillna(df['Frekuensi'].mean())
X_train, y_train = df[['Usia', 'Penghasilan', 'Rating']], df['Target']

# Latih model XGBoost
model_xgb = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
model_xgb.fit(X_train, y_train)
df['Predicted_Frekuensi'] = model_xgb.predict(X_train)

# Visualisasi Prediksi
fig_pred = px.line(df, x=df.index, y=["Frekuensi", "Predicted_Frekuensi"],
                   title="Prediksi Frekuensi Belanja dengan XGBoost")
st.plotly_chart(fig_pred, use_container_width=True)

# 🔥 PCA + DBSCAN UNTUK CLUSTERING
st.subheader("🎯 PCA + DBSCAN Clustering")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['Usia', 'Penghasilan', 'Frekuensi']])
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

# 🔥 NETWORK ANALYSIS - CHORD DIAGRAM
st.subheader("🌐 Network Science - Chord Diagram")
G = nx.erdos_renyi_graph(20, 0.3)
edges = list(G.edges())
fig_network = px.scatter(x=[e[0] for e in edges], y=[e[1] for e in edges],
                         title="Network Chord Diagram", color_discrete_sequence=["#636EFA"])
st.plotly_chart(fig_network, use_container_width=True)

# 🔥 SCATTER DENGAN HEATMAP
st.subheader("🎨 Scatter + Heatmap")
fig_scatter = px.scatter(df, x="Usia", y="Penghasilan", color="Frekuensi",
                         title="Scatter Plot dengan Heatmap",
                         color_continuous_scale="Bluered")
st.plotly_chart(fig_scatter, use_container_width=True)

# 🔥 TERNARY PLOT
st.subheader("🔺 Ternary Plot")
fig_ternary = px.scatter_ternary(df, a="Usia", b="Penghasilan", c="Frekuensi",
                                 title="Ternary Plot untuk Segmen Pelanggan")
st.plotly_chart(fig_ternary, use_container_width=True)

# 🔥 NETWORK ANALYSIS - DEGREE, BETWEENNESS, CLOSENESS CENTRALITY
st.subheader("🌐 Network Science - Centrality Analysis")

# Buat graph acak
G = nx.erdos_renyi_graph(20, 0.3)

# Hitung centrality
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)
closeness_centrality = nx.closeness_centrality(G)

# Konversi ke DataFrame
centrality_df = pd.DataFrame({
    "Node": list(G.nodes()),
    "Degree Centrality": list(degree_centrality.values()),
    "Betweenness Centrality": list(betweenness_centrality.values()),
    "Closeness Centrality": list(closeness_centrality.values())
})

# Scatter Plot untuk Visualisasi Centrality
fig_network = px.scatter(centrality_df, x="Degree Centrality", y="Betweenness Centrality",
                         size="Closeness Centrality", hover_name="Node",
                         title="Network Analysis - Degree, Betweenness, Closeness Centrality",
                         color_discrete_sequence=["#636EFA"])
st.plotly_chart(fig_network, use_container_width=True)

# 🔥 SUNBURST CHART
st.subheader("🌞 Sunburst Chart")
fig_sunburst = px.sunburst(df, path=["Jenis_Kelamin", "Frekuensi"], values="Penghasilan",
                           title="Sunburst Chart - Jenis Kelamin vs Frekuensi Belanja")
st.plotly_chart(fig_sunburst, use_container_width=True)

# 🔥 GENERATE PDF REPORT
import io
import matplotlib.pyplot as plt
from fpdf import FPDF
import pandas as pd
import numpy as np

def generate_pdf():
    buffer = io.BytesIO()
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Gunakan font Unicode agar tidak error
    pdf.set_font("Arial", size=12)

    # 📝 Tambahkan Judul Laporan
    pdf.cell(200, 10, txt="Laporan Analisis Kuesioner", ln=True, align='C')
    pdf.ln(10)

    # 🔍 Pastikan Data Tidak Kosong, Jika Kosong Buat Dummy Data
    if df.empty or "Penghasilan" not in df.columns:
        df["Usia"] = np.random.randint(18, 65, 100)
        df["Penghasilan"] = np.random.randint(1000, 10000, 100)
        df["Frekuensi"] = np.random.randint(1, 30, 100)
        df["Rating"] = np.random.randint(1, 5, 100)

    # 📊 Tambahkan Ringkasan Data ke PDF
    pdf.cell(200, 10, txt=f"📊 Jumlah Responden: {len(df)}", ln=True)
    pdf.cell(200, 10, txt=f"💰 Rata-rata Penghasilan: ${df['Penghasilan'].mean():,.2f}", ln=True)
    pdf.cell(200, 10, txt=f"⭐ Rata-rata Rating: {df['Rating'].mean():.2f} / 5.0", ln=True)
    pdf.cell(200, 10, txt=f"🛍️ Rata-rata Frekuensi Belanja: {df['Frekuensi'].mean():.1f} kali/bulan", ln=True)
    pdf.ln(10)

    # 🏆 Tambahkan 5 Sampel Data ke PDF (Tabel)
    pdf.cell(200, 10, txt="📄 Contoh Data Responden:", ln=True)
    pdf.ln(5)
    
    pdf.set_font("DejaVu", size=10)
    pdf.cell(40, 10, "Usia", 1)
    pdf.cell(50, 10, "Penghasilan ($)", 1)
    pdf.cell(40, 10, "Frekuensi", 1)
    pdf.cell(30, 10, "Rating", 1)
    pdf.ln()

    for i in range(min(5, len(df))):  
        row = df.iloc[i]
        pdf.cell(40, 10, str(row["Usia"]), 1)
        pdf.cell(50, 10, f"${row['Penghasilan']:,.2f}", 1)
        pdf.cell(40, 10, str(row["Frekuensi"]), 1)
        pdf.cell(30, 10, str(row["Rating"]), 1)
        pdf.ln()

    pdf.ln(10)

    # 📊 BUAT CHART MATPLOTLIB & SIMPAN SEBAGAI GAMBAR
    fig, ax = plt.subplots(figsize=(5, 3))
    df.groupby("Rating")["Frekuensi"].mean().plot(kind="bar", ax=ax, color="skyblue")
    ax.set_title("Rata-rata Frekuensi Belanja Berdasarkan Rating")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Frekuensi")
    
    img_path = "chart.png"
    plt.savefig(img_path, format="png")  # Simpan gambar
    
    pdf.image(img_path, x=30, w=150)  # Tambahkan gambar ke PDF
    pdf.ln(10)

    pdf.cell(200, 10, txt="🔍 Analisis lebih lengkap tersedia di dashboard!", ln=True)

    # 🔥 Simpan PDF ke Buffer
    pdf.output(buffer)
    buffer.seek(0)  # Kembali ke awal buffer

    return buffer

# 🔥 Tombol untuk Generate PDF
if st.button("📥 Generate PDF Report"):
    pdf_file = generate_pdf()
    st.download_button(label="⬇️ Download Report PDF", data=pdf_file,
                       file_name="Laporan_Analisis.pdf", mime="application/pdf")

