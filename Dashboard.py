import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load Data
day_df = pd.read_csv("day.csv")
hour_df = pd.read_csv("hour.csv")

# Konversi kolom tanggal
day_df['dteday'] = pd.to_datetime(day_df['dteday'])
hour_df['dteday'] = pd.to_datetime(hour_df['dteday'])

# Sidebar untuk Filter Data
st.sidebar.header("Filter Data")
tahun_options = ["All", 0, 1]
tahun = st.sidebar.multiselect("Pilih Tahun", options=tahun_options, default="All", 
                               format_func=lambda x: "Semua" if x == "All" else ("2011" if x == 0 else "2012"))

musim = st.sidebar.multiselect("Pilih Musim", options=[1, 2, 3, 4], default=[1, 2, 3, 4],
                               format_func=lambda x: {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}[x])

# Filter data berdasarkan pilihan user
if "All" in tahun or len(tahun) == 2:
    day_filtered = day_df[day_df['season'].isin(musim)]
    hour_filtered = hour_df
else:
    day_filtered = day_df[(day_df['yr'].isin(tahun)) & (day_df['season'].isin(musim))]
    hour_filtered = hour_df[hour_df['yr'].isin(tahun)]

# Dashboard Utama
st.title("Dashboard Analisis Bike Sharing")

# KPI Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Sewa Sepeda", f"{day_filtered['cnt'].sum():,}")
col2.metric("Rata-rata Sewa Harian", f"{day_filtered['cnt'].mean():,.2f}")
col3.metric("Persentase Casual vs Registered", f"{day_filtered['casual'].sum() / day_filtered['cnt'].sum() * 100:.2f}% Casual")

# Grafik Tren Penggunaan
st.subheader("Tren Penyewaan Sepeda")
fig = px.line(day_filtered, x="dteday", y="cnt", title="Jumlah Penyewaan Sepeda Per Hari")
st.plotly_chart(fig)

# Visualisasi Penggunaan Sepeda per Jam dengan Bar Chart
st.subheader("Rata-rata Penyewaan Sepeda per Jam")
avg_hourly = hour_filtered.groupby("hr")["cnt"].mean().reset_index()
fig = px.bar(avg_hourly, x="hr", y="cnt", title="Rata-rata Penyewaan Sepeda Berdasarkan Jam",
             labels={"hr": "Jam", "cnt": "Jumlah Penyewaan"}, color="cnt", color_continuous_scale="viridis")
st.plotly_chart(fig)

# Scatter Plot Pengaruh Cuaca
st.subheader("Pengaruh Cuaca terhadap Penyewaan")
fig = px.scatter(day_filtered, x="temp", y="cnt", color="weathersit", title="Hubungan Temperatur dan Penyewaan Sepeda")
st.plotly_chart(fig)

# Model Prediksi Sewa Sepeda
st.subheader("Prediksi Penyewaan Sepeda")
X = day_filtered[['temp', 'atemp', 'hum', 'windspeed']]
y = day_filtered['cnt']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
st.write(f"Skor Model (R²): {model.score(X_test, y_test):.2f}")

# Footer
st.sidebar.write("Data Source: Bike Sharing Dataset")
