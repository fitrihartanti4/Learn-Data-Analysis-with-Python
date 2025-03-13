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

musim_mapping = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
musim = st.sidebar.multiselect("Pilih Musim", options=[1, 2, 3, 4], default=[1, 2, 3, 4], format_func=lambda x: musim_mapping[x])

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

# Grafik Penyewaan Sepeda per Musim
seasonal_data = day_filtered.groupby('season')['cnt'].sum().reset_index()
seasonal_data['Musim'] = seasonal_data['season'].map(musim_mapping)

st.subheader("Jumlah Penyewaan Sepeda per Musim")
fig = px.bar(seasonal_data, x="Musim", y="cnt", text_auto=True, color="Musim",
             labels={"cnt": "Jumlah Penyewaan Sepeda"},
             title="Total Penyewaan Sepeda Berdasarkan Musim")
st.plotly_chart(fig)

# Pie Chart Total Penyewaan Sepeda per Musim
st.subheader("Distribusi Penyewaan Sepeda per Musim")
fig, ax = plt.subplots(figsize=(6, 6))
ax.pie(seasonal_data['cnt'], labels=seasonal_data['Musim'], autopct='%1.1f%%', colors=['#FF9999', '#66B3FF', '#99FF99', '#FFCC99'])
ax.set_title("Persentase Penyewaan Sepeda Berdasarkan Musim")
st.pyplot(fig)


# Grafik Pengguna Kasual vs Terdaftar per Musim
seasonal_user_data = day_filtered.groupby('season')[['casual', 'registered']].sum().reset_index()
seasonal_user_data['Musim'] = seasonal_user_data['season'].map(musim_mapping)

st.subheader("Pengguna Kasual vs Terdaftar per Musim")
fig, ax = plt.subplots(figsize=(8, 6))
seasonal_user_data.set_index('Musim')[['casual', 'registered']].plot(kind='bar', stacked=True, color=['skyblue', 'orange'], ax=ax)
ax.set_ylabel("Jumlah Pengguna")
ax.set_title("Jumlah Pengguna Kasual dan Terdaftar per Musim")
st.pyplot(fig)

# Grafik Rata-rata Penggunaan Sepeda per Jam
st.subheader("Rata-rata Penyewaan Sepeda per Jam")
hourly_user_data = hour_filtered.groupby('hr')[['casual', 'registered']].mean().reset_index()
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(hourly_user_data['hr'], hourly_user_data['casual'], label='Casual', color='skyblue')
ax.plot(hourly_user_data['hr'], hourly_user_data['registered'], label='Registered', color='orange')
ax.set_xlabel("Jam")
ax.set_ylabel("Jumlah Pengguna")
ax.set_title("Rata-rata Penggunaan Sepeda per Jam")
ax.legend()
st.pyplot(fig)

# Scatter Plot Pengaruh Cuaca terhadap Penyewaan Sepeda
st.subheader("Pengaruh Cuaca terhadap Penyewaan")
fig = px.scatter(day_filtered, x="temp", y="cnt", color="weathersit", title="Hubungan Temperatur dan Penyewaan Sepeda")
st.plotly_chart(fig)

# Heatmap Korelasi
st.subheader("Heatmap Korelasi Fitur")
selected_features = ['temp', 'hum', 'windspeed', 'cnt']
correlation_matrix_filtered = day_filtered[selected_features].corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(correlation_matrix_filtered, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
st.pyplot(fig)


# Rata-rata Penyewaan Sepeda di Hari Kerja vs Hari Libur
workday_holiday_avg = day_df.groupby('holiday')['cnt'].mean().reset_index()
workday_holiday_avg['Kategori'] = workday_holiday_avg['holiday'].map({0: 'Hari Kerja', 1: 'Hari Libur'})

st.subheader("Rata-rata Penyewaan Sepeda: Hari Kerja vs Hari Libur")
fig = px.bar(workday_holiday_avg, x='Kategori', y='cnt', text_auto=True, color='Kategori',
             labels={'cnt': 'Rata-rata Penyewaan Sepeda'},
             title="Perbandingan Rata-rata Penyewaan Sepeda di Hari Kerja dan Hari Libur")
st.plotly_chart(fig)


# Grafik Tren Penyewaan Sepeda
st.subheader("Tren Penyewaan Sepeda")
fig = px.line(day_filtered, x="dteday", y="cnt", title="Jumlah Penyewaan Sepeda Per Hari")
st.plotly_chart(fig)

# Model Prediksi Sewa Sepeda
st.subheader("Prediksi Penyewaan Sepeda")
X = day_filtered[['temp', 'atemp', 'hum', 'windspeed']]
y = day_filtered['cnt']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
st.write(f"Skor Model (R²): {model.score(X_test, y_test):.2f}")

# Footer
st.sidebar.write("Data Source: Bike Sharing Dataset")