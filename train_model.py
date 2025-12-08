import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import os

# Set seed agar hasil konsisten
np.random.seed(1234)

# --- KONFIGURASI ---
DATA_FILE = 'dataset.csv'
LOOK_BACK = 15      # Berapa hari ke belakang yang dilihat model untuk prediksi besok
TEST_SIZE = 30      # Jumlah data terakhir yang akan dites (diprediksi)
EPOCHS = 50         # Berapa kali model belajar (semakin lama semakin akurat, tapi lambat)
BATCH_SIZE = 16

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

def main():
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} belum ada. Jalankan convert_data.py dulu!")
        return

    print("1. Memuat Data...")
    df = pd.read_csv(DATA_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Mengambil nilai 'Close' (Jumlah Pemain)
    dataset = df['Close'].values.reshape(-1, 1).astype('float32')
    
    # Normalisasi data (skala 0-1) agar LSTM bekerja optimal
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset_scaled = scaler.fit_transform(dataset)

    # Membagi Data Training dan Testing
    # Data test diambil dari bagian paling akhir (waktu terbaru)
    train_size = len(dataset_scaled) - TEST_SIZE
    train, test = dataset_scaled[0:train_size, :], dataset_scaled[train_size - LOOK_BACK:, :]

    print(f"   Total data: {len(dataset)}")
    print(f"   Data Training: {len(train)}")
    print(f"   Data Testing: {TEST_SIZE}")

    # Membuat struktur data untuk LSTM (X=masa lalu, Y=masa depan)
    X_train, y_train = create_dataset(train, LOOK_BACK)
    X_test, y_test = create_dataset(test, LOOK_BACK)

    # Reshape input menjadi [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    print("\n2. Membangun Model LSTM...")
    model = Sequential()
    # Layer LSTM (mirip dengan repo asli: size_layer=128)
    model.add(LSTM(128, input_shape=(1, LOOK_BACK)))
    model.add(Dropout(0.2)) # Mencegah overfitting
    model.add(Dense(1))     # Output layer (1 nilai prediksi)
    
    model.compile(loss='mean_squared_error', optimizer='adam')

    print("\n3. Memulai Training (Tunggu sebentar)...")
    history = model.fit(
        X_train, y_train, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        verbose=1
    )

    print("\n4. Melakukan Prediksi...")
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Mengembalikan skala data ke aslinya (Jumlah Pemain Asli)
    train_predict = scaler.inverse_transform(train_predict)
    y_train_inv = scaler.inverse_transform([y_train])
    test_predict = scaler.inverse_transform(test_predict)
    y_test_inv = scaler.inverse_transform([y_test])
    original_data = scaler.inverse_transform(dataset_scaled)

    # Menghitung akurasi sederhana (MAPE - Mean Absolute Percentage Error)
    # Hati-hati pembagian dengan nol
    with np.errstate(divide='ignore', invalid='ignore'):
        test_score = np.mean(np.abs((y_test_inv[0] - test_predict[:,0]) / y_test_inv[0])) * 100
    print(f"   Rata-rata Error Prediksi (Test Set): {test_score:.2f}%")

    print("\n5. Menyimpan Hasil Grafik...")
    # Plotting
    plt.figure(figsize=(12, 6))
    
    # Plot Data Asli
    plt.plot(df['Date'], original_data, label='Data Asli', color='gray', alpha=0.5)
    
    # Plot Prediksi Training
    # (Perlu penyesuaian indeks waktu untuk plotting)
    train_dates = df['Date'].iloc[LOOK_BACK+1:len(train_predict)+LOOK_BACK+1]
    plt.plot(train_dates, train_predict[:,0], label='Prediksi Training')

    # Plot Prediksi Testing
    # Ambil tanggal sejumlah data hasil prediksi test, diambil dari urutan paling belakang
    test_dates = df['Date'].iloc[-len(test_predict):]
    
    plt.plot(test_dates, test_predict[:,0], label='Prediksi Masa Depan (Test)', color='red')

    plt.title(f'Prediksi Jumlah Pemain: ')
    plt.xlabel('Waktu')
    plt.ylabel('Jumlah Pemain')
    plt.legend()
    plt.grid(True)
    
    output_img = 'hasil_prediksi.png'
    plt.savefig(output_img)
    print(f"   Grafik disimpan sebagai '{output_img}'. Silakan buka file ini di VS Code.")

if __name__ == "__main__":
    main()