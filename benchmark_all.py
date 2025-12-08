import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from tqdm import tqdm

# --- IMPORTS MACHINE LEARNING ---
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# --- IMPORTS DEEP LEARNING (TENSORFLOW/KERAS) ---
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Input, Dense, LSTM, GRU, SimpleRNN, 
                                     Bidirectional, Dropout, Conv1D, MaxPooling1D, 
                                     Flatten, RepeatVector, TimeDistributed, 
                                     GlobalAveragePooling1D, Concatenate, MultiHeadAttention, LayerNormalization)

# Konfigurasi
DATA_FILE = 'dataset.csv'
TARGET_GAME = "Counter-Strike 2" # Hanya untuk label grafik
LOOK_BACK = 15
TEST_SIZE = 30
EPOCHS = 15      # Dikurangi sedikit agar benchmark tidak terlalu lama
BATCH_SIZE = 16

# ==========================================
# 1. PERSIAPAN DATA
# ==========================================
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

def get_data():
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} tidak ditemukan. Jalankan convert_data.py dulu.")
        return None, None, None, None, None, None

    df = pd.read_csv(DATA_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    dataset = df['Close'].values.reshape(-1, 1).astype('float32')
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset_scaled = scaler.fit_transform(dataset)

    train_size = len(dataset_scaled) - TEST_SIZE
    train, test = dataset_scaled[0:train_size, :], dataset_scaled[train_size - LOOK_BACK:, :]

    X_train, y_train = create_dataset(train, LOOK_BACK)
    X_test, y_test = create_dataset(test, LOOK_BACK)

    # Reshape untuk Deep Learning [Samples, Time Steps, Features]
    X_train_dl = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test_dl = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    return df, scaler, X_train, y_train, X_test, y_test, X_train_dl, X_test_dl

# ==========================================
# 2. DEFINISI ARSITEKTUR MODEL
# ==========================================
def build_dl_model(name, input_shape):
    inputs = Input(shape=input_shape)
    
    # --- RNN FAMILY ---
    if name == 'Vanilla (RNN)':
        x = SimpleRNN(64, return_sequences=False)(inputs)
    elif name == 'Vanilla Bidirectional':
        x = Bidirectional(SimpleRNN(64, return_sequences=False))(inputs)
    elif name == 'LSTM':
        x = LSTM(64, return_sequences=False)(inputs)
    elif name == 'LSTM Bidirectional':
        x = Bidirectional(LSTM(64, return_sequences=False))(inputs)
    elif name == 'GRU':
        x = GRU(64, return_sequences=False)(inputs)
    elif name == 'GRU Bidirectional':
        x = Bidirectional(GRU(64, return_sequences=False))(inputs)
        
    # --- 2-PATH (ENSEMBLE LAYER) ---
    # Menggabungkan dua jenis layer berbeda
    elif name == 'LSTM 2-Path':
        path1 = LSTM(32, return_sequences=False)(inputs)
        path2 = LSTM(32, return_sequences=False, go_backwards=True)(inputs) # Reverse direction
        x = Concatenate()([path1, path2])
    elif name == 'GRU 2-Path':
        path1 = GRU(32, return_sequences=False)(inputs)
        path2 = GRU(32, return_sequences=False, go_backwards=True)(inputs)
        x = Concatenate()([path1, path2])
        
    # --- SEQ2SEQ ARCHITECTURES ---
    # Encoder -> Decoder structure adapted for single step
    elif 'Seq2seq' in name:
        if 'LSTM' in name:
            encoder = LSTM(64, return_sequences=False)(inputs)
        else: # GRU
            encoder = GRU(64, return_sequences=False)(inputs)
            
        x = RepeatVector(1)(encoder) # Menyiapkan konteks untuk decoder
        
        if 'Bidirectional' in name:
            if 'LSTM' in name: x = Bidirectional(LSTM(64, return_sequences=False))(x)
            else: x = Bidirectional(GRU(64, return_sequences=False))(x)
        else:
            if 'LSTM' in name: x = LSTM(64, return_sequences=False)(x)
            else: x = GRU(64, return_sequences=False)(x)

    # --- CNN FAMILY ---
    elif name == 'CNN-Seq2seq':
        x = Conv1D(filters=64, kernel_size=2, activation='relu', padding='same')(inputs)
        x = MaxPooling1D(pool_size=1)(x) # Dummy pool
        x = Flatten()(x)
    elif name == 'Dilated-CNN':
        x = Conv1D(filters=64, kernel_size=2, dilation_rate=2, activation='relu', padding='same')(inputs)
        x = GlobalAveragePooling1D()(x)

    # --- ATTENTION MECHANISM (Transformer-like) ---
    elif name == 'Attention-is-all-you-Need':
        # Simple implementation of Self-Attention
        att = MultiHeadAttention(num_heads=2, key_dim=2)(inputs, inputs)
        x = LayerNormalization()(inputs + att)
        x = GlobalAveragePooling1D()(x)

    else:
        # Default fallback
        x = LSTM(64)(inputs)

    x = Dropout(0.2)(x)
    outputs = Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# ==========================================
# 3. MAIN LOOP
# ==========================================
def main():
    # Load Data
    df, scaler, X_train, y_train, X_test, y_test, X_train_dl, X_test_dl = get_data()
    if df is None: return

    # List Model yang akan diuji
    dl_models = [
        'LSTM', 'LSTM Bidirectional', 'LSTM 2-Path', 
        'GRU', 'GRU Bidirectional', 'GRU 2-Path',
        'Vanilla (RNN)', 'Vanilla Bidirectional',
        'LSTM Seq2seq', 'GRU Seq2seq', 
        'CNN-Seq2seq', 'Dilated-CNN', 'Attention-is-all-you-Need'
    ]
    
    ml_models = ['Linear Regression', 'Random Forest', 'XGBoost']

    results = {}
    predictions = {}
    
    test_dates = df['Date'].iloc[-len(y_test):]
    
    print(f"--- MULAI BENCHMARK ({len(dl_models) + len(ml_models)} Models) ---")

    # --- 1. RUN DEEP LEARNING MODELS ---
    for name in tqdm(dl_models, desc="Training DL Models"):
        try:
            # Build & Train
            model = build_dl_model(name, (1, LOOK_BACK))
            model.fit(X_train_dl, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
            
            # Predict
            pred = model.predict(X_test_dl)
            pred_inv = scaler.inverse_transform(pred).flatten()
            
            predictions[name] = pred_inv
        except Exception as e:
            print(f"\nModel {name} Gagal: {e}")

    # --- 2. RUN MACHINE LEARNING MODELS ---
    # ML model butuh input 2D (Samples, Features), bukan 3D
    X_train_ml = X_train  
    X_test_ml = X_test

    for name in tqdm(ml_models, desc="Training ML Models"):
        try:
            if name == 'Linear Regression':
                model = LinearRegression()
            elif name == 'Random Forest':
                model = RandomForestRegressor(n_estimators=100)
            elif name == 'XGBoost':
                model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
            
            model.fit(X_train_ml, y_train)
            pred = model.predict(X_test_ml)
            
            # Inverse transform
            pred = pred.reshape(-1, 1)
            pred_inv = scaler.inverse_transform(pred).flatten()
            
            predictions[name] = pred_inv
        except Exception as e:
            print(f"\nModel {name} Gagal: {e}")

    # ==========================================
    # 4. EVALUASI & VISUALISASI
    # ==========================================
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # Hitung Error (MAPE)
    score_board = []
    for name, pred in predictions.items():
        # MAPE: Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_test_inv - pred) / y_test_inv)) * 100
        score_board.append({'Model': name, 'MAPE (%)': mape})
    
    df_scores = pd.DataFrame(score_board).sort_values('MAPE (%)')
    print("\n=== HASIL PERINGKAT (Error Terkecil ke Terbesar) ===")
    print(df_scores)

    # --- PLOT 1: GARIS PREDIKSI (Top 5 Terbaik + Asli) ---
    plt.figure(figsize=(14, 7))
    plt.plot(test_dates, y_test_inv, label='Data Asli (Actual)', color='black', linewidth=3, linestyle='--')
    
    # Ambil 5 model terbaik untuk di-plot agar tidak pusing lihatnya
    top_5_models = df_scores.head(5)['Model'].tolist()
    
    for name in top_5_models:
        plt.plot(test_dates, predictions[name], label=f"{name}")

    plt.title(f'Top 5 Model Terbaik: {TARGET_GAME}')
    plt.xlabel('Tanggal')
    plt.ylabel('Players')
    plt.legend()
    plt.grid(True)
    plt.savefig('benchmark_line_chart.png')

    # --- PLOT 2: BAR CHART ERROR ---
    plt.figure(figsize=(12, 8))
    sns.barplot(x='MAPE (%)', y='Model', data=df_scores, palette='viridis')
    plt.title('Perbandingan Error Model (Semakin Kecil Semakin Bagus)')
    plt.xlabel('Error (MAPE %)')
    plt.tight_layout()
    plt.savefig('benchmark_bar_chart.png')

    print("\nSelesai! Cek file 'benchmark_line_chart.png' dan 'benchmark_bar_chart.png'.")

if __name__ == "__main__":
    main()