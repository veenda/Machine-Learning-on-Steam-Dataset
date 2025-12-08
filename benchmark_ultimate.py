import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import xgboost as xgb
from tqdm import tqdm

# --- IMPORTS SKLEARN (Standard & Advanced) ---
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.compose import TransformedTargetRegressor

# Linear & Regularized
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet, 
                                  HuberRegressor, TheilSenRegressor, PoissonRegressor, 
                                  TweedieRegressor, SGDRegressor, BayesianRidge)
# Support Vector
from sklearn.svm import SVR
# Tree Based
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# --- IMPORTS DEEP LEARNING ---
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, LSTM, GRU, SimpleRNN, 
                                     Bidirectional, Dropout, Conv1D, MaxPooling1D, 
                                     Flatten, RepeatVector, Concatenate, 
                                     GlobalAveragePooling1D, MultiHeadAttention, LayerNormalization)

# ==========================================
# KONFIGURASI
# ==========================================
DATA_FILE = 'dataset.csv'
TARGET_GAME = "Counter-Strike 2"
LOOK_BACK = 15      # Window size
TEST_SIZE = 30
EPOCHS = 15         # DL Epochs
BATCH_SIZE = 16

# ==========================================
# 1. PERSIAPAN DATA
# ==========================================
def get_data():
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} tidak ditemukan!")
        return None, None, None, None, None, None, None, None

    df = pd.read_csv(DATA_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Pastikan data positif untuk Poisson/Log models
    dataset = df['Close'].values.reshape(-1, 1).astype('float32')
    
    scaler = MinMaxScaler(feature_range=(0.01, 1)) # Hindari 0 pas untuk log
    dataset_scaled = scaler.fit_transform(dataset)

    train_size = len(dataset_scaled) - TEST_SIZE
    train, test = dataset_scaled[0:train_size, :], dataset_scaled[train_size - LOOK_BACK:, :]

    # Helper function untuk sliding window
    def create_dataset(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    X_train, y_train = create_dataset(train, LOOK_BACK)
    X_test, y_test = create_dataset(test, LOOK_BACK)

    # Reshape untuk Deep Learning [Samples, Time Steps, Features]
    X_train_dl = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test_dl = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    return df, scaler, X_train, y_train, X_test, y_test, X_train_dl, X_test_dl

# ==========================================
# 2. MODEL DEFINITIONS (Deep Learning)
# ==========================================
def build_dl_model(name, input_shape):
    inputs = Input(shape=input_shape)
    
    if name == 'Vanilla (RNN)':
        x = SimpleRNN(64)(inputs)
    elif name == 'LSTM':
        x = LSTM(64)(inputs)
    elif name == 'GRU':
        x = GRU(64)(inputs)
    elif name == 'LSTM Bidirectional':
        x = Bidirectional(LSTM(64))(inputs)
    elif name == 'CNN-Seq2seq':
        x = Conv1D(64, 2, activation='relu', padding='same')(inputs)
        x = Flatten()(x)
    elif name == 'Attention':
        att = MultiHeadAttention(num_heads=2, key_dim=2)(inputs, inputs)
        x = LayerNormalization()(inputs + att)
        x = GlobalAveragePooling1D()(x)
    else:
        x = LSTM(64)(inputs) # Default

    x = Dropout(0.2)(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

# ==========================================
# 3. MODEL DEFINITIONS (Machine Learning)
# ==========================================
def get_ml_models():
    models = {}
    
    # --- BASIC LINEAR ---
    models['Linear Regression'] = LinearRegression()
    models['Ridge Regression'] = Ridge(alpha=1.0)
    models['Lasso Regression'] = Lasso(alpha=0.1)
    models['Elastic Net'] = ElasticNet(alpha=0.1, l1_ratio=0.5)
    
    # --- NON-LINEAR / POLYNOMIAL ---
    # Quadratic Regression (Polynomial degree 2)
    models['Quadratic Regression'] = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    
    # --- DIMENSION REDUCTION ---
    # Principal Components Regression (PCR)
    models['PCR (Principal Comp)'] = make_pipeline(StandardScaler(), PCA(n_components='mle'), LinearRegression())
    # Partial Least Squares (PLS)
    models['PLS Regression'] = PLSRegression(n_components=5)

    # --- SUPPORT VECTOR ---
    models['SVR (Support Vector)'] = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)

    # --- ROBUST REGRESSION (Tahan Outlier) ---
    models['Huber Regression'] = HuberRegressor()
    
    # --- COUNT & PROBABILISTIC (GLM) ---
    # Poisson Regression (Cocok untuk count data seperti pemain game)
    models['Poisson Regression'] = PoissonRegressor()
    # Gamma / Negative Binomial proxy
    models['Gamma Regression'] = TweedieRegressor(power=2) 
    
    # --- QUANTILE REGRESSION ---
    # Menggunakan Gradient Boosting dengan loss quantile (Median)
    models['Quantile Regression'] = GradientBoostingRegressor(loss='quantile', alpha=0.5)

    # --- TREE BASED ---
    models['Decision Tree'] = DecisionTreeRegressor(max_depth=10)
    models['Random Forest'] = RandomForestRegressor(n_estimators=100)
    models['XGBoost'] = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)

    # --- TRANSFORMED TARGETS ---
    # Exponential Regression (Log transform target, fit Linear, exp output)
    models['Exponential Regression'] = TransformedTargetRegressor(
        regressor=LinearRegression(), func=np.log, inverse_func=np.exp
    )
    
    return models

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def main():
    print("Memyiapkan Data...")
    df, scaler, X_train, y_train, X_test, y_test, X_train_dl, X_test_dl = get_data()
    if df is None: return

    predictions = {}
    
    # --- A. RUN DEEP LEARNING ---
    dl_names = ['LSTM', 'LSTM Bidirectional', 'GRU', 'Vanilla (RNN)', 'CNN-Seq2seq', 'Attention']
    
    print(f"\n--- TRAINING DEEP LEARNING ({len(dl_names)} Models) ---")
    for name in tqdm(dl_names):
        try:
            model = build_dl_model(name, (1, LOOK_BACK))
            model.fit(X_train_dl, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
            pred = model.predict(X_test_dl)
            predictions[name] = scaler.inverse_transform(pred).flatten()
        except Exception as e:
            print(f"DL Model {name} Skip: {e}")

    # --- B. RUN MACHINE LEARNING ---
    ml_models = get_ml_models()
    print(f"\n--- TRAINING REGRESSION ML ({len(ml_models)} Models) ---")
    
    # ML Models butuh input 2D
    for name, model in tqdm(ml_models.items()):
        try:
            # Fit model
            model.fit(X_train, y_train)
            
            # Predict
            pred = model.predict(X_test)
            
            # Reshape & Inverse
            if pred.ndim == 1: pred = pred.reshape(-1, 1)
            pred_inv = scaler.inverse_transform(pred).flatten()
            predictions[name] = pred_inv
            
        except Exception as e:
            print(f"ML Model {name} Gagal: {e}")

    # ==========================================
    # 5. EVALUASI
    # ==========================================
    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    results = []
    for name, pred in predictions.items():
        # MAPE (Mean Absolute Percentage Error)
        # Menghindari pembagian nol
        mape = np.mean(np.abs((y_test_real - pred) / np.maximum(y_test_real, 1))) * 100
        results.append({'Model': name, 'MAPE (%)': mape})

    df_res = pd.DataFrame(results).sort_values('MAPE (%)')
    
    print("\n\nTOP 10 MODEL TERBAIK (Error Terkecil):")
    print(df_res.head(10))

    # --- PLOTTING ---
    test_dates = df['Date'].iloc[-len(y_test_real):]
    
    # 1. Bar Chart Error
    plt.figure(figsize=(12, 10))
    sns.barplot(x='MAPE (%)', y='Model', data=df_res, palette='magma')
    plt.title('Benchmark Error Semua Model (Lebih Pendek = Lebih Baik)')
    plt.tight_layout()
    plt.savefig('benchmark_ultimate_bar.png')
    
    # 2. Line Chart (Real vs Top 5)
    plt.figure(figsize=(15, 7))
    plt.plot(test_dates, y_test_real, label='DATA ASLI', color='black', linewidth=3, linestyle='--')
    
    top_models = df_res['Model'].head(5).tolist()
    for name in top_models:
        plt.plot(test_dates, predictions[name], label=name)
        
    plt.title(f'Prediksi vs Realita (Top 5 Models dari {len(predictions)} Model)')
    plt.legend()
    plt.grid(True)
    plt.savefig('benchmark_ultimate_line.png')

    print("\nSelesai! Cek 'benchmark_ultimate_bar.png' dan 'benchmark_ultimate_line.png'")

if __name__ == "__main__":
    main()